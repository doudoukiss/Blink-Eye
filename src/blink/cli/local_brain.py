"""Local identity and lightweight memory support for Blink's browser and voice runtimes."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from blink.frames.frames import Frame, LLMContextFrame
from blink.processors.frame_processor import FrameDirection, FrameProcessor
from blink.project_identity import PROJECT_IDENTITY, cache_dir, local_env_name
from blink.transcriptions.language import Language

if TYPE_CHECKING:
    from blink.processors.aggregators.llm_context import LLMContext

MEMORY_SCHEMA_VERSION = "blink_local_brain_memory/v1"
DEFAULT_MEMORY_FACT_LIMIT = 8
DEFAULT_MEMORY_PATH = cache_dir("local_brain", "memory.json")


def _is_chinese_language(language: Language) -> bool:
    normalized = language.value.lower()
    return normalized.startswith("zh") or normalized.startswith("cmn")


@dataclass(frozen=True)
class LocalBrainMemoryFact:
    """One remembered local-user fact."""

    category: str
    statement: str
    updated_at: str


class LocalBrainMemoryStore:
    """Small on-device fact store for the everyday local Blink runtime."""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        max_facts: int = DEFAULT_MEMORY_FACT_LIMIT,
    ):
        """Initialize the local memory store.

        Args:
            path: Optional override path for the persisted JSON store.
            max_facts: Maximum number of remembered facts to keep.
        """
        self._path = Path(path) if path else DEFAULT_MEMORY_PATH
        self._max_facts = max(1, int(max_facts))

    @property
    def path(self) -> Path:
        """Return the backing JSON path."""
        return self._path

    def facts(self) -> list[LocalBrainMemoryFact]:
        """Load the current memory facts."""
        if not self._path.exists():
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if payload.get("schema_version") != MEMORY_SCHEMA_VERSION:
            return []
        facts = payload.get("facts", [])
        return [
            LocalBrainMemoryFact(
                category=str(item.get("category", "")),
                statement=str(item.get("statement", "")),
                updated_at=str(item.get("updated_at", "")),
            )
            for item in facts
            if item.get("category") and item.get("statement")
        ]

    def remember_from_text(self, text: str) -> list[LocalBrainMemoryFact]:
        """Extract durable user facts from one utterance and persist them.

        Args:
            text: Latest user text from the local conversation.

        Returns:
            The updated fact list after any changes.
        """
        candidates = self._extract_candidate_facts(text)
        if not candidates:
            return self.facts()

        facts = self.facts()
        by_key = {(fact.category, fact.statement): fact for fact in facts}
        by_category = {fact.category: fact for fact in facts}
        updated_at = datetime.now(UTC).isoformat()

        for category, statement in candidates:
            fact = LocalBrainMemoryFact(
                category=category,
                statement=statement,
                updated_at=updated_at,
            )
            if category in {"user_name", "user_role", "user_origin"}:
                previous = by_category.get(category)
                if previous:
                    by_key.pop((previous.category, previous.statement), None)
                by_category[category] = fact
            by_key[(category, statement)] = fact

        merged = sorted(by_key.values(), key=lambda item: item.updated_at, reverse=True)
        merged = merged[: self._max_facts]
        self._write(merged)
        return merged

    def prompt_summary(self, language: Language) -> str:
        """Return a short system-prompt summary of remembered user facts."""
        facts = self.facts()
        if not facts:
            return ""

        statements = [fact.statement for fact in facts[: self._max_facts]]
        if _is_chinese_language(language):
            joined = "；".join(statements)
            return (
                "你有一份本地记忆，可在相关时自然使用这些信息："
                f"{joined}。如果用户纠正这些信息，以最新说法为准。"
            )

        joined = "; ".join(statements)
        return (
            "You have local remembered user facts that you may use naturally when relevant: "
            f"{joined}. If the user corrects any of them, treat the latest statement as canonical."
        )

    def _write(self, facts: list[LocalBrainMemoryFact]) -> None:
        """Persist the memory facts to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "facts": [asdict(fact) for fact in facts],
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _extract_candidate_facts(self, text: str) -> list[tuple[str, str]]:
        """Extract a small set of stable user facts from natural language."""
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if not normalized:
            return []

        patterns = (
            (
                "user_name",
                [
                    r"我叫([A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,23})",
                    r"我的名字是([A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,23})",
                    r"请叫我([A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,23})",
                    r"\bmy name is ([A-Za-z][A-Za-z \-]{0,30})",
                    r"\bcall me ([A-Za-z][A-Za-z \-]{0,30})",
                ],
            ),
            (
                "user_role",
                [
                    r"我是([^，。！？,.!?]{1,24})",
                    r"\bI am an? ([A-Za-z][A-Za-z \-]{0,30})",
                    r"\bI'm an? ([A-Za-z][A-Za-z \-]{0,30})",
                ],
            ),
            (
                "user_origin",
                [
                    r"我来自([^，。！？,.!?]{1,24})",
                    r"\bI am from ([A-Za-z][A-Za-z ,\-]{0,30})",
                    r"\bI live in ([A-Za-z][A-Za-z ,\-]{0,30})",
                ],
            ),
            (
                "user_like",
                [
                    r"我喜欢([^，。！？,.!?]{1,28})",
                    r"\bI like ([A-Za-z0-9][A-Za-z0-9 ,\-]{0,30})",
                    r"\bI love ([A-Za-z0-9][A-Za-z0-9 ,\-]{0,30})",
                ],
            ),
            (
                "user_dislike",
                [
                    r"我不喜欢([^，。！？,.!?]{1,28})",
                    r"\bI do not like ([A-Za-z0-9][A-Za-z0-9 ,\-]{0,30})",
                    r"\bI don't like ([A-Za-z0-9][A-Za-z0-9 ,\-]{0,30})",
                ],
            ),
        )

        candidates: list[tuple[str, str]] = []
        for category, regexes in patterns:
            for pattern in regexes:
                match = re.search(pattern, normalized, flags=re.IGNORECASE)
                if not match:
                    continue
                value = re.sub(r"\s+", " ", match.group(1).strip(" ，。！？,.!?"))
                if not value:
                    continue
                if category == "user_name":
                    statement = f"用户名字是 {value}" if re.search(r"[\u4e00-\u9fff]", value) else f"User name is {value}"
                elif category == "user_role":
                    statement = f"用户身份是 {value}" if re.search(r"[\u4e00-\u9fff]", value) else f"User role is {value}"
                elif category == "user_origin":
                    statement = f"用户来自 {value}" if re.search(r"[\u4e00-\u9fff]", value) else f"User is from {value}"
                elif category == "user_like":
                    statement = f"用户喜欢 {value}" if re.search(r"[\u4e00-\u9fff]", value) else f"User likes {value}"
                else:
                    statement = (
                        f"用户不喜欢 {value}"
                        if re.search(r"[\u4e00-\u9fff]", value)
                        else f"User does not like {value}"
                    )
                candidates.append((category, statement))
                break
        return candidates


class LocalBrainMemoryProcessor(FrameProcessor):
    """Persist lightweight user facts whenever a new local user turn reaches the LLM."""

    def __init__(
        self,
        *,
        memory_store: LocalBrainMemoryStore,
    ):
        """Initialize the memory processor.

        Args:
            memory_store: Persistent store used for lightweight local facts.
        """
        super().__init__(name="local-brain-memory")
        self._memory_store = memory_store
        self._last_seen_user_text: Optional[str] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Update memory from user context frames and pass all frames through."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            latest_user_text = latest_user_text_from_context(frame.context)
            if latest_user_text and latest_user_text != self._last_seen_user_text:
                self._memory_store.remember_from_text(latest_user_text)
                self._last_seen_user_text = latest_user_text

        await self.push_frame(frame, direction)


def latest_user_text_from_context(context: LLMContext) -> str:
    """Return the latest plain-text user message in a context."""
    for message in reversed(context.get_messages()):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue

        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        parts.append(text)
            if parts:
                return " ".join(parts)

    return ""


def build_local_brain_prompt(
    base_prompt: str,
    *,
    language: Language,
    robot_head_enabled: bool,
    memory_store: LocalBrainMemoryStore | None = None,
) -> str:
    """Append explicit Blink identity and remembered facts to a local system prompt."""
    if _is_chinese_language(language):
        identity_prompt = (
            "你不是外部代班助手，你就是 Blink。"
            "Blink 是当前对话主体，也是这套系统的本地软件大脑。"
            f"{'当前连接的机器人头是你的物理身体。' if robot_head_enabled else '如果机器人头连接上，它会是你的物理身体。'}"
            "当用户问你是谁时，请直接回答你是 Blink。"
            "对用户明确告诉你的身份信息要保持一致，若用户更正，以最新说法为准。"
        )
    else:
        identity_prompt = (
            f"You are {PROJECT_IDENTITY.display_name}, not an external helper acting on its behalf. "
            f"{PROJECT_IDENTITY.display_name} is the conversation subject and the local software brain. "
            f"{'The connected robot head is your physical embodiment. ' if robot_head_enabled else 'If the robot head is connected, it becomes your physical embodiment. '}"
            "If the user asks who you are, answer that you are Blink. "
            "Keep user-provided identity facts consistent, and treat the latest correction as canonical."
        )

    parts = [base_prompt.strip(), identity_prompt.strip()]
    if memory_store is not None:
        summary = memory_store.prompt_summary(language)
        if summary:
            parts.append(summary.strip())
    return " ".join(part for part in parts if part)


def resolve_local_brain_memory_store() -> LocalBrainMemoryStore | None:
    """Resolve the configured local memory store from Blink env vars."""
    disabled = os.getenv(local_env_name("BRAIN_MEMORY_DISABLED"), "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return None

    path = os.getenv(local_env_name("BRAIN_MEMORY_PATH"))
    return LocalBrainMemoryStore(path=path)
