"""Semantic memory records and extraction helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from blink.transcriptions.language import Language


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


@dataclass(frozen=True)
class BrainFactCandidate:
    """One typed semantic memory candidate."""

    namespace: str
    subject: str
    rendered_text: str
    value: dict[str, str]
    confidence: float
    singleton: bool


@dataclass(frozen=True)
class BrainSemanticMemoryRecord:
    """One canonical semantic memory row."""

    id: int
    user_id: str
    namespace: str
    subject: str
    value_json: str
    rendered_text: str
    confidence: float
    status: str
    source_event_id: str | None
    source_episode_id: int | None
    provenance_json: str
    contradiction_key: str | None
    supersedes_memory_id: int | None
    observed_at: str
    updated_at: str
    stale_after_seconds: int | None

    @property
    def value(self) -> dict[str, Any]:
        """Return the decoded semantic payload."""
        return json.loads(self.value_json)

    @property
    def provenance(self) -> dict[str, Any]:
        """Return the decoded provenance metadata."""
        return json.loads(self.provenance_json)

    @property
    def is_stale(self) -> bool:
        """Return whether the memory is older than its freshness window."""
        if self.stale_after_seconds in (None, 0):
            return False
        observed = datetime.fromisoformat(self.observed_at)
        return datetime.now(UTC) > observed + timedelta(seconds=int(self.stale_after_seconds))


def render_profile_fact(namespace: str, value: str) -> str:
    """Render a profile fact into prompt-safe text."""
    if namespace == "profile.name":
        return f"用户名字是 {value}" if _contains_cjk(value) else f"User name is {value}"
    if namespace == "profile.role":
        return f"用户身份是 {value}" if _contains_cjk(value) else f"User role is {value}"
    if namespace == "profile.origin":
        return f"用户来自 {value}" if _contains_cjk(value) else f"User is from {value}"
    raise ValueError(f"Unsupported profile namespace: {namespace}")


def render_preference_fact(namespace: str, value: str) -> str:
    """Render a preference fact into prompt-safe text."""
    if namespace == "preference.like":
        return f"用户喜欢 {value}" if _contains_cjk(value) else f"User likes {value}"
    if namespace == "preference.dislike":
        return f"用户不喜欢 {value}" if _contains_cjk(value) else f"User does not like {value}"
    raise ValueError(f"Unsupported preference namespace: {namespace}")


def semantic_default_staleness(namespace: str) -> int | None:
    """Return the default freshness horizon for one semantic namespace."""
    if namespace == "profile.name":
        return 365 * 24 * 60 * 60
    if namespace in {"profile.role", "profile.origin"}:
        return 180 * 24 * 60 * 60
    if namespace.startswith("preference."):
        return 120 * 24 * 60 * 60
    if namespace == "session.summary":
        return 7 * 24 * 60 * 60
    return 30 * 24 * 60 * 60


def semantic_contradiction_key(namespace: str, subject: str, singleton: bool) -> str | None:
    """Return the contradiction/supersession key for one semantic record."""
    normalized_subject = " ".join((subject or "").split()).strip().lower()
    if singleton:
        return f"{namespace}:{normalized_subject or 'user'}"
    if namespace in {"preference.like", "preference.dislike"}:
        return f"preference:{normalized_subject}"
    return None


def extract_memory_candidates(text: str) -> list[BrainFactCandidate]:
    """Extract typed semantic memory candidates from one user utterance."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return []

    patterns = (
        (
            "profile.name",
            True,
            [
                r"我叫([A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,23})",
                r"我的名字是([A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,23})",
                r"请叫我([A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,23})",
                r"\bmy name is ([A-Za-z][A-Za-z \-]{0,30})",
                r"\bcall me ([A-Za-z][A-Za-z \-]{0,30})",
            ],
        ),
        (
            "profile.role",
            True,
            [
                r"我是([^，。！？,.!?]{1,24})",
                r"\bI am an? ([A-Za-z][A-Za-z \-]{0,30})",
                r"\bI'm an? ([A-Za-z][A-Za-z \-]{0,30})",
            ],
        ),
        (
            "profile.origin",
            True,
            [
                r"我来自([^，。！？,.!?]{1,24})",
                r"\bI am from ([A-Za-z][A-Za-z ,\-]{0,30})",
                r"\bI live in ([A-Za-z][A-Za-z ,\-]{0,30})",
            ],
        ),
        (
            "preference.like",
            False,
            [
                r"我喜欢([^，。！？,.!?]{1,28})",
                r"\bI like ([A-Za-z0-9][A-Za-z0-9 ,\-]{0,30})",
                r"\bI love ([A-Za-z0-9][A-Za-z0-9 ,\-]{0,30})",
            ],
        ),
        (
            "preference.dislike",
            False,
            [
                r"我不喜欢([^，。！？,.!?]{1,28})",
                r"\bI do not like ([A-Za-z0-9][A-Za-z0-9 ,\-]{0,30})",
                r"\bI don't like ([A-Za-z0-9][A-Za-z0-9 ,\-]{0,30})",
            ],
        ),
    )

    candidates: list[BrainFactCandidate] = []
    for namespace, singleton, regexes in patterns:
        for pattern in regexes:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            value = re.sub(r"\s+", " ", match.group(1).strip(" ，。！？,.!?"))
            if not value:
                continue
            if namespace.startswith("profile."):
                rendered = render_profile_fact(namespace, value)
                subject = "user"
            elif namespace == "preference.like":
                rendered = render_preference_fact(namespace, value)
                subject = value.lower()
            else:
                rendered = render_preference_fact(namespace, value)
                subject = value.lower()

            candidates.append(
                BrainFactCandidate(
                    namespace=namespace,
                    subject=subject,
                    rendered_text=rendered,
                    value={"value": value},
                    confidence=0.72 if singleton else 0.64,
                    singleton=singleton,
                )
            )
            break

    return candidates


def build_user_profile_summary(facts: list, language: Language) -> str:
    """Render a prompt-safe user profile summary from typed facts."""
    if not facts:
        return "无" if language.value.lower().startswith(("zh", "cmn")) else "None"
    rendered = [fact.rendered_text for fact in facts]
    if language.value.lower().startswith(("zh", "cmn")):
        return "；".join(rendered)
    return "; ".join(rendered)
