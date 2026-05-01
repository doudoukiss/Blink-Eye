"""Typed memory extraction and consolidation for Blink brain runtimes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from blink.adapters.schemas.function_schema import FunctionSchema
from blink.adapters.schemas.tools_schema import ToolsSchema
from blink.brain.events import BrainEventType
from blink.brain.memory_layers.episodic import build_episodic_candidates_from_event
from blink.brain.memory_layers.exports import BrainMemoryExporter
from blink.brain.memory_layers.narrative import (
    BrainTaskCandidate,
    build_thread_summary_text,
    extract_task_candidates,
)
from blink.brain.memory_layers.retrieval import (
    BrainMemoryEmbeddingProvider,
    BrainMemoryQuery,
    BrainMemoryRetriever,
    BrainMemorySearchResult,
)
from blink.brain.memory_layers.semantic import (
    BrainFactCandidate,
    build_user_profile_summary,
    extract_memory_candidates,
    render_preference_fact,
    render_profile_fact,
)
from blink.brain.memory_layers.working import (
    BrainWorkingMemorySnapshot,
    build_working_memory_snapshot,
)
from blink.brain.memory_v2 import (
    BrainContinuityQuery,
    ContinuityRetriever,
    apply_memory_governance_action,
    build_memory_continuity_trace,
    build_memory_palace_snapshot,
)
from blink.brain.session import BrainSessionIds
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language

if TYPE_CHECKING:
    from blink.function_calling import FunctionCallParams

__all__ = [
    "BrainConsolidationResult",
    "BrainContinuityQuery",
    "BrainFactCandidate",
    "BrainMemoryConsolidator",
    "BrainMemoryEmbeddingProvider",
    "BrainMemoryQuery",
    "BrainMemoryRetriever",
    "BrainMemorySearchResult",
    "BrainTaskCandidate",
    "BrainWorkingMemorySnapshot",
    "ContinuityRetriever",
    "build_user_profile_summary",
    "build_working_memory_snapshot",
    "extract_memory_candidates",
    "extract_task_candidates",
    "apply_memory_governance_action",
    "memory_tool_prompt",
    "register_memory_tools",
]


@dataclass(frozen=True)
class BrainConsolidationResult:
    """Result of one incremental memory consolidation pass."""

    promoted_facts: int
    upserted_tasks: int
    latest_episode_id: int | None
    summary: str
    latest_event_id: int | None = None


def memory_tool_prompt(language: Language) -> str:
    """Return prompt guidance for bounded explicit memory tools."""
    if language.value.lower().startswith(("zh", "cmn")):
        return (
            "当用户明确要求你记住、纠正、忘记个人信息或待办事项时，"
            "只能使用受限的 brain_* 记忆工具。"
            "只保存稳定的个人资料、偏好和待办，不要记录闲聊、推测或敏感原始日志。"
            "当用户问你现在记得什么时，只能给出可见记忆的简短摘要。"
            "当用户问你为什么这样回答时，只能用公开的记忆连续性和风格摘要解释。"
        )
    return (
        "When the user explicitly asks you to remember, correct, forget, or complete personal facts "
        "or tasks, only use the bounded brain_* memory tools. "
        "Store only durable profile facts, preferences, and tasks, not incidental chatter or raw logs. "
        "When the user asks what you remember, answer only with a short visible-memory summary. "
        "When the user asks why you answered that way, explain only from public memory-continuity "
        "and style summaries."
    )


def register_memory_tools(
    *,
    llm,
    store: BrainStore,
    session_resolver: Callable[[], BrainSessionIds],
    language: Language,
) -> ToolsSchema:
    """Register bounded explicit memory tools for the local Blink brain."""
    profile_fields = {
        "name": "profile.name",
        "role": "profile.role",
        "origin": "profile.origin",
    }
    preference_kinds = {
        "like": "preference.like",
        "dislike": "preference.dislike",
    }

    async def brain_remember_profile(params: FunctionCallParams):
        field = str(params.arguments.get("field", "")).strip()
        value = str(params.arguments.get("value", "")).strip()
        namespace = profile_fields.get(field)
        if namespace is None or not value:
            await params.result_callback(
                {
                    "accepted": False,
                    "error": "invalid_profile_memory_request",
                    "supported_fields": sorted(profile_fields),
                }
            )
            return

        session_ids = session_resolver()
        rendered_text = render_profile_fact(namespace, value)
        fact = store.remember_fact(
            user_id=session_ids.user_id,
            namespace=namespace,
            subject="user",
            value={"value": value},
            rendered_text=rendered_text,
            confidence=0.92,
            singleton=True,
            provenance={"source": "tool", "tool_name": "brain_remember_profile"},
            source_episode_id=None,
            agent_id=session_ids.agent_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
        )
        if field == "name":
            store.ensure_user(user_id=session_ids.user_id, display_name=value)

        await params.result_callback(
            {
                "accepted": True,
                "memory_kind": namespace,
                "rendered_text": fact.rendered_text,
            }
        )

    async def brain_remember_preference(params: FunctionCallParams):
        sentiment = str(params.arguments.get("sentiment", "")).strip()
        topic = str(params.arguments.get("topic", "")).strip()
        namespace = preference_kinds.get(sentiment)
        if namespace is None or not topic:
            await params.result_callback(
                {
                    "accepted": False,
                    "error": "invalid_preference_memory_request",
                    "supported_sentiments": sorted(preference_kinds),
                }
            )
            return

        session_ids = session_resolver()
        normalized_topic = " ".join(topic.split()).strip()
        fact = store.remember_fact(
            user_id=session_ids.user_id,
            namespace=namespace,
            subject=normalized_topic.lower(),
            value={"value": normalized_topic},
            rendered_text=render_preference_fact(namespace, normalized_topic),
            confidence=0.86,
            singleton=False,
            provenance={"source": "tool", "tool_name": "brain_remember_preference"},
            source_episode_id=None,
            agent_id=session_ids.agent_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
        )
        await params.result_callback(
            {
                "accepted": True,
                "memory_kind": namespace,
                "rendered_text": fact.rendered_text,
            }
        )

    async def brain_remember_task(params: FunctionCallParams):
        title = str(params.arguments.get("title", "")).strip()
        details = str(params.arguments.get("details", "")).strip()
        if not title:
            await params.result_callback(
                {
                    "accepted": False,
                    "error": "invalid_task_memory_request",
                }
            )
            return

        session_ids = session_resolver()
        task_id = store.upsert_task(
            user_id=session_ids.user_id,
            title=title,
            details={"details": details} if details else {},
            status="open",
            thread_id=session_ids.thread_id,
            provenance={"source": "tool", "tool_name": "brain_remember_task"},
            agent_id=session_ids.agent_id,
            session_id=session_ids.session_id,
        )
        await params.result_callback(
            {
                "accepted": True,
                "task_id": task_id,
                "title": title,
                "status": "open",
            }
        )

    async def brain_forget_memory(params: FunctionCallParams):
        kind = str(params.arguments.get("kind", "")).strip()
        target = str(params.arguments.get("target", "")).strip()
        session_ids = session_resolver()

        if kind == "task":
            if not target:
                await params.result_callback(
                    {
                        "accepted": False,
                        "error": "task_target_required",
                    }
                )
                return
            affected = store.update_task_status(
                user_id=session_ids.user_id,
                title=target,
                status="cancelled",
                thread_id=session_ids.thread_id,
                agent_id=session_ids.agent_id,
                session_id=session_ids.session_id,
            )
        elif kind in profile_fields.values():
            affected = store.forget_facts(
                user_id=session_ids.user_id,
                namespace=kind,
                subject="user",
                agent_id=session_ids.agent_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
            )
        elif kind in preference_kinds.values():
            if not target:
                await params.result_callback(
                    {
                        "accepted": False,
                        "error": "preference_target_required",
                    }
                )
                return
            affected = store.forget_facts(
                user_id=session_ids.user_id,
                namespace=kind,
                subject=target.lower(),
                agent_id=session_ids.agent_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
            )
        else:
            await params.result_callback(
                {
                    "accepted": False,
                    "error": "unsupported_memory_kind",
                    "supported_kinds": [
                        "profile.name",
                        "profile.role",
                        "profile.origin",
                        "preference.like",
                        "preference.dislike",
                        "task",
                    ],
                }
            )
            return

        await params.result_callback(
            {
                "accepted": affected > 0,
                "memory_kind": kind,
                "affected": affected,
            }
        )

    async def brain_apply_memory_governance(params: FunctionCallParams):
        memory_id = str(params.arguments.get("memory_id", "")).strip()
        action = str(params.arguments.get("action", "")).strip()
        replacement_value = params.arguments.get("replacement_value")
        notes = params.arguments.get("notes")
        session_ids = session_resolver()
        try:
            result = apply_memory_governance_action(
                store=store,
                session_ids=session_ids,
                memory_id=memory_id,
                action=action,
                replacement_value=str(replacement_value).strip()
                if replacement_value is not None
                else None,
                notes=str(notes).strip() if notes is not None else None,
                source="memory_tool",
            )
            payload = result.as_dict()
        except Exception as exc:  # pragma: no cover - defensive tool boundary
            payload = {
                "schema_version": 1,
                "accepted": False,
                "applied": False,
                "action": action,
                "memory_id": memory_id,
                "record_kind": None,
                "event_id": None,
                "replacement_memory_id": None,
                "reason_codes": [
                    "memory_action_rejected",
                    f"memory_governance_error:{type(exc).__name__}",
                ],
            }
        await params.result_callback(payload)

    async def brain_complete_task(params: FunctionCallParams):
        title = str(params.arguments.get("title", "")).strip()
        if not title:
            await params.result_callback(
                {
                    "accepted": False,
                    "error": "task_title_required",
                }
            )
            return

        session_ids = session_resolver()
        affected = store.update_task_status(
            user_id=session_ids.user_id,
            title=title,
            status="done",
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            session_id=session_ids.session_id,
        )
        await params.result_callback(
            {
                "accepted": affected > 0,
                "title": title,
                "status": "done",
                "affected": affected,
            }
        )

    async def brain_list_visible_memories(params: FunctionCallParams):
        raw_limit = params.arguments.get("limit", 8)
        try:
            requested_limit = int(raw_limit)
        except (TypeError, ValueError):
            requested_limit = 8
        bounded_limit = max(0, min(12, requested_limit or 8))
        session_ids = session_resolver()
        snapshot = build_memory_palace_snapshot(
            store=store,
            session_ids=session_ids,
            include_suppressed=False,
            include_historical=False,
            limit=bounded_limit,
        )
        records = [
            {
                "memory_id": record.memory_id,
                "display_kind": record.display_kind,
                "title": record.title,
                "summary": record.summary,
                "status": record.status,
                "currentness_status": record.currentness_status,
                "pinned": record.pinned,
                "used_in_current_turn": record.used_in_current_turn,
            }
            for record in snapshot.records[:bounded_limit]
        ]
        await params.result_callback(
            {
                "schema_version": 1,
                "available": True,
                "record_count": len(records),
                "records": records,
                "hidden_counts": dict(snapshot.hidden_counts),
                "summary": snapshot.health_summary,
                "reason_codes": [
                    "visible_memory_recall:v1",
                    "visible_memory_recall:available",
                    *snapshot.reason_codes,
                ],
            }
        )

    async def brain_explain_memory_continuity(params: FunctionCallParams):
        session_ids = session_resolver()
        trace = store.latest_memory_continuity_trace(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        )
        if trace is None:
            latest_use_trace = store.latest_memory_use_trace(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
            )
            trace = build_memory_continuity_trace(
                memory_use_trace=latest_use_trace,
                session_id=session_ids.session_id,
                profile="browser",
                language=language.value,
                reason_codes=("memory_continuity:explain_fallback",),
            )
        trace_payload = trace.as_dict()
        selected = trace_payload.get("selected_memories", [])
        suppressed = trace_payload.get("suppressed_memories", [])
        continuity_v3 = (
            trace_payload.get("memory_continuity_v3")
            if isinstance(trace_payload.get("memory_continuity_v3"), dict)
            else {}
        )
        await params.result_callback(
            {
                "schema_version": 1,
                "available": True,
                "memory_effect": trace_payload.get("memory_effect", "none"),
                "memory_effect_labels": continuity_v3.get("effect_labels", ["none"]),
                "selected_memory_count": trace_payload.get("selected_memory_count", 0),
                "suppressed_memory_count": trace_payload.get("suppressed_memory_count", 0),
                "cross_language_count": trace_payload.get("cross_language_count", 0),
                "cross_language_transfer_count": continuity_v3.get(
                    "cross_language_transfer_count",
                    0,
                ),
                "selected_memories": [
                    {
                        "memory_id": item.get("memory_id"),
                        "display_kind": item.get("display_kind"),
                        "summary": item.get("summary"),
                        "effect_labels": item.get("effect_labels", ["none"]),
                        "linked_discourse_episode_ids": item.get(
                            "linked_discourse_episode_ids",
                            [],
                        ),
                        "cross_language": item.get("cross_language") is True,
                        "editable": item.get("editable") is not False,
                    }
                    for item in selected[:8]
                    if isinstance(item, dict)
                ],
                "selected_discourse_episodes": [
                    {
                        "discourse_episode_id": item.get("discourse_episode_id"),
                        "category_labels": item.get("category_labels", []),
                        "effect_labels": item.get("effect_labels", ["none"]),
                        "confidence_bucket": item.get("confidence_bucket", "medium"),
                    }
                    for item in continuity_v3.get("selected_discourse_episodes", [])[:8]
                    if isinstance(item, dict)
                ],
                "suppressed_memories": [
                    {
                        "bucket": item.get("bucket"),
                        "count": item.get("count"),
                    }
                    for item in suppressed[:8]
                    if isinstance(item, dict)
                ],
                "summary": (
                    "Memory continuity shaped this answer."
                    if selected
                    else "No user-visible memory was used for the latest answer."
                ),
                "reason_codes": [
                    "memory_continuity_explanation:v1",
                    "memory_continuity_explanation:available",
                    *trace_payload.get("reason_codes", [])[:12],
                ],
            }
        )

    if hasattr(llm, "register_function"):
        llm.register_function("brain_remember_profile", brain_remember_profile)
        llm.register_function("brain_remember_preference", brain_remember_preference)
        llm.register_function("brain_remember_task", brain_remember_task)
        llm.register_function("brain_forget_memory", brain_forget_memory)
        llm.register_function("brain_apply_memory_governance", brain_apply_memory_governance)
        llm.register_function("brain_complete_task", brain_complete_task)
        llm.register_function("brain_list_visible_memories", brain_list_visible_memories)
        llm.register_function("brain_explain_memory_continuity", brain_explain_memory_continuity)

    if language.value.lower().startswith(("zh", "cmn")):
        remember_profile_description = "记住或纠正稳定的个人资料字段，例如名字、身份或来自哪里。"
        remember_preference_description = "记住或纠正稳定偏好，例如喜欢或不喜欢什么。"
        remember_task_description = "记住一个明确的待办或提醒事项。"
        forget_memory_description = "忘记某条受支持的记忆，或取消一条待办事项。"
        governance_description = "对记忆面板中的 scoped memory_id 执行受限治理操作。"
        complete_task_description = "将一条待办事项标记为已完成。"
        list_visible_memories_description = (
            "当用户问“你现在记得什么”时，列出当前可见记忆的简短安全摘要。"
        )
        explain_memory_continuity_description = (
            "当用户问“为什么这样回答”时，解释本轮公开记忆连续性摘要，不泄露隐藏提示或原始记忆。"
        )
    else:
        remember_profile_description = (
            "Remember or correct a stable profile field such as name, role, or origin."
        )
        remember_preference_description = (
            "Remember or correct a stable preference such as like or dislike."
        )
        remember_task_description = "Remember an explicit task or reminder."
        forget_memory_description = "Forget one supported memory item or cancel a task."
        governance_description = "Apply a bounded governance action to one scoped memory_id."
        complete_task_description = "Mark one remembered task as completed."
        list_visible_memories_description = (
            "List a short safe summary of currently visible memories when the user asks what Blink remembers."
        )
        explain_memory_continuity_description = (
            "Explain the latest public memory-continuity summary when the user asks why Blink answered that way."
        )

    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="brain_remember_profile",
                description=remember_profile_description,
                properties={
                    "field": {
                        "type": "string",
                        "enum": sorted(profile_fields),
                        "description": "Profile field name.",
                    },
                    "value": {
                        "type": "string",
                        "description": "Canonical value to store.",
                    },
                },
                required=["field", "value"],
            ),
            FunctionSchema(
                name="brain_remember_preference",
                description=remember_preference_description,
                properties={
                    "sentiment": {
                        "type": "string",
                        "enum": sorted(preference_kinds),
                        "description": "Preference polarity.",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Preference topic to remember.",
                    },
                },
                required=["sentiment", "topic"],
            ),
            FunctionSchema(
                name="brain_remember_task",
                description=remember_task_description,
                properties={
                    "title": {
                        "type": "string",
                        "description": "Short task title.",
                    },
                    "details": {
                        "type": "string",
                        "description": "Optional extra task details.",
                    },
                },
                required=["title"],
            ),
            FunctionSchema(
                name="brain_forget_memory",
                description=forget_memory_description,
                properties={
                    "kind": {
                        "type": "string",
                        "enum": [
                            "profile.name",
                            "profile.role",
                            "profile.origin",
                            "preference.like",
                            "preference.dislike",
                            "task",
                        ],
                        "description": "Supported memory kind to forget.",
                    },
                    "target": {
                        "type": "string",
                        "description": "Required for preference topics and task titles; omitted for profile fields.",
                    },
                },
                required=["kind"],
            ),
            FunctionSchema(
                name="brain_apply_memory_governance",
                description=governance_description,
                properties={
                    "memory_id": {
                        "type": "string",
                        "description": "Scoped memory id from the memory palace read model.",
                    },
                    "action": {
                        "type": "string",
                        "enum": [
                            "pin",
                            "suppress",
                            "correct",
                            "forget",
                            "mark_stale",
                            "mark_done",
                            "cancel",
                        ],
                        "description": "Governance action to apply.",
                    },
                    "replacement_value": {
                        "type": "string",
                        "description": "Required only when action is correct.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional user-facing note about the action.",
                    },
                },
                required=["memory_id", "action"],
            ),
            FunctionSchema(
                name="brain_complete_task",
                description=complete_task_description,
                properties={
                    "title": {
                        "type": "string",
                        "description": "Exact task title to mark completed.",
                    }
                },
                required=["title"],
            ),
            FunctionSchema(
                name="brain_list_visible_memories",
                description=list_visible_memories_description,
                properties={
                    "limit": {
                        "type": "integer",
                        "description": "Maximum visible memories to return, capped at 12.",
                    }
                },
                required=[],
            ),
            FunctionSchema(
                name="brain_explain_memory_continuity",
                description=explain_memory_continuity_description,
                properties={},
                required=[],
            ),
        ]
    )


class BrainMemoryConsolidator:
    """Background memory consolidation for typed Blink brain memory."""

    def __init__(self, *, store: BrainStore):
        """Initialize the consolidator.

        Args:
            store: Canonical local-first brain store.
        """
        self._store = store

    def run_once(self, *, user_id: str, thread_id: str) -> BrainConsolidationResult:
        """Incrementally promote layered memory from the event spine with episode fallback."""
        event_metadata_key = f"consolidator:last_event_id:{user_id}:{thread_id}"
        last_event_raw = self._store.get_metadata(event_metadata_key)
        last_event_id = int(last_event_raw) if last_event_raw not in (None, "") else 0

        episode_metadata_key = f"consolidator:last_episode_id:{user_id}:{thread_id}"
        last_episode_raw = self._store.get_metadata(episode_metadata_key)
        last_episode_id = int(last_episode_raw) if last_episode_raw not in (None, "") else 0

        new_events = self._store.brain_events_since(
            user_id=user_id,
            thread_id=thread_id,
            after_id=last_event_id,
            limit=64,
        )
        promoted_facts = 0
        upserted_tasks = 0
        latest_episode_seen = last_episode_id
        latest_event_seen = last_event_id
        has_event_history = last_event_id > 0 or bool(
            self._store.recent_brain_events(
                user_id=user_id,
                thread_id=thread_id,
                limit=1,
            )
        )

        for event in new_events:
            latest_event_seen = event.id
            provenance = {
                "source": event.source,
                "source_event_id": event.event_id,
                "source_event_type": event.event_type,
                "correlation_id": event.correlation_id,
            }
            for candidate in build_episodic_candidates_from_event(event):
                self._store.add_episodic_memory(
                    agent_id=event.agent_id,
                    user_id=event.user_id,
                    session_id=event.session_id,
                    thread_id=event.thread_id,
                    kind=candidate.kind,
                    summary=candidate.summary,
                    payload=candidate.payload,
                    confidence=candidate.confidence,
                    source_event_id=event.event_id,
                    provenance=provenance,
                    stale_after_seconds=candidate.stale_after_seconds,
                    observed_at=event.ts,
                )

            if event.event_type == BrainEventType.USER_TURN_TRANSCRIBED:
                text = str(event.payload.get("text", "")).strip()
                for candidate in extract_memory_candidates(text):
                    self._store.remember_fact(
                        user_id=user_id,
                        namespace=candidate.namespace,
                        subject=candidate.subject,
                        value=candidate.value,
                        rendered_text=candidate.rendered_text,
                        confidence=min(0.95, candidate.confidence + 0.08),
                        singleton=candidate.singleton,
                        source_event_id=event.event_id,
                        source_episode_id=None,
                        provenance=provenance,
                        agent_id=event.agent_id,
                        session_id=event.session_id,
                        thread_id=event.thread_id,
                    )
                    promoted_facts += 1

                for candidate in extract_task_candidates(text):
                    self._store.upsert_task(
                        user_id=user_id,
                        title=candidate.title,
                        details=candidate.details,
                        status=candidate.status,
                        thread_id=thread_id,
                        source_event_id=event.event_id,
                        provenance=provenance,
                        agent_id=event.agent_id,
                        session_id=event.session_id,
                    )
                    upserted_tasks += 1

            if event.event_type == BrainEventType.GOAL_COMPLETED:
                goal_payload = event.payload.get("goal")
                title = ""
                if isinstance(goal_payload, dict):
                    title = str(goal_payload.get("title", "")).strip()
                if not title:
                    title = str(event.payload.get("title", "")).strip()
                if title:
                    self._store.update_task_status(
                        user_id=user_id,
                        title=title,
                        status="done",
                        thread_id=thread_id,
                        agent_id=event.agent_id,
                        session_id=event.session_id,
                        source_event_id=event.event_id,
                    )

        if latest_event_seen is not None and latest_event_seen != last_event_id:
            self._store.set_metadata(event_metadata_key, str(latest_event_seen))

        if not new_events and not has_event_history:
            new_episodes = self._store.episodes_since(
                user_id=user_id,
                thread_id=thread_id,
                after_id=last_episode_id,
                limit=32,
            )
            for episode in new_episodes:
                latest_episode_seen = episode.id
                for candidate in extract_memory_candidates(episode.user_text):
                    self._store.remember_fact(
                        user_id=user_id,
                        namespace=candidate.namespace,
                        subject=candidate.subject,
                        value=candidate.value,
                        rendered_text=candidate.rendered_text,
                        confidence=min(0.95, candidate.confidence + 0.08),
                        singleton=candidate.singleton,
                        source_episode_id=episode.id,
                        provenance={"source": "episode", "source_episode_id": episode.id},
                        agent_id=episode.agent_id,
                        session_id=episode.session_id,
                        thread_id=episode.thread_id,
                    )
                    promoted_facts += 1

                for candidate in extract_task_candidates(episode.user_text):
                    self._store.upsert_task(
                        user_id=user_id,
                        title=candidate.title,
                        details=candidate.details,
                        status=candidate.status,
                        thread_id=thread_id,
                        provenance={"source": "episode", "source_episode_id": episode.id},
                        agent_id=episode.agent_id,
                        session_id=episode.session_id,
                    )
                    upserted_tasks += 1

            if latest_episode_seen is not None and latest_episode_seen != last_episode_id:
                self._store.set_metadata(episode_metadata_key, str(latest_episode_seen))

        summary = self._build_thread_summary(user_id=user_id, thread_id=thread_id)
        if summary:
            self._store.upsert_session_summary(
                user_id=user_id,
                thread_id=thread_id,
                summary=summary,
                agent_id=new_events[-1].agent_id if new_events else None,
                session_id=new_events[-1].session_id if new_events else None,
                source_event_id=new_events[-1].event_id if new_events else None,
            )
        self._store.consolidate_procedural_skills(
            user_id=user_id,
            thread_id=thread_id,
            scope_type="thread",
            scope_id=thread_id,
        )
        self._store.refresh_private_working_memory_projection(
            user_id=user_id,
            thread_id=thread_id,
            source_event_id=(new_events[-1].event_id if new_events else None),
            updated_at=(new_events[-1].ts if new_events else None),
            agent_id=(new_events[-1].agent_id if new_events else None),
            commit=True,
        )
        self._store.refresh_active_situation_model_projection(
            user_id=user_id,
            thread_id=thread_id,
            source_event_id=(new_events[-1].event_id if new_events else None),
            updated_at=(new_events[-1].ts if new_events else None),
            agent_id=(new_events[-1].agent_id if new_events else None),
            commit=True,
        )
        BrainMemoryExporter(store=self._store).export_thread_digest(
            user_id=user_id,
            thread_id=thread_id,
        )

        return BrainConsolidationResult(
            promoted_facts=promoted_facts,
            upserted_tasks=upserted_tasks,
            latest_episode_id=latest_episode_seen or None,
            latest_event_id=latest_event_seen,
            summary=summary,
        )

    def _build_thread_summary(self, *, user_id: str, thread_id: str) -> str:
        """Build a concise narrative summary from layered memory with episode fallback."""
        recent_user_turns = [
            record.summary
            for record in reversed(
                self._store.episodic_memories(
                    user_id=user_id,
                    thread_id=thread_id,
                    kinds=("user_turn",),
                    limit=3,
                )
            )
        ]
        recent_assistant_turns = [
            record.summary
            for record in reversed(
                self._store.episodic_memories(
                    user_id=user_id,
                    thread_id=thread_id,
                    kinds=("assistant_turn",),
                    limit=3,
                )
            )
        ]
        if not recent_user_turns and not recent_assistant_turns:
            episodes = list(
                reversed(self._store.recent_episodes(user_id=user_id, thread_id=thread_id, limit=4))
            )
            recent_user_turns = [episode.user_text for episode in episodes if episode.user_text][
                -3:
            ]
            recent_assistant_turns = [
                episode.assistant_summary for episode in episodes if episode.assistant_summary
            ][-3:]

        active_tasks = self._store.active_tasks(user_id=user_id, limit=3)
        return build_thread_summary_text(
            recent_user_turns=recent_user_turns,
            recent_assistant_turns=recent_assistant_turns,
            open_commitments=[task["title"] for task in active_tasks],
        )
