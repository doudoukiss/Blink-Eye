from __future__ import annotations

import json
import string
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from hypothesis import strategies as st

from blink.brain.context_surfaces import BrainContextSurfaceSnapshot
from blink.brain.core import (
    BrainCandidateGoal,
    BrainCommitmentRecord,
    BrainCommitmentScopeType,
    BrainCommitmentStatus,
    BrainCommitmentWakeRouteKind,
    BrainCommitmentWakeRoutingDecision,
    BrainCommitmentWakeTrigger,
    BrainCoreReplayScenario,
    BrainCoreStore,
    BrainEventRecord,
    BrainEventType,
    BrainInitiativeClass,
    BrainPresenceSnapshot,
    BrainReevaluationConditionKind,
    BrainReevaluationTrigger,
    BrainSessionIds,
    BrainWakeCondition,
    BrainWakeConditionKind,
    resolve_brain_session_ids,
)
from blink.brain.replay import BrainReplayScenario

SESSION_IDS = resolve_brain_session_ids(runtime_kind="text", client_id="brain-property")
PRESENCE_SCOPE_KEY = "local:presence"

_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)
_WORD = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=8)
_TOOL_NAMES = ("lookup_profile", "inspect_scene", "remember_note")


def _ts(offset_seconds: int) -> str:
    return (_BASE_TIME + timedelta(seconds=offset_seconds)).isoformat()


def _phrase_strategy(*, min_words: int = 1, max_words: int = 4) -> st.SearchStrategy[str]:
    return st.lists(_WORD, min_size=min_words, max_size=max_words).map(" ".join)


@dataclass(frozen=True)
class ReplaySemanticCase:
    user_text: str
    goal_title: str
    candidate_summary: str
    reevaluation_summary: str
    reevaluation_condition: str
    reevaluation_condition_kind: str
    robot_head_mode: str
    vision_connected: bool
    warnings: tuple[str, ...]
    tail_kind: str
    assistant_text: str | None = None
    tool_name: str | None = None
    tool_result_text: str | None = None


@st.composite
def replay_case_strategy(draw) -> ReplaySemanticCase:
    tail_kind = draw(st.sampled_from(("none", "assistant", "tool")))
    assistant_text = draw(_phrase_strategy(max_words=5)) if tail_kind == "assistant" else None
    tool_name = draw(st.sampled_from(_TOOL_NAMES)) if tail_kind == "tool" else None
    tool_result_text = draw(_phrase_strategy(max_words=5)) if tail_kind == "tool" else None
    return ReplaySemanticCase(
        user_text=draw(_phrase_strategy(max_words=5)),
        goal_title=draw(_phrase_strategy(max_words=4)),
        candidate_summary=draw(_phrase_strategy(max_words=5)),
        reevaluation_summary=draw(_phrase_strategy(max_words=5)),
        reevaluation_condition=draw(_phrase_strategy(max_words=5)),
        reevaluation_condition_kind=draw(
            st.sampled_from(
                (
                    BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
                    BrainReevaluationConditionKind.THREAD_IDLE.value,
                    BrainReevaluationConditionKind.TIME_REACHED.value,
                )
            )
        ),
        robot_head_mode=draw(st.sampled_from(("none", "simulation"))),
        vision_connected=draw(st.booleans()),
        warnings=tuple(
            draw(st.lists(_phrase_strategy(max_words=2), min_size=0, max_size=1, unique=True))
        ),
        tail_kind=tail_kind,
        assistant_text=assistant_text,
        tool_name=tool_name,
        tool_result_text=tool_result_text,
    )


def build_core_replay_scenario(
    case: ReplaySemanticCase,
    *,
    stream_name: str,
) -> BrainCoreReplayScenario:
    return BrainCoreReplayScenario(
        name=f"property_{stream_name}",
        description=f"property replay case ({case.tail_kind})",
        session_ids=SESSION_IDS,
        events=materialize_core_events(case, stream_name=stream_name),
        expected_terminal_state={},
    )


def materialize_core_events(
    case: ReplaySemanticCase,
    *,
    stream_name: str,
    equivalent_stream: bool = False,
) -> tuple[BrainEventRecord, ...]:
    tags = ["beta", "alpha"] if equivalent_stream else ["alpha", "beta"]
    events: list[BrainEventRecord] = []

    snapshot = BrainPresenceSnapshot(
        runtime_kind="text",
        robot_head_enabled=case.robot_head_mode != "none",
        robot_head_mode=case.robot_head_mode,
        robot_head_available=case.robot_head_mode != "none",
        vision_enabled=True,
        vision_connected=case.vision_connected,
        warnings=list(case.warnings),
        updated_at=_ts(0),
    )
    events.append(
        _event_record(
            stream_name=stream_name,
            index=0,
            event_type=BrainEventType.BODY_STATE_UPDATED,
            payload={"scope_key": PRESENCE_SCOPE_KEY, "snapshot": snapshot.as_dict()},
            tags=tags,
        )
    )
    events.append(
        _event_record(
            stream_name=stream_name,
            index=1,
            event_type=BrainEventType.USER_TURN_TRANSCRIBED,
            payload={"text": _pad(case.user_text) if equivalent_stream else case.user_text},
            tags=tags,
        )
    )
    events.append(
        _event_record(
            stream_name=stream_name,
            index=2,
            event_type=BrainEventType.GOAL_CREATED,
            payload={
                "title": _pad(case.goal_title) if equivalent_stream else case.goal_title,
                "status": "open",
            },
            tags=tags,
        )
    )
    candidate_goal = BrainCandidateGoal(
        candidate_goal_id="candidate-property-1",
        candidate_type="presence_acknowledgement",
        source="runtime",
        summary=case.candidate_summary,
        goal_family="conversation",
        urgency=0.6,
        confidence=0.9,
        initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        dedupe_key="dedupe:property",
        created_at=_ts(3),
    )
    events.append(
        _event_record(
            stream_name=stream_name,
            index=3,
            event_type=BrainEventType.GOAL_CANDIDATE_CREATED,
            payload={"candidate_goal": candidate_goal.as_dict()},
            tags=tags,
        )
    )
    events.append(
        _event_record(
            stream_name=stream_name,
            index=4,
            event_type=BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
            payload={
                "candidate_goal_id": candidate_goal.candidate_goal_id,
                "reason": "policy_hold",
                "reason_details": {"tail_kind": case.tail_kind},
                "expected_reevaluation_condition": case.reevaluation_condition,
                "expected_reevaluation_condition_kind": case.reevaluation_condition_kind,
                "expected_reevaluation_condition_details": {
                    "candidate_goal_id": candidate_goal.candidate_goal_id
                },
            },
            tags=tags,
        )
    )

    if case.tail_kind == "assistant" and case.assistant_text is not None:
        events.append(
            _event_record(
                stream_name=stream_name,
                index=5,
                event_type=BrainEventType.ASSISTANT_TURN_ENDED,
                payload={
                    "text": _pad(case.assistant_text)
                    if equivalent_stream
                    else case.assistant_text
                },
                tags=tags,
            )
        )
    elif (
        case.tail_kind == "tool"
        and case.tool_name is not None
        and case.tool_result_text is not None
    ):
        tool_payload = {
            "tool_call_id": "tool-call-1",
            "function_name": case.tool_name,
        }
        events.append(
            _event_record(
                stream_name=stream_name,
                index=5,
                event_type=BrainEventType.TOOL_CALLED,
                payload={**tool_payload, "arguments": {"subject": case.goal_title}},
                tags=tags,
            )
        )
        events.append(
            _event_record(
                stream_name=stream_name,
                index=6,
                event_type=BrainEventType.TOOL_COMPLETED,
                payload={
                    **tool_payload,
                    "result": {"summary": case.tool_result_text},
                },
                tags=tags,
            )
        )

    assert 3 <= len(events) <= 7
    return tuple(events)


def materialize_duplicate_noise_events(
    case: ReplaySemanticCase,
    *,
    duplicate_kinds: Sequence[str],
) -> tuple[BrainEventRecord, ...]:
    events: list[BrainEventRecord] = []
    for offset, kind in enumerate(duplicate_kinds, start=1):
        index = 30 + offset
        if kind == "reevaluation":
            trigger = BrainReevaluationTrigger(
                kind=case.reevaluation_condition_kind,
                summary=case.reevaluation_summary,
                details={"candidate_goal_id": "candidate-property-1"},
                source_event_type=BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
                source_event_id=f"ignored-reeval-{offset}",
                ts=_ts(index),
            )
            payload = {
                "trigger": trigger.as_dict(),
                "candidate_goal_ids": ["candidate-property-1"],
            }
            event_type = BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED
        else:
            commitment = BrainCommitmentRecord(
                commitment_id="commitment-property-1",
                scope_type=BrainCommitmentScopeType.THREAD.value,
                scope_id=SESSION_IDS.thread_id,
                title=case.goal_title,
                goal_family="conversation",
                intent="follow_up",
                status=BrainCommitmentStatus.DEFERRED.value,
                current_goal_id="goal-property-1",
                created_at=_ts(index),
                updated_at=_ts(index),
            )
            wake_condition = BrainWakeCondition(
                kind=BrainWakeConditionKind.THREAD_IDLE.value,
                summary="Wait for the thread to become idle.",
            )
            trigger = BrainCommitmentWakeTrigger(
                commitment_id=commitment.commitment_id,
                wake_kind=wake_condition.kind,
                summary="Wake routing replay noise.",
                details={"candidate_goal_id": "candidate-property-1"},
                source_event_type=BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
                source_event_id=f"ignored-wake-{offset}",
                ts=_ts(index),
            )
            routing = BrainCommitmentWakeRoutingDecision(
                route_kind=BrainCommitmentWakeRouteKind.KEEP_WAITING.value,
                summary="Keep waiting until a real wake pass runs.",
                details={"candidate_goal_id": "candidate-property-1"},
            )
            payload = {
                "commitment": commitment.as_dict(),
                "wake_condition": wake_condition.as_dict(),
                "trigger": trigger.as_dict(),
                "routing": routing.as_dict(),
            }
            event_type = BrainEventType.COMMITMENT_WAKE_TRIGGERED
        events.append(
            _event_record(
                stream_name=f"duplicate-{kind}",
                index=index,
                event_type=event_type,
                payload=payload,
                tags=["noise", "ignored"],
            )
        )
    return tuple(events)


def load_replay_scenario_from_artifact(path: Path) -> BrainReplayScenario:
    payload = json.loads(path.read_text(encoding="utf-8"))
    session = dict(payload["session"])
    session_ids = BrainSessionIds(
        agent_id=str(session["agent_id"]),
        user_id=str(session["user_id"]),
        session_id=str(session["session_id"]),
        thread_id=str(session["thread_id"]),
    )
    scenario = dict(payload["scenario"])
    events = tuple(
        BrainEventRecord(
            id=index + 1,
            event_id=str(item["event_id"]),
            event_type=str(item["event_type"]),
            ts=str(item["ts"]),
            agent_id=str(item.get("agent_id", session_ids.agent_id)),
            user_id=str(item.get("user_id", session_ids.user_id)),
            session_id=str(item.get("session_id", session_ids.session_id)),
            thread_id=str(item.get("thread_id", session_ids.thread_id)),
            source=str(item.get("source", "artifact")),
            correlation_id=item.get("correlation_id"),
            causal_parent_id=item.get("causal_parent_id"),
            confidence=float(item.get("confidence", 1.0)),
            payload_json=json.dumps(item.get("payload", {}), ensure_ascii=False, sort_keys=True),
            tags_json=json.dumps(item.get("tags", []), ensure_ascii=False),
        )
        for index, item in enumerate(payload["events"])
    )
    return BrainReplayScenario(
        name=str(scenario["name"]),
        description=str(scenario.get("description", scenario["name"])),
        session_ids=session_ids,
        events=events,
        expected_terminal_state=dict(payload.get("qa", {}).get("expected_terminal_state", {})),
    )


def normalized_core_projection_bundle_from_store(
    store: BrainCoreStore,
    *,
    include_heartbeat: bool = True,
) -> dict[str, Any]:
    projections = {
        "body": store.get_body_state_projection(scope_key=PRESENCE_SCOPE_KEY).as_dict(),
        "scene": store.get_scene_state_projection(scope_key=PRESENCE_SCOPE_KEY).as_dict(),
        "engagement": store.get_engagement_state_projection(scope_key=PRESENCE_SCOPE_KEY).as_dict(),
        "relationship_state": store.get_relationship_state_projection(
            scope_key=PRESENCE_SCOPE_KEY
        ).as_dict(),
        "working_context": store.get_working_context_projection(
            scope_key=SESSION_IDS.thread_id
        ).as_dict(),
        "agenda": store.get_agenda_projection(scope_key=SESSION_IDS.thread_id).as_dict(),
        "autonomy_ledger": store.get_autonomy_ledger_projection(
            scope_key=SESSION_IDS.thread_id
        ).as_dict(),
    }
    if include_heartbeat:
        projections["heartbeat"] = store.get_heartbeat_projection(
            scope_key=SESSION_IDS.thread_id
        ).as_dict()
    return normalize_projection_bundle(projections)


def normalized_context_surface_snapshot(
    snapshot: BrainContextSurfaceSnapshot,
) -> dict[str, Any]:
    return normalize_projection_bundle(
        {
            "body": snapshot.body.as_dict(),
            "working_context": snapshot.working_context.as_dict(),
            "agenda": snapshot.agenda.as_dict(),
            "autonomy_ledger": snapshot.autonomy_ledger.as_dict(),
        }
    )


def normalized_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "projections": normalize_projection_bundle(dict(payload.get("projections", {}))),
        "terminal_state": _strip_updated_at(dict(payload.get("terminal_state", {}))),
    }


def normalize_projection_bundle(projections: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, payload in projections.items():
        cleaned = _strip_updated_at(payload)
        if name == "autonomy_ledger":
            cleaned = {
                **cleaned,
                "recent_entries": [
                    {key: value for key, value in entry.items() if key != "event_id"}
                    for entry in cleaned.get("recent_entries", [])
                ],
            }
        normalized[name] = cleaned
    return normalized


def _event_record(
    *,
    stream_name: str,
    index: int,
    event_type: str,
    payload: dict[str, Any],
    tags: Sequence[str],
) -> BrainEventRecord:
    return BrainEventRecord(
        id=index + 1,
        event_id=f"{stream_name}-{index}-{uuid4().hex}",
        event_type=event_type,
        ts=_ts(index),
        agent_id=SESSION_IDS.agent_id,
        user_id=SESSION_IDS.user_id,
        session_id=SESSION_IDS.session_id,
        thread_id=SESSION_IDS.thread_id,
        source="property-test",
        correlation_id=f"{stream_name}-corr-{index}",
        causal_parent_id=None if index == 0 else f"{stream_name}-cause-{index}",
        confidence=1.0,
        payload_json=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        tags_json=json.dumps(list(tags), ensure_ascii=False),
    )


def _pad(text: str) -> str:
    return f"  {text}  "


def _strip_updated_at(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_updated_at(item)
            for key, item in value.items()
            if key != "updated_at"
        }
    if isinstance(value, list):
        return [_strip_updated_at(item) for item in value]
    return value
