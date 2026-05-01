"""Shared structured-input helpers for narrow Blink Atheris harnesses."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import atheris

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parents[3]
SRC_ROOT = REPO_ROOT / "src"

for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


SAFE_REJECTION_EXCEPTIONS = (TypeError, ValueError, OverflowError)

_MAX_DEPTH = 3
_MAX_ITEMS = 6
_MAX_STRING_LENGTH = 64
_DEFAULT_SESSION = {
    "agent_id": "agent-fuzz",
    "user_id": "user-fuzz",
    "session_id": "session-fuzz",
    "thread_id": "thread-fuzz",
}
_DEFAULT_REPLAY_EVENT_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "event_type": "memory.block.upserted",
        "source": "atheris",
        "tags": ["atheris", "memory"],
        "payload": {
            "block_kind": "self_core",
            "scope_type": "agent",
            "scope_id": _DEFAULT_SESSION["agent_id"],
            "content": {"summary": "seed"},
        },
    },
    {
        "event_type": "memory.claim.recorded",
        "source": "atheris",
        "tags": ["atheris", "claim"],
        "payload": {
            "subject_entity_id": _DEFAULT_SESSION["user_id"],
            "predicate": "preference.like",
            "object": {"value": "tea"},
            "scope_type": "user",
            "scope_id": _DEFAULT_SESSION["user_id"],
        },
    },
    {
        "event_type": "autobiography.entry.upserted",
        "source": "atheris",
        "tags": ["atheris", "autobiography"],
        "payload": {
            "scope_type": "user",
            "scope_id": _DEFAULT_SESSION["user_id"],
            "entry_kind": "relationship_milestone",
            "rendered_summary": "Seed milestone",
            "content": {"project_key": "seed-project"},
        },
    },
    {
        "event_type": "goal.created",
        "source": "atheris",
        "tags": ["atheris", "goal"],
        "payload": {
            "commitment": {
                "commitment_id": "commitment-fuzz",
                "scope_type": "thread",
                "scope_id": _DEFAULT_SESSION["thread_id"],
                "status": "active",
            },
            "goal": {
                "goal_id": "goal-fuzz",
                "title": "Fuzz goal",
                "intent": "conversation.reply",
                "source": "atheris",
                "goal_family": "conversation",
                "details": {},
                "steps": [],
            },
        },
    },
    {
        "event_type": "goal.completed",
        "source": "atheris",
        "tags": ["atheris", "goal"],
        "payload": {
            "commitment": {
                "commitment_id": "commitment-fuzz",
                "scope_type": "thread",
                "scope_id": _DEFAULT_SESSION["thread_id"],
                "status": "completed",
            },
            "goal": {
                "goal_id": "goal-fuzz",
                "title": "Fuzz goal",
                "intent": "conversation.reply",
                "source": "atheris",
                "goal_family": "conversation",
                "status": "completed",
                "details": {"commitment_status": "completed"},
                "steps": [],
            },
        },
    },
)
_DEFAULT_ACTIVE_STATE_INPUTS = {
    "private_working_memory": {
        "scope_type": "thread",
        "scope_id": _DEFAULT_SESSION["thread_id"],
        "records": [
            {
                "record_id": "pwm-1",
                "buffer_kind": "plan_assumption",
                "summary": "Need operator review.",
                "state": "active",
                "evidence_kind": "derived",
                "backing_ids": ["proposal-1"],
                "source_event_ids": ["evt-1"],
            }
        ],
    },
    "active_situation_model": {
        "scope_type": "thread",
        "scope_id": _DEFAULT_SESSION["thread_id"],
        "records": [
            {
                "record_id": "situation-1",
                "record_kind": "plan_state",
                "summary": "Need operator review.",
                "state": "active",
                "evidence_kind": "derived",
                "backing_ids": ["proposal-1"],
                "source_event_ids": ["evt-1"],
            }
        ],
    },
    "scene_world_state": {
        "scope_type": "presence",
        "scope_id": "browser:presence",
        "entities": [
            {
                "entity_id": "entity-1",
                "entity_kind": "object",
                "canonical_label": "cup",
                "summary": "A cup is visible.",
                "state": "active",
                "evidence_kind": "observed",
                "affordance_ids": ["aff-1"],
            }
        ],
        "affordances": [
            {
                "affordance_id": "aff-1",
                "entity_id": "entity-1",
                "capability_family": "vision.inspect",
                "summary": "Inspect the cup.",
                "availability": "available",
            }
        ],
    },
}
_DEFAULT_AUTONOMY_LEDGER = {
    "current_candidates": [
        {
            "candidate_goal_id": "candidate-1",
            "candidate_type": "presence_acknowledgement",
            "source": "runtime",
            "summary": "Ack presence.",
            "goal_family": "environment",
            "urgency": 0.7,
            "confidence": 0.9,
            "initiative_class": "inspect_only",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ],
    "recent_entries": [],
    "updated_at": "2026-01-01T00:00:00+00:00",
}
_DEFAULT_AGENDA = {
    "goals": [
        {
            "goal_id": "goal-1",
            "title": "Ack presence.",
            "intent": "autonomy.presence_acknowledgement",
            "source": "atheris",
        }
    ],
    "updated_at": "2026-01-01T00:00:00+00:00",
}
_DEFAULT_OPERATOR_ARTIFACT_AGENDA = {
    "goals": [
        {
            "goal_id": "goal-review",
            "title": "Review the held wake.",
            "intent": "maintenance.review",
            "source": "atheris",
            "plan_revision": 1,
            "details": {
                "pending_plan_proposal_id": "proposal-review",
                "plan_review_policy": "needs_user_review",
            },
        },
        {
            "goal_id": "goal-adopt",
            "title": "Adopt the bounded follow-up plan.",
            "intent": "conversation.reply",
            "source": "atheris",
            "plan_revision": 1,
            "details": {
                "current_plan_proposal_id": "proposal-adopt",
                "plan_review_policy": "auto_adopt_ok",
            },
        },
    ],
    "updated_at": "2026-01-01T00:00:00+00:00",
}
_DEFAULT_OPERATOR_ARTIFACT_COMMITMENT_PROJECTION = {
    "active_commitments": [
        {
            "commitment_id": "commitment-adopt",
            "scope_type": "thread",
            "scope_id": _DEFAULT_SESSION["thread_id"],
            "title": "Adopt the bounded follow-up plan.",
            "goal_family": "conversation",
            "intent": "conversation.reply",
            "status": "active",
            "details": {"summary": "Active follow-up plan."},
            "current_goal_id": "goal-adopt",
            "plan_revision": 1,
            "resume_count": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    ],
    "deferred_commitments": [
        {
            "commitment_id": "commitment-review",
            "scope_type": "thread",
            "scope_id": _DEFAULT_SESSION["thread_id"],
            "title": "Review the held wake.",
            "goal_family": "memory_maintenance",
            "intent": "maintenance.review",
            "status": "deferred",
            "details": {"summary": "Waiting for explicit resume."},
            "current_goal_id": "goal-review",
            "blocked_reason": {
                "kind": "explicit_defer",
                "summary": "Deferred by the operator shell.",
                "details": {},
            },
            "wake_conditions": [
                {
                    "kind": "explicit_resume",
                    "summary": "Resume explicitly when ready.",
                    "details": {},
                }
            ],
            "plan_revision": 1,
            "resume_count": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    ],
    "blocked_commitments": [],
    "recent_terminal_commitments": [],
    "current_active_summary": "Adopt the bounded follow-up plan.",
    "updated_at": "2026-01-01T00:00:00+00:00",
}
_DEFAULT_RECENT_EVENT_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "id": 1,
        "event_id": "evt-1",
        "event_type": "goal.candidate.created",
        "ts": "2026-01-01T00:00:01+00:00",
        "agent_id": _DEFAULT_SESSION["agent_id"],
        "user_id": _DEFAULT_SESSION["user_id"],
        "session_id": _DEFAULT_SESSION["session_id"],
        "thread_id": _DEFAULT_SESSION["thread_id"],
        "source": "atheris",
        "correlation_id": None,
        "causal_parent_id": None,
        "confidence": 1.0,
        "payload": {
            "candidate_goal": _DEFAULT_AUTONOMY_LEDGER["current_candidates"][0],
        },
        "tags": ["atheris", "autonomy"],
    },
    {
        "id": 2,
        "event_id": "evt-2",
        "event_type": "director.reevaluation.triggered",
        "ts": "2026-01-01T00:00:02+00:00",
        "agent_id": _DEFAULT_SESSION["agent_id"],
        "user_id": _DEFAULT_SESSION["user_id"],
        "session_id": _DEFAULT_SESSION["session_id"],
        "thread_id": _DEFAULT_SESSION["thread_id"],
        "source": "atheris",
        "correlation_id": None,
        "causal_parent_id": None,
        "confidence": 1.0,
        "payload": {
            "trigger": {
                "kind": "user_turn_closed",
                "summary": "User turn closed.",
                "details": {"turn": "user"},
            },
            "candidate_goal_ids": ["candidate-1"],
        },
        "tags": ["atheris", "reevaluation"],
    },
)
_DEFAULT_SCENE_EVENT_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "id": 1,
        "event_id": "evt-scene-1",
        "event_type": "scene.changed",
        "ts": "2026-01-01T00:00:01+00:00",
        "agent_id": _DEFAULT_SESSION["agent_id"],
        "user_id": _DEFAULT_SESSION["user_id"],
        "session_id": _DEFAULT_SESSION["session_id"],
        "thread_id": _DEFAULT_SESSION["thread_id"],
        "source": "atheris",
        "payload": {
            "presence_scope_key": "browser:presence",
            "degraded_mode": "limited",
            "summary": "Scene changed under limited visibility.",
        },
        "tags": ["atheris", "scene"],
    },
    {
        "id": 2,
        "event_id": "evt-engagement-1",
        "event_type": "engagement.changed",
        "ts": "2026-01-01T00:00:02+00:00",
        "agent_id": _DEFAULT_SESSION["agent_id"],
        "user_id": _DEFAULT_SESSION["user_id"],
        "session_id": _DEFAULT_SESSION["session_id"],
        "thread_id": _DEFAULT_SESSION["thread_id"],
        "source": "atheris",
        "payload": {
            "presence_scope_key": "browser:presence",
            "focus_entity_id": "person-1",
            "summary": "A person engaged with the visible scene.",
        },
        "tags": ["atheris", "scene"],
    },
)
_DEFAULT_MULTIMODAL_AUTOBIOGRAPHY_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "entry_id": "entry-scene-current",
        "scope_type": "presence",
        "scope_id": "browser:presence",
        "entry_kind": "scene_episode",
        "modality": "scene_world",
        "review_state": "requested",
        "retention_class": "session",
        "privacy_class": "sensitive",
        "governance_reason_codes": ["privacy_boundary", "degraded_scene_evidence"],
        "last_governance_event_id": "evt-engagement-1",
        "source_presence_scope_key": "browser:presence",
        "source_scene_entity_ids": ["person-1", "entity-1"],
        "source_scene_affordance_ids": ["aff-1"],
        "rendered_summary": "limited; person near the desk; affordances: vision.inspect",
        "content": {
            "summary": "limited; person near the desk; affordances: vision.inspect",
            "semantic_fingerprint": "fingerprint-scene-current",
            "degraded_mode": "limited",
            "degraded_reason_codes": ["camera_occluded"],
            "anchor_entity_ids": ["person-1", "entity-1"],
            "anchor_affordance_ids": ["aff-1"],
            "supporting_event_ids": ["evt-scene-1", "evt-engagement-1"],
            "confidence_band": "medium",
            "salience": 1.65,
            "observed_at": "2026-01-01T00:00:02+00:00",
            "updated_at": "2026-01-01T00:00:02+00:00",
        },
        "status": "current",
        "salience": 1.65,
        "source_episode_ids": [],
        "source_claim_ids": [],
        "source_event_ids": ["evt-scene-1", "evt-engagement-1"],
        "supersedes_entry_id": "entry-scene-redacted",
        "valid_from": "2026-01-01T00:00:02+00:00",
        "valid_to": None,
        "created_at": "2026-01-01T00:00:02+00:00",
        "updated_at": "2026-01-01T00:00:02+00:00",
    },
    {
        "entry_id": "entry-scene-redacted",
        "scope_type": "presence",
        "scope_id": "browser:presence",
        "entry_kind": "scene_episode",
        "modality": "scene_world",
        "review_state": "resolved",
        "retention_class": "session",
        "privacy_class": "redacted",
        "governance_reason_codes": ["privacy_boundary"],
        "last_governance_event_id": "evt-redacted-1",
        "source_presence_scope_key": "browser:presence",
        "source_scene_entity_ids": ["person-1"],
        "source_scene_affordance_ids": ["aff-1"],
        "redacted_at": "2026-01-01T00:00:03+00:00",
        "rendered_summary": "[redacted scene episode]",
        "content": {
            "redacted": True,
            "redacted_summary": "[redacted scene episode]",
            "degraded_mode": "limited",
            "anchor_entity_ids": ["person-1"],
            "anchor_affordance_ids": ["aff-1"],
            "supporting_event_ids": ["evt-scene-0"],
        },
        "status": "superseded",
        "salience": 1.2,
        "source_episode_ids": [],
        "source_claim_ids": [],
        "source_event_ids": ["evt-scene-0"],
        "supersedes_entry_id": None,
        "valid_from": "2025-12-31T23:59:59+00:00",
        "valid_to": "2026-01-01T00:00:02+00:00",
        "created_at": "2025-12-31T23:59:59+00:00",
        "updated_at": "2026-01-01T00:00:03+00:00",
    },
)
_DEFAULT_OPERATOR_ARTIFACT_EVENT_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "id": 1,
        "event_id": "evt-shell-deferred",
        "event_type": "goal.deferred",
        "ts": "2026-01-01T00:00:01+00:00",
        "agent_id": _DEFAULT_SESSION["agent_id"],
        "user_id": _DEFAULT_SESSION["user_id"],
        "session_id": _DEFAULT_SESSION["session_id"],
        "thread_id": _DEFAULT_SESSION["thread_id"],
        "source": "runtime_shell",
        "payload": {
            "goal": {
                "goal_id": "goal-review",
                "title": "Review the held wake.",
                "intent": "maintenance.review",
                "source": "atheris",
                "goal_family": "memory_maintenance",
                "status": "waiting",
                "details": {"commitment_status": "deferred"},
                "steps": [],
            },
            "commitment": {
                "commitment_id": "commitment-review",
                "scope_type": "thread",
                "scope_id": _DEFAULT_SESSION["thread_id"],
                "status": "deferred",
            },
            "runtime_shell_control": {
                "control_kind": "interrupt",
                "status_before": "active",
                "status_after": "deferred",
                "reason_summary": "Pause while the operator inspects.",
            },
        },
        "tags": ["atheris", "runtime_shell"],
    },
    {
        "id": 2,
        "event_id": "evt-shell-resumed",
        "event_type": "goal.resumed",
        "ts": "2026-01-01T00:00:02+00:00",
        "agent_id": _DEFAULT_SESSION["agent_id"],
        "user_id": _DEFAULT_SESSION["user_id"],
        "session_id": _DEFAULT_SESSION["session_id"],
        "thread_id": _DEFAULT_SESSION["thread_id"],
        "source": "runtime_shell",
        "payload": {
            "goal": {
                "goal_id": "goal-review",
                "title": "Review the held wake.",
                "intent": "maintenance.review",
                "source": "atheris",
                "goal_family": "memory_maintenance",
                "status": "open",
                "details": {"commitment_status": "active"},
                "steps": [],
            },
            "commitment": {
                "commitment_id": "commitment-review",
                "scope_type": "thread",
                "scope_id": _DEFAULT_SESSION["thread_id"],
                "status": "active",
            },
            "runtime_shell_control": {
                "control_kind": "resume",
                "status_before": "deferred",
                "status_after": "active",
                "reason_summary": "Operator approved the resume.",
            },
        },
        "tags": ["atheris", "runtime_shell"],
    },
    {
        "id": 3,
        "event_id": "evt-wake-resume",
        "event_type": "commitment.wake.triggered",
        "ts": "2026-01-01T00:00:03+00:00",
        "agent_id": _DEFAULT_SESSION["agent_id"],
        "user_id": _DEFAULT_SESSION["user_id"],
        "session_id": _DEFAULT_SESSION["session_id"],
        "thread_id": _DEFAULT_SESSION["thread_id"],
        "source": "executive",
        "payload": {
            "commitment": {
                "commitment_id": "commitment-review",
                "title": "Review the held wake.",
                "status": "deferred",
            },
            "trigger": {
                "commitment_id": "commitment-review",
                "wake_kind": "explicit_resume",
                "summary": "Resume explicitly when ready.",
                "details": {"boundary_kind": "manual"},
                "source_event_type": "goal.deferred",
                "source_event_id": "evt-shell-deferred",
                "ts": "2026-01-01T00:00:03+00:00",
            },
            "routing": {
                "route_kind": "resume_direct",
                "summary": "Resume the commitment directly.",
                "details": {"reason": "explicit_resume_matched", "boundary_kind": "manual"},
                "reason_codes": ["explicit_resume_matched", "bounded_plan_available"],
                "executive_policy": {
                    "action_posture": "allow",
                    "approval_requirement": "none",
                },
            },
        },
        "tags": ["atheris", "wake"],
    },
    {
        "id": 4,
        "event_id": "evt-wake-goal-resumed",
        "event_type": "goal.resumed",
        "ts": "2026-01-01T00:00:04+00:00",
        "agent_id": _DEFAULT_SESSION["agent_id"],
        "user_id": _DEFAULT_SESSION["user_id"],
        "session_id": _DEFAULT_SESSION["session_id"],
        "thread_id": _DEFAULT_SESSION["thread_id"],
        "source": "executive",
        "causal_parent_id": "evt-wake-resume",
        "payload": {
            "goal": {
                "goal_id": "goal-review",
                "title": "Review the held wake.",
                "intent": "maintenance.review",
                "source": "atheris",
                "goal_family": "memory_maintenance",
                "status": "open",
                "details": {"commitment_status": "active"},
                "steps": [],
            },
            "commitment": {
                "commitment_id": "commitment-review",
                "scope_type": "thread",
                "scope_id": _DEFAULT_SESSION["thread_id"],
                "status": "active",
            },
        },
        "tags": ["atheris", "wake"],
    },
    {
        "id": 5,
        "event_id": "evt-plan-proposed",
        "event_type": "planning.proposed",
        "ts": "2026-01-01T00:00:05+00:00",
        "agent_id": _DEFAULT_SESSION["agent_id"],
        "user_id": _DEFAULT_SESSION["user_id"],
        "session_id": _DEFAULT_SESSION["session_id"],
        "thread_id": _DEFAULT_SESSION["thread_id"],
        "source": "executive",
        "payload": {
            "proposal": {
                "plan_proposal_id": "proposal-review",
                "goal_id": "goal-review",
                "commitment_id": "commitment-review",
                "source": "bounded_planner",
                "summary": "Draft a review-only plan.",
                "current_plan_revision": 1,
                "plan_revision": 1,
                "review_policy": "needs_user_review",
                "steps": [],
                "details": {
                    "procedural": {
                        "origin": "skill_reuse",
                        "selected_skill_id": "skill-review",
                        "selected_skill_support_trace_ids": ["trace-review"],
                        "selected_skill_support_plan_proposal_ids": ["proposal-seed"],
                        "policy": {
                            "action_posture": "defer",
                            "approval_requirement": "user_confirmation",
                            "procedural_reuse_eligibility": "advisory_only",
                            "reason_codes": ["held_claim_present", "policy_requires_confirmation"],
                            "effect": "advisory_only",
                        },
                    }
                },
                "created_at": "2026-01-01T00:00:05+00:00",
            },
            "decision": {
                "reason": "policy_requires_review",
                "reason_codes": ["policy_requires_confirmation", "held_claim_present"],
                "executive_policy": {
                    "action_posture": "defer",
                    "approval_requirement": "user_confirmation",
                },
            },
        },
        "tags": ["atheris", "planning"],
    },
    {
        "id": 6,
        "event_id": "evt-plan-adopted",
        "event_type": "planning.adopted",
        "ts": "2026-01-01T00:00:06+00:00",
        "agent_id": _DEFAULT_SESSION["agent_id"],
        "user_id": _DEFAULT_SESSION["user_id"],
        "session_id": _DEFAULT_SESSION["session_id"],
        "thread_id": _DEFAULT_SESSION["thread_id"],
        "source": "executive",
        "payload": {
            "proposal": {
                "plan_proposal_id": "proposal-adopt",
                "goal_id": "goal-adopt",
                "commitment_id": "commitment-adopt",
                "source": "bounded_planner",
                "summary": "Adopt a bounded follow-up plan.",
                "current_plan_revision": 1,
                "plan_revision": 1,
                "review_policy": "auto_adopt_ok",
                "steps": [],
                "details": {
                    "procedural": {
                        "origin": "skill_delta",
                        "selected_skill_id": "skill-adopt",
                        "selected_skill_support_trace_ids": ["trace-adopt"],
                        "selected_skill_support_plan_proposal_ids": ["proposal-seed"],
                        "delta": {"operations": [{"kind": "append_step"}]},
                        "policy": {
                            "action_posture": "allow",
                            "approval_requirement": "none",
                            "procedural_reuse_eligibility": "allowed",
                            "reason_codes": ["bounded_plan_available"],
                            "effect": "allowed",
                        },
                    }
                },
                "created_at": "2026-01-01T00:00:06+00:00",
            },
            "decision": {
                "reason": "bounded_plan_available",
                "reason_codes": ["bounded_plan_available"],
                "executive_policy": {
                    "action_posture": "allow",
                    "approval_requirement": "none",
                },
            },
        },
        "tags": ["atheris", "planning"],
    },
)
_DEFAULT_REFLECTION_CYCLE_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "cycle_id": "reflection-shell-1",
        "trigger": "runtime_shell:manual",
        "status": "completed",
        "draft_artifact_path": "/tmp/runtime-shell-reflection.json",
        "started_at": "2026-01-01T00:00:07+00:00",
        "completed_at": "2026-01-01T00:00:08+00:00",
    },
)
_DEFAULT_MEMORY_EXPORT_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "id": 1,
        "export_kind": "continuity_audit",
        "path": "/tmp/runtime-shell-audit.json",
        "generated_at": "2026-01-01T00:00:09+00:00",
        "metadata": {
            "source": "runtime_shell",
            "action_kind": "audit_export",
        },
    },
)


def decode_jsonish_input(data: bytes) -> Any:
    """Decode fuzz bytes into a bounded JSON-safe value."""
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        provider = atheris.FuzzedDataProvider(data)
        return _consume_jsonish_value(provider, depth=0)


def bound_jsonish_value(value: Any, *, depth: int = 0) -> Any:
    """Bound one JSON-like value so harnesses stay compact and readable."""
    if depth >= _MAX_DEPTH:
        return _coerce_leaf(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return max(min(int(value), 10_000), -10_000)
    if isinstance(value, float):
        return round(max(min(float(value), 10_000.0), -10_000.0), 3)
    if isinstance(value, str):
        return value[:_MAX_STRING_LENGTH]
    if isinstance(value, list):
        return [bound_jsonish_value(item, depth=depth + 1) for item in value[:_MAX_ITEMS]]
    if isinstance(value, dict):
        bounded: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_ITEMS:
                break
            normalized_key = str(key).strip()[:_MAX_STRING_LENGTH] or f"key_{index}"
            bounded[normalized_key] = bound_jsonish_value(item, depth=depth + 1)
        return bounded
    return _coerce_leaf(value)


def as_mapping(value: Any, *, field_name: str = "value") -> dict[str, Any]:
    """Return one bounded mapping view for a fuzz value."""
    bounded = bound_jsonish_value(value)
    if isinstance(bounded, dict):
        return bounded
    return {field_name: bounded}


def as_list(value: Any) -> list[Any]:
    """Return one bounded list view for a fuzz value."""
    bounded = bound_jsonish_value(value)
    if isinstance(bounded, list):
        return bounded[:_MAX_ITEMS]
    return [bounded]


def normalized_session(value: Any) -> dict[str, str]:
    """Return one normalized replay session mapping with stable defaults."""
    mapping = as_mapping(value, field_name="session")
    return {
        "agent_id": _bounded_text(mapping.get("agent_id"), default=_DEFAULT_SESSION["agent_id"]),
        "user_id": _bounded_text(mapping.get("user_id"), default=_DEFAULT_SESSION["user_id"]),
        "session_id": _bounded_text(
            mapping.get("session_id"),
            default=_DEFAULT_SESSION["session_id"],
        ),
        "thread_id": _bounded_text(mapping.get("thread_id"), default=_DEFAULT_SESSION["thread_id"]),
    }


def replay_artifact_mapping(value: Any) -> dict[str, Any]:
    """Coerce one fuzz value into a compact replay-artifact mapping."""
    artifact = _artifact_mapping(value)
    return {
        **artifact,
        "session": normalized_session(artifact.get("session")),
        "events": coerce_replay_events(artifact),
    }


def coerce_replay_events(value: Any) -> list[dict[str, Any]]:
    """Return 1-6 compact replay event payloads with unique imported ids."""
    artifact = _artifact_mapping(value)
    raw_events: Any
    if "events" in artifact:
        raw_events = artifact["events"]
    elif "payloads" in artifact:
        raw_events = artifact["payloads"]
    else:
        raw_events = [artifact]
    if isinstance(raw_events, list):
        events = raw_events[:_MAX_ITEMS]
    else:
        events = as_list(raw_events)[:_MAX_ITEMS]
    if not events:
        events = [artifact]

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(events):
        template = copy.deepcopy(_DEFAULT_REPLAY_EVENT_TEMPLATES[index % len(_DEFAULT_REPLAY_EVENT_TEMPLATES)])
        if isinstance(item, dict):
            candidate = _merge_event_mapping(template, _event_mapping(item))
            if "event_type" not in item:
                candidate["event_type"] = template["event_type"]
        else:
            candidate = template
            candidate["payload"] = {"value": _bound_event_payload(item)}
        candidate["event_id"] = unique_event_id(candidate.get("event_id"), index=index)
        candidate["ts"] = _bounded_text(candidate.get("ts"), default=_default_ts(index))
        candidate["source"] = _bounded_text(candidate.get("source"), default="atheris")
        candidate["tags"] = [
            _bounded_text(tag, default=f"tag-{tag_index}")
            for tag_index, tag in enumerate(as_list(candidate.get("tags", []))[:_MAX_ITEMS])
        ]
        normalized.append(candidate)
    return normalized


def unique_event_id(value: Any, *, index: int) -> str:
    """Return a stable imported event id with an index suffix."""
    base = _bounded_text(value, default="fuzz-event")
    return f"{base}-{index}"


def normalized_identifier(value: Any, *, default: str) -> str:
    """Return one reusable compact identifier."""
    return _bounded_text(value, default=default)


def active_state_projection_mapping(value: Any) -> dict[str, Any]:
    """Return compact active-state sub-mappings for projection and digest harnesses."""
    root = _artifact_mapping(value)
    return {
        "private_working_memory": _merge_mapping(
            _DEFAULT_ACTIVE_STATE_INPUTS["private_working_memory"],
            root.get("private_working_memory"),
        ),
        "active_situation_model": _merge_mapping(
            _DEFAULT_ACTIVE_STATE_INPUTS["active_situation_model"],
            root.get("active_situation_model"),
        ),
        "scene_world_state": _merge_mapping(
            _DEFAULT_ACTIVE_STATE_INPUTS["scene_world_state"],
            root.get("scene_world_state"),
        ),
    }


def autonomy_digest_inputs(value: Any) -> dict[str, Any]:
    """Return compact autonomy-ledger and agenda mappings for digest harnesses."""
    root = _artifact_mapping(value)
    return {
        "autonomy_ledger": _merge_mapping(
            _DEFAULT_AUTONOMY_LEDGER,
            root.get("autonomy_ledger"),
        ),
        "agenda": _merge_mapping(_DEFAULT_AGENDA, root.get("agenda")),
    }


def reevaluation_digest_inputs(value: Any) -> dict[str, Any]:
    """Return compact autonomy-ledger plus recent event mappings for reevaluation digests."""
    root = _artifact_mapping(value)
    return {
        "autonomy_ledger": _merge_mapping(
            _DEFAULT_AUTONOMY_LEDGER,
            root.get("autonomy_ledger"),
        ),
        "recent_events": coerce_brain_event_dicts(root.get("recent_events", root.get("events"))),
    }


def coerce_brain_event_dicts(
    value: Any,
    *,
    default_templates: tuple[dict[str, Any], ...] | None = None,
) -> list[dict[str, Any]]:
    """Return compact brain-event payloads suitable for digest harness hydration."""
    raw_events = as_list(value)[:_MAX_ITEMS] if value is not None else []
    if not raw_events:
        raw_events = list(default_templates or _DEFAULT_RECENT_EVENT_TEMPLATES)
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_events[:_MAX_ITEMS]):
        template = copy.deepcopy(_DEFAULT_RECENT_EVENT_TEMPLATES[index % len(_DEFAULT_RECENT_EVENT_TEMPLATES)])
        if isinstance(item, dict):
            candidate = _merge_event_record_mapping(template, item, index=index)
        else:
            candidate = template
            candidate["payload"] = {"value": bound_jsonish_value(item)}
        normalized.append(candidate)
    return normalized


def operator_artifact_inputs(value: Any) -> dict[str, Any]:
    """Return compact cross-phase operator-artifact inputs for digest harnesses."""
    root = _artifact_mapping(value)
    agenda_override = root.get("agenda")
    agenda = _merge_mapping(
        _DEFAULT_OPERATOR_ARTIFACT_AGENDA,
        agenda_override,
    )
    agenda["goals"] = _coerce_projection_record_list(
        agenda_override.get("goals") if isinstance(agenda_override, dict) else None,
        default_templates=tuple(_DEFAULT_OPERATOR_ARTIFACT_AGENDA["goals"]),
    )

    commitment_override = root.get("commitment_projection")
    commitment_projection = _merge_mapping(
        _DEFAULT_OPERATOR_ARTIFACT_COMMITMENT_PROJECTION,
        commitment_override,
    )
    for key in (
        "active_commitments",
        "deferred_commitments",
        "blocked_commitments",
        "recent_terminal_commitments",
    ):
        commitment_projection[key] = _coerce_projection_record_list(
            commitment_override.get(key) if isinstance(commitment_override, dict) else None,
            default_templates=tuple(_DEFAULT_OPERATOR_ARTIFACT_COMMITMENT_PROJECTION.get(key, [])),
        )

    procedural_skills_override = root.get("procedural_skills")
    procedural_skills = _merge_mapping(
        {
            "scope_type": "thread",
            "scope_id": _DEFAULT_SESSION["thread_id"],
            "skills": [],
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        procedural_skills_override,
    )
    procedural_skills["skills"] = _coerce_projection_record_list(
        procedural_skills_override.get("skills")
        if isinstance(procedural_skills_override, dict)
        else None,
    )

    procedural_traces_override = root.get("procedural_traces")
    procedural_traces = _merge_mapping(
        {
            "scope_type": "thread",
            "scope_id": _DEFAULT_SESSION["thread_id"],
            "traces": [],
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        procedural_traces_override,
    )
    procedural_traces["traces"] = _coerce_projection_record_list(
        procedural_traces_override.get("traces")
        if isinstance(procedural_traces_override, dict)
        else None,
    )
    return {
        "autonomy_ledger": _merge_mapping(
            _DEFAULT_AUTONOMY_LEDGER,
            root.get("autonomy_ledger"),
        ),
        "agenda": agenda,
        "commitment_projection": commitment_projection,
        "recent_events": coerce_brain_event_dicts(
            root.get("recent_events", root.get("events")),
            default_templates=_DEFAULT_OPERATOR_ARTIFACT_EVENT_TEMPLATES,
        ),
        "procedural_skills": procedural_skills,
        "procedural_traces": procedural_traces,
        "reflection_cycles": _coerce_named_records(
            root.get("reflection_cycles"),
            default_templates=_DEFAULT_REFLECTION_CYCLE_TEMPLATES,
        ),
        "memory_exports": _coerce_named_records(
            root.get("memory_exports"),
            default_templates=_DEFAULT_MEMORY_EXPORT_TEMPLATES,
        ),
    }


def scene_event_inputs(value: Any) -> list[dict[str, Any]]:
    """Return compact scene/perception events for multimodal distillation seams."""
    root = _artifact_mapping(value)
    events = coerce_brain_event_dicts(
        root.get("recent_events", root.get("events")),
        default_templates=_DEFAULT_SCENE_EVENT_TEMPLATES,
    )
    scene_event_types = {
        "perception.observed",
        "scene.changed",
        "engagement.changed",
        "attention.changed",
    }
    if any(str(event.get("event_type", "")).strip() in scene_event_types for event in events):
        return events
    fallback = list(_DEFAULT_SCENE_EVENT_TEMPLATES)
    keep = max(0, _MAX_ITEMS - len(fallback))
    return [*events[:keep], *fallback][: _MAX_ITEMS]


def multimodal_autobiography_inputs(value: Any) -> list[dict[str, Any]]:
    """Return bounded scene-episode autobiography rows with redaction coverage."""
    root = _artifact_mapping(value)
    return _coerce_projection_record_list(
        root.get("multimodal_autobiography"),
        default_templates=_DEFAULT_MULTIMODAL_AUTOBIOGRAPHY_TEMPLATES,
    )


def _consume_jsonish_value(provider: atheris.FuzzedDataProvider, *, depth: int) -> Any:
    if depth >= _MAX_DEPTH:
        return _consume_leaf(provider)
    kind = provider.ConsumeIntInRange(0, 5)
    if kind == 0:
        return None
    if kind == 1:
        return bool(provider.ConsumeIntInRange(0, 1))
    if kind == 2:
        return provider.ConsumeIntInRange(-10_000, 10_000)
    if kind == 3:
        return provider.ConsumeUnicodeNoSurrogates(_MAX_STRING_LENGTH)
    if kind == 4:
        length = provider.ConsumeIntInRange(0, _MAX_ITEMS)
        return [
            _consume_jsonish_value(provider, depth=depth + 1)
            for _ in range(length)
        ]
    length = provider.ConsumeIntInRange(0, _MAX_ITEMS)
    mapping: dict[str, Any] = {}
    for index in range(length):
        key = provider.ConsumeUnicodeNoSurrogates(_MAX_STRING_LENGTH).strip() or f"key_{index}"
        mapping[f"{key}:{index}"] = _consume_jsonish_value(provider, depth=depth + 1)
    return mapping


def _consume_leaf(provider: atheris.FuzzedDataProvider) -> Any:
    kind = provider.ConsumeIntInRange(0, 3)
    if kind == 0:
        return None
    if kind == 1:
        return bool(provider.ConsumeIntInRange(0, 1))
    if kind == 2:
        return provider.ConsumeIntInRange(-10_000, 10_000)
    return provider.ConsumeUnicodeNoSurrogates(_MAX_STRING_LENGTH)


def _artifact_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return as_mapping(value, field_name="artifact")
    bounded: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= _MAX_ITEMS:
            break
        normalized_key = str(key).strip()[:_MAX_STRING_LENGTH] or f"key_{index}"
        if normalized_key in {
            "events",
            "payloads",
            "session",
            "private_working_memory",
            "active_situation_model",
            "scene_world_state",
            "multimodal_autobiography",
            "autonomy_ledger",
            "agenda",
            "commitment_projection",
            "procedural_skills",
            "procedural_traces",
            "recent_events",
            "reflection_cycles",
            "memory_exports",
        }:
            bounded[normalized_key] = item
            continue
        bounded[normalized_key] = bound_jsonish_value(item)
    return bounded


def _coerce_leaf(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return max(min(int(value), 10_000), -10_000)
    if isinstance(value, float):
        return round(max(min(float(value), 10_000.0), -10_000.0), 3)
    return str(value)[:_MAX_STRING_LENGTH]


def _bounded_text(value: Any, *, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return (text or default)[:_MAX_STRING_LENGTH]


def _default_ts(index: int) -> str:
    return f"2026-01-01T00:00:0{index % 10}+00:00"


def _merge_event_mapping(template: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(template)
    for key, value in override.items():
        if (
            key == "payload"
            and isinstance(merged.get("payload"), dict)
            and isinstance(value, dict)
        ):
            merged["payload"] = {**merged["payload"], **value}
            continue
        merged[key] = bound_jsonish_value(value)
    return merged


def _event_mapping(value: dict[str, Any]) -> dict[str, Any]:
    bounded: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= _MAX_ITEMS:
            break
        normalized_key = str(key).strip()[:_MAX_STRING_LENGTH] or f"key_{index}"
        if normalized_key == "payload":
            bounded[normalized_key] = _bound_event_payload(item)
            continue
        if normalized_key == "tags":
            bounded[normalized_key] = [bound_jsonish_value(tag) for tag in as_list(item)[:_MAX_ITEMS]]
            continue
        bounded[normalized_key] = bound_jsonish_value(item)
    return bounded


def _bound_event_payload(value: Any, *, depth: int = 0) -> Any:
    if depth >= _MAX_DEPTH + 1:
        return _coerce_leaf(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return max(min(int(value), 10_000), -10_000)
    if isinstance(value, float):
        return round(max(min(float(value), 10_000.0), -10_000.0), 3)
    if isinstance(value, str):
        return value[:_MAX_STRING_LENGTH]
    if isinstance(value, list):
        return [_bound_event_payload(item, depth=depth + 1) for item in value[:_MAX_ITEMS]]
    if isinstance(value, dict):
        bounded: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_ITEMS:
                break
            normalized_key = str(key).strip()[:_MAX_STRING_LENGTH] or f"key_{index}"
            bounded[normalized_key] = _bound_event_payload(item, depth=depth + 1)
        return bounded
    return _coerce_leaf(value)


def _merge_mapping(default: dict[str, Any], override: Any) -> dict[str, Any]:
    merged = copy.deepcopy(default)
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        normalized_key = str(key).strip()[:_MAX_STRING_LENGTH] or key
        if (
            isinstance(merged.get(normalized_key), dict)
            and isinstance(value, dict)
        ):
            merged[normalized_key] = {
                **merged[normalized_key],
                **bound_jsonish_value(value),
            }
            continue
        merged[normalized_key] = bound_jsonish_value(value)
    return merged


def _merge_event_record_mapping(template: dict[str, Any], override: dict[str, Any], *, index: int) -> dict[str, Any]:
    merged = copy.deepcopy(template)
    for key, value in override.items():
        normalized_key = str(key).strip()[:_MAX_STRING_LENGTH] or key
        if normalized_key == "payload":
            merged["payload"] = _bound_event_payload(value)
            continue
        if normalized_key == "tags":
            merged["tags"] = [bound_jsonish_value(tag) for tag in as_list(value)[:_MAX_ITEMS]]
            continue
        merged[normalized_key] = bound_jsonish_value(value)
    merged["id"] = int(merged.get("id", index + 1) or (index + 1))
    merged["event_id"] = unique_event_id(merged.get("event_id"), index=index)
    merged["event_type"] = _bounded_text(merged.get("event_type"), default="goal.candidate.created")
    merged["ts"] = _bounded_text(merged.get("ts"), default=_default_ts(index))
    merged["agent_id"] = _bounded_text(merged.get("agent_id"), default=_DEFAULT_SESSION["agent_id"])
    merged["user_id"] = _bounded_text(merged.get("user_id"), default=_DEFAULT_SESSION["user_id"])
    merged["session_id"] = _bounded_text(
        merged.get("session_id"),
        default=_DEFAULT_SESSION["session_id"],
    )
    merged["thread_id"] = _bounded_text(merged.get("thread_id"), default=_DEFAULT_SESSION["thread_id"])
    merged["source"] = _bounded_text(merged.get("source"), default="atheris")
    merged["confidence"] = float(merged.get("confidence", 1.0))
    return merged


def _coerce_named_records(
    value: Any,
    *,
    default_templates: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    records = as_list(value)[:_MAX_ITEMS] if value is not None else []
    if not records:
        records = list(default_templates)
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(records[:_MAX_ITEMS]):
        template = copy.deepcopy(default_templates[index % len(default_templates)])
        if isinstance(item, dict):
            template = _merge_mapping(template, item)
        normalized.append(as_mapping(template, field_name=f"record_{index}"))
    return normalized


def _coerce_projection_record_list(
    value: Any,
    *,
    default_templates: tuple[dict[str, Any], ...] = (),
) -> list[dict[str, Any]]:
    records = as_list(value)[:_MAX_ITEMS] if value is not None else []
    if not records and default_templates:
        records = list(default_templates)
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(records[:_MAX_ITEMS]):
        template = copy.deepcopy(default_templates[index % len(default_templates)]) if default_templates else {}
        if isinstance(item, dict):
            template = _merge_mapping(template, item) if template else as_mapping(item)
        normalized.append(as_mapping(template, field_name=f"record_{index}"))
    return normalized
