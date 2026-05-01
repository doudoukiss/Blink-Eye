"""Canonical task-mode policy for the Blink situation compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BrainContextTask(str, Enum):
    """Task-specific context-selection modes."""

    REPLY = "reply"
    PLANNING = "planning"
    SIM_REHEARSAL = "sim_rehearsal"
    RECALL = "recall"
    REFLECTION = "reflection"
    CRITIQUE = "critique"
    WAKE = "wake"
    REEVALUATION = "reevaluation"
    OPERATOR_AUDIT = "operator_audit"
    GOVERNANCE_REVIEW = "governance_review"


class BrainContextTraceVerbosity(str, Enum):
    """Trace detail policy for one compiler mode."""

    STANDARD = "standard"
    VERBOSE = "verbose"


class BrainContextSurfaceQueryStrategy(str, Enum):
    """Fallback query strategy for context-surface retrieval."""

    REPLY = "reply"
    PLANNING = "planning"
    WAKE = "wake"
    REEVALUATION = "reevaluation"
    AUDIT = "audit"


class BrainContextEdgeProfile(str, Enum):
    """Allowed continuity-edge profile for one mode."""

    CONTINUITY = "continuity"
    PLANNING = "planning"


class BrainContextSceneEpisodePolicy(str, Enum):
    """Task-specific packet eligibility policy for scene episodes."""

    DISABLED = "disabled"
    CONSERVATIVE = "conservative"
    ALLOWED = "allowed"


class BrainContextPredictionPolicy(str, Enum):
    """Task-specific packet eligibility policy for predictive items."""

    DISABLED = "disabled"
    CONSERVATIVE = "conservative"
    ALLOWED = "allowed"


@dataclass(frozen=True)
class BrainContextModePolicy:
    """Single-source task policy for the bounded situation compiler."""

    task: BrainContextTask
    surface_query_strategy: BrainContextSurfaceQueryStrategy
    include_historical_claims: bool
    static_section_keys: tuple[str, ...]
    dynamic_section_keys: tuple[str, ...]
    max_tokens: int
    dynamic_token_reserve: int
    section_caps: dict[str, int] = field(default_factory=dict)
    anchor_caps: dict[str, int] = field(default_factory=dict)
    allowed_edge_profile: BrainContextEdgeProfile = BrainContextEdgeProfile.CONTINUITY
    includes_continuity_items: bool = True
    uses_planning_anchors: bool = False
    continuity_section_key: str | None = None
    prefer_unresolved_for_nonclean_continuity: bool = False
    trace_verbosity: BrainContextTraceVerbosity = BrainContextTraceVerbosity.STANDARD
    scene_episode_policy: BrainContextSceneEpisodePolicy = BrainContextSceneEpisodePolicy.DISABLED
    scene_episode_cap: int = 0
    prediction_policy: BrainContextPredictionPolicy = BrainContextPredictionPolicy.DISABLED
    prediction_cap: int = 0

    def as_dict(self) -> dict[str, object]:
        """Serialize one mode policy."""
        return {
            "task": self.task.value,
            "surface_query_strategy": self.surface_query_strategy.value,
            "include_historical_claims": self.include_historical_claims,
            "static_section_keys": list(self.static_section_keys),
            "dynamic_section_keys": list(self.dynamic_section_keys),
            "max_tokens": self.max_tokens,
            "dynamic_token_reserve": self.dynamic_token_reserve,
            "section_caps": dict(self.section_caps),
            "anchor_caps": dict(self.anchor_caps),
            "allowed_edge_profile": self.allowed_edge_profile.value,
            "includes_continuity_items": self.includes_continuity_items,
            "uses_planning_anchors": self.uses_planning_anchors,
            "continuity_section_key": self.continuity_section_key,
            "prefer_unresolved_for_nonclean_continuity": (
                self.prefer_unresolved_for_nonclean_continuity
            ),
            "trace_verbosity": self.trace_verbosity.value,
            "scene_episode_policy": self.scene_episode_policy.value,
            "scene_episode_cap": self.scene_episode_cap,
            "prediction_policy": self.prediction_policy.value,
            "prediction_cap": self.prediction_cap,
        }


_REPLY_POLICY = BrainContextModePolicy(
    task=BrainContextTask.REPLY,
    surface_query_strategy=BrainContextSurfaceQueryStrategy.REPLY,
    include_historical_claims=False,
    static_section_keys=(
        "policy",
        "identity",
        "style",
        "persona_expression",
        "teaching_knowledge",
        "capabilities",
        "presence",
        "scene",
        "engagement",
        "working_context",
        "private_working_memory",
        "agenda",
        "heartbeat",
        "user_profile",
        "relationship_state",
    ),
    dynamic_section_keys=(
        "active_state",
        "active_continuity",
        "unresolved_state",
        "recent_changes",
    ),
    max_tokens=1100,
    dynamic_token_reserve=48,
    section_caps={
        "autobiography": 3,
        "current_claims": 6,
        "recent_memory": 4,
    },
    anchor_caps={
        "max_anchors": 4,
        "max_graph_nodes": 12,
        "max_dossiers": 2,
        "max_current_items": 6,
        "max_history_items": 2,
        "max_hops": 2,
    },
    continuity_section_key="active_continuity",
    prefer_unresolved_for_nonclean_continuity=True,
    scene_episode_policy=BrainContextSceneEpisodePolicy.DISABLED,
    prediction_policy=BrainContextPredictionPolicy.DISABLED,
)
_PLANNING_POLICY = BrainContextModePolicy(
    task=BrainContextTask.PLANNING,
    surface_query_strategy=BrainContextSurfaceQueryStrategy.PLANNING,
    include_historical_claims=False,
    static_section_keys=(
        "agenda",
        "commitment_projection",
        "private_working_memory",
        "internal_capabilities",
        "core_blocks",
    ),
    dynamic_section_keys=(
        "planning_anchors",
        "active_state",
        "unresolved_state",
        "relevant_continuity",
        "recent_changes",
    ),
    max_tokens=900,
    dynamic_token_reserve=72,
    section_caps={
        "current_claims": 8,
        "recent_memory": 4,
        "core_blocks": 4,
    },
    anchor_caps={
        "max_anchors": 5,
        "max_graph_nodes": 16,
        "max_dossiers": 2,
        "max_current_items": 8,
        "max_history_items": 3,
        "max_hops": 3,
    },
    allowed_edge_profile=BrainContextEdgeProfile.PLANNING,
    uses_planning_anchors=True,
    continuity_section_key="relevant_continuity",
    prefer_unresolved_for_nonclean_continuity=True,
    scene_episode_policy=BrainContextSceneEpisodePolicy.CONSERVATIVE,
    scene_episode_cap=1,
    prediction_policy=BrainContextPredictionPolicy.CONSERVATIVE,
    prediction_cap=2,
)
_SIM_REHEARSAL_POLICY = BrainContextModePolicy(
    task=BrainContextTask.SIM_REHEARSAL,
    surface_query_strategy=BrainContextSurfaceQueryStrategy.PLANNING,
    include_historical_claims=True,
    static_section_keys=(
        "agenda",
        "commitment_projection",
        "private_working_memory",
        "internal_capabilities",
        "core_blocks",
    ),
    dynamic_section_keys=(
        "planning_anchors",
        "active_state",
        "unresolved_state",
        "relevant_continuity",
        "recent_changes",
    ),
    max_tokens=950,
    dynamic_token_reserve=80,
    section_caps={
        "current_claims": 8,
        "recent_memory": 4,
        "core_blocks": 4,
    },
    anchor_caps={
        "max_anchors": 5,
        "max_graph_nodes": 16,
        "max_dossiers": 2,
        "max_current_items": 8,
        "max_history_items": 3,
        "max_hops": 3,
    },
    allowed_edge_profile=BrainContextEdgeProfile.PLANNING,
    uses_planning_anchors=True,
    continuity_section_key="relevant_continuity",
    prefer_unresolved_for_nonclean_continuity=True,
    scene_episode_policy=BrainContextSceneEpisodePolicy.CONSERVATIVE,
    scene_episode_cap=2,
    prediction_policy=BrainContextPredictionPolicy.ALLOWED,
    prediction_cap=4,
)
_RECALL_POLICY = BrainContextModePolicy(
    task=BrainContextTask.RECALL,
    surface_query_strategy=BrainContextSurfaceQueryStrategy.REPLY,
    include_historical_claims=True,
    static_section_keys=(
        "current_claims",
        "historical_claims",
        "claim_provenance",
        "core_blocks",
    ),
    dynamic_section_keys=("active_state", "relevant_continuity", "recent_changes"),
    max_tokens=1000,
    dynamic_token_reserve=56,
    section_caps={
        "current_claims": 8,
        "historical_claims": 6,
        "claim_provenance": 6,
        "core_blocks": 4,
        "recent_memory": 6,
        "episodic_fallback": 4,
    },
    anchor_caps={
        "max_anchors": 4,
        "max_graph_nodes": 14,
        "max_dossiers": 3,
        "max_current_items": 7,
        "max_history_items": 4,
        "max_hops": 2,
    },
    continuity_section_key="relevant_continuity",
    scene_episode_policy=BrainContextSceneEpisodePolicy.DISABLED,
    prediction_policy=BrainContextPredictionPolicy.DISABLED,
)
_REFLECTION_POLICY = BrainContextModePolicy(
    task=BrainContextTask.REFLECTION,
    surface_query_strategy=BrainContextSurfaceQueryStrategy.PLANNING,
    include_historical_claims=True,
    static_section_keys=(
        "current_claims",
        "historical_claims",
        "claim_supersessions",
        "claim_provenance",
        "autobiography",
        "commitment_projection",
        "memory_health",
    ),
    dynamic_section_keys=(
        "active_state",
        "relevant_continuity",
        "unresolved_state",
        "recent_changes",
    ),
    max_tokens=1200,
    dynamic_token_reserve=72,
    section_caps={
        "current_claims": 10,
        "historical_claims": 8,
        "claim_provenance": 8,
        "claim_supersessions": 8,
        "autobiography": 6,
        "recent_memory": 6,
        "episodic_fallback": 4,
    },
    anchor_caps={
        "max_anchors": 5,
        "max_graph_nodes": 16,
        "max_dossiers": 3,
        "max_current_items": 7,
        "max_history_items": 5,
        "max_hops": 3,
    },
    continuity_section_key="relevant_continuity",
    scene_episode_policy=BrainContextSceneEpisodePolicy.ALLOWED,
    scene_episode_cap=3,
    prediction_policy=BrainContextPredictionPolicy.ALLOWED,
    prediction_cap=4,
)
_CRITIQUE_POLICY = BrainContextModePolicy(
    task=BrainContextTask.CRITIQUE,
    surface_query_strategy=BrainContextSurfaceQueryStrategy.PLANNING,
    include_historical_claims=False,
    static_section_keys=(
        "agenda",
        "commitment_projection",
        "internal_capabilities",
        "memory_health",
        "current_claims",
    ),
    dynamic_section_keys=(
        "planning_anchors",
        "active_state",
        "unresolved_state",
        "recent_changes",
    ),
    max_tokens=850,
    dynamic_token_reserve=48,
    section_caps={
        "current_claims": 4,
        "recent_memory": 3,
    },
    anchor_caps={
        "max_anchors": 5,
        "max_graph_nodes": 14,
        "max_dossiers": 2,
        "max_current_items": 7,
        "max_history_items": 3,
        "max_hops": 2,
    },
    allowed_edge_profile=BrainContextEdgeProfile.PLANNING,
    includes_continuity_items=False,
    uses_planning_anchors=True,
    prefer_unresolved_for_nonclean_continuity=True,
    scene_episode_policy=BrainContextSceneEpisodePolicy.DISABLED,
    prediction_policy=BrainContextPredictionPolicy.DISABLED,
)
_WAKE_POLICY = BrainContextModePolicy(
    task=BrainContextTask.WAKE,
    surface_query_strategy=BrainContextSurfaceQueryStrategy.WAKE,
    include_historical_claims=False,
    static_section_keys=(
        "agenda",
        "heartbeat",
        "commitment_projection",
        "private_working_memory",
        "internal_capabilities",
    ),
    dynamic_section_keys=(
        "planning_anchors",
        "unresolved_state",
        "active_state",
        "recent_changes",
    ),
    max_tokens=900,
    dynamic_token_reserve=64,
    section_caps={
        "current_claims": 6,
        "core_blocks": 3,
    },
    anchor_caps={
        "max_anchors": 4,
        "max_graph_nodes": 14,
        "max_dossiers": 2,
        "max_current_items": 7,
        "max_history_items": 3,
        "max_hops": 2,
    },
    allowed_edge_profile=BrainContextEdgeProfile.PLANNING,
    uses_planning_anchors=True,
    prefer_unresolved_for_nonclean_continuity=True,
    scene_episode_policy=BrainContextSceneEpisodePolicy.DISABLED,
    prediction_policy=BrainContextPredictionPolicy.DISABLED,
)
_REEVALUATION_POLICY = BrainContextModePolicy(
    task=BrainContextTask.REEVALUATION,
    surface_query_strategy=BrainContextSurfaceQueryStrategy.REEVALUATION,
    include_historical_claims=False,
    static_section_keys=(
        "agenda",
        "commitment_projection",
        "private_working_memory",
        "memory_health",
        "current_claims",
    ),
    dynamic_section_keys=(
        "unresolved_state",
        "active_state",
        "relevant_continuity",
        "recent_changes",
    ),
    max_tokens=950,
    dynamic_token_reserve=72,
    section_caps={
        "current_claims": 8,
        "recent_memory": 4,
        "core_blocks": 4,
    },
    anchor_caps={
        "max_anchors": 5,
        "max_graph_nodes": 14,
        "max_dossiers": 2,
        "max_current_items": 7,
        "max_history_items": 4,
        "max_hops": 2,
    },
    allowed_edge_profile=BrainContextEdgeProfile.PLANNING,
    continuity_section_key="relevant_continuity",
    prefer_unresolved_for_nonclean_continuity=True,
    scene_episode_policy=BrainContextSceneEpisodePolicy.DISABLED,
    prediction_policy=BrainContextPredictionPolicy.DISABLED,
)
_OPERATOR_AUDIT_POLICY = BrainContextModePolicy(
    task=BrainContextTask.OPERATOR_AUDIT,
    surface_query_strategy=BrainContextSurfaceQueryStrategy.AUDIT,
    include_historical_claims=True,
    static_section_keys=(
        "heartbeat",
        "agenda",
        "commitment_projection",
        "current_claims",
        "historical_claims",
        "claim_provenance",
        "claim_supersessions",
        "memory_health",
    ),
    dynamic_section_keys=(
        "unresolved_state",
        "relevant_continuity",
        "active_state",
        "recent_changes",
    ),
    max_tokens=1200,
    dynamic_token_reserve=96,
    section_caps={
        "current_claims": 10,
        "historical_claims": 8,
        "claim_provenance": 8,
        "claim_supersessions": 8,
        "recent_memory": 6,
        "core_blocks": 4,
    },
    anchor_caps={
        "max_anchors": 5,
        "max_graph_nodes": 16,
        "max_dossiers": 3,
        "max_current_items": 7,
        "max_history_items": 5,
        "max_hops": 3,
    },
    allowed_edge_profile=BrainContextEdgeProfile.PLANNING,
    continuity_section_key="relevant_continuity",
    prefer_unresolved_for_nonclean_continuity=True,
    trace_verbosity=BrainContextTraceVerbosity.VERBOSE,
    scene_episode_policy=BrainContextSceneEpisodePolicy.ALLOWED,
    scene_episode_cap=4,
    prediction_policy=BrainContextPredictionPolicy.ALLOWED,
    prediction_cap=4,
)
_GOVERNANCE_REVIEW_POLICY = BrainContextModePolicy(
    task=BrainContextTask.GOVERNANCE_REVIEW,
    surface_query_strategy=BrainContextSurfaceQueryStrategy.AUDIT,
    include_historical_claims=True,
    static_section_keys=(
        "current_claims",
        "historical_claims",
        "claim_provenance",
        "claim_supersessions",
        "core_blocks",
        "memory_health",
    ),
    dynamic_section_keys=(
        "relevant_continuity",
        "recent_changes",
        "unresolved_state",
    ),
    max_tokens=1100,
    dynamic_token_reserve=88,
    section_caps={
        "current_claims": 10,
        "historical_claims": 8,
        "claim_provenance": 8,
        "claim_supersessions": 8,
        "core_blocks": 4,
        "recent_memory": 6,
    },
    anchor_caps={
        "max_anchors": 5,
        "max_graph_nodes": 16,
        "max_dossiers": 3,
        "max_current_items": 6,
        "max_history_items": 6,
        "max_hops": 3,
    },
    continuity_section_key="relevant_continuity",
    prefer_unresolved_for_nonclean_continuity=True,
    trace_verbosity=BrainContextTraceVerbosity.VERBOSE,
    scene_episode_policy=BrainContextSceneEpisodePolicy.ALLOWED,
    scene_episode_cap=4,
    prediction_policy=BrainContextPredictionPolicy.ALLOWED,
    prediction_cap=4,
)

_MODE_POLICIES = {
    policy.task.value: policy
    for policy in (
        _REPLY_POLICY,
        _PLANNING_POLICY,
        _SIM_REHEARSAL_POLICY,
        _RECALL_POLICY,
        _REFLECTION_POLICY,
        _CRITIQUE_POLICY,
        _WAKE_POLICY,
        _REEVALUATION_POLICY,
        _OPERATOR_AUDIT_POLICY,
        _GOVERNANCE_REVIEW_POLICY,
    )
}


def get_brain_context_mode_policy(task: BrainContextTask | str) -> BrainContextModePolicy:
    """Return one canonical mode policy."""
    task_value = task.value if isinstance(task, BrainContextTask) else str(task).strip()
    policy = _MODE_POLICIES.get(task_value)
    if policy is None:
        raise ValueError(f"Unknown BrainContextTask: {task_value}")
    return policy


def all_brain_context_tasks() -> tuple[BrainContextTask, ...]:
    """Return the supported context task modes in stable order."""
    return tuple(
        policy.task
        for policy in _MODE_POLICIES.values()
        if policy.task != BrainContextTask.SIM_REHEARSAL
    )


__all__ = [
    "BrainContextEdgeProfile",
    "BrainContextModePolicy",
    "BrainContextPredictionPolicy",
    "BrainContextSceneEpisodePolicy",
    "BrainContextSurfaceQueryStrategy",
    "BrainContextTask",
    "BrainContextTraceVerbosity",
    "all_brain_context_tasks",
    "get_brain_context_mode_policy",
]
