"""Brain-side adapter descriptors for predictive, embodied, and governance backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BrainAdapterDescriptor:
    """Stable identity and capability summary for one brain-side adapter."""

    backend_id: str
    backend_version: str
    capabilities: tuple[str, ...]
    degraded_mode_id: str | None = None
    default_timeout_ms: int | None = None

    def supports(self, capability: str) -> bool:
        """Return whether one declared capability is supported."""
        return capability in self.capabilities


LOCAL_PERCEPTION_DESCRIPTOR = BrainAdapterDescriptor(
    backend_id="local_perception",
    backend_version="v1",
    capabilities=("presence_detection", "scene_enrichment"),
    degraded_mode_id="unavailable_result",
    default_timeout_ms=3000,
)

LOCAL_WORLD_MODEL_DESCRIPTOR = BrainAdapterDescriptor(
    backend_id="local_world_model",
    backend_version="v1",
    capabilities=("prediction_proposal",),
    degraded_mode_id="empty_proposals",
    default_timeout_ms=250,
)

LOCAL_EMBODIED_POLICY_DESCRIPTOR = BrainAdapterDescriptor(
    backend_id="local_robot_head_policy",
    backend_version="v1",
    capabilities=("status", "embodied_action_execution"),
    degraded_mode_id="preview_only",
    default_timeout_ms=5000,
)

from blink.brain.adapters.live_controller import (  # noqa: E402
    LiveRoutingController,
    LiveRoutingControllerResult,
    LiveRoutingControllerStatus,
    LiveRoutingDecisionStatus,
    LiveRoutingPlanStatus,
)
from blink.brain.adapters.live_routing import (  # noqa: E402
    AdapterRoutingPlan,
    AdapterRoutingState,
    RolloutDecisionRecord,
    active_routing_plan_for_family,
    apply_rollout_decision,
    build_adapter_routing_plan,
    build_rollout_decision,
)
from blink.brain.adapters.rollout_budget import RolloutBudget, build_rollout_budget  # noqa: E402

__all__ = [
    "AdapterRoutingPlan",
    "AdapterRoutingState",
    "BrainAdapterDescriptor",
    "LOCAL_EMBODIED_POLICY_DESCRIPTOR",
    "LOCAL_PERCEPTION_DESCRIPTOR",
    "LOCAL_WORLD_MODEL_DESCRIPTOR",
    "LiveRoutingController",
    "LiveRoutingControllerResult",
    "LiveRoutingControllerStatus",
    "LiveRoutingDecisionStatus",
    "LiveRoutingPlanStatus",
    "RolloutBudget",
    "RolloutDecisionRecord",
    "active_routing_plan_for_family",
    "apply_rollout_decision",
    "build_adapter_routing_plan",
    "build_rollout_budget",
    "build_rollout_decision",
]
