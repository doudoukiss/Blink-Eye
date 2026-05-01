from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from blink.brain._executive import BrainPlanningDraft, BrainPlanningOutcome
from blink.brain.capability_registry import build_brain_capability_registry
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.executive import BrainExecutive
from blink.brain.projections import (
    BrainActiveSituationProjection,
    BrainGoalFamily,
    BrainGoalStatus,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language

pytestmark = pytest.mark.brain_stateful


class _PlanningSurfaceBuilder:
    """Wrap the real context surface with mutable scene-mode policy signals only."""

    def __init__(self, *, store: BrainStore, session_ids, capability_registry):
        self.scene_mode = "healthy"
        self._builder = BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
            capability_registry=capability_registry,
        )

    def build(self, **kwargs):
        base = self._builder.build(**kwargs)
        return replace(
            base,
            active_situation_model=BrainActiveSituationProjection(
                scope_type=base.active_situation_model.scope_type,
                scope_id=base.active_situation_model.scope_id,
                updated_at=base.generated_at,
            ),
            commitment_projection=replace(
                base.commitment_projection,
                blocked_commitments=[],
                deferred_commitments=[],
            ),
            scene_world_state=replace(
                base.scene_world_state,
                degraded_mode=self.scene_mode,
                degraded_reason_codes=(
                    ["scene_stale"]
                    if self.scene_mode == "limited"
                    else ["scene_pipeline_failed"]
                    if self.scene_mode == "unavailable"
                    else []
                ),
            ),
        )


class _SequencedPlanner:
    def __init__(self):
        self.request_counter = 0

    async def __call__(self, request):
        self.request_counter += 1
        draft = BrainPlanningDraft.from_dict(
            {
                "summary": f"Stateful plan draft {self.request_counter}.",
                "remaining_steps": [
                    {
                        "capability_id": "maintenance.review_memory_health",
                        "arguments": {},
                    },
                    {
                        "capability_id": "reporting.record_maintenance_note",
                        "arguments": {},
                    },
                ],
                "review_policy": "auto_adopt_ok",
                "procedural_origin": "fresh_draft",
                "details": {"stateful_request_counter": self.request_counter},
            }
        )
        assert draft is not None
        return draft


class PolicyCoupledPlanningStateMachine(RuleBasedStateMachine):
    """Exercise planning review floors through the single coordinator path."""

    def __init__(self):
        super().__init__()
        self._tmpdir = TemporaryDirectory()
        self.store = BrainStore(path=Path(self._tmpdir.name) / "brain.db")
        self.session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="stateful-policy-planning",
        )
        self.capability_registry = build_brain_capability_registry(language=Language.EN)
        self.surface_builder = _PlanningSurfaceBuilder(
            store=self.store,
            session_ids=self.session_ids,
            capability_registry=self.capability_registry,
        )
        self.executive = BrainExecutive(
            store=self.store,
            session_resolver=lambda: self.session_ids,
            capability_registry=self.capability_registry,
            planning_callback=_SequencedPlanner(),
            context_surface_builder=self.surface_builder,
        )
        self.goal_counter = 0
        self.last_result = None

    def teardown(self):
        self.store.close()
        self._tmpdir.cleanup()

    @rule()
    def set_healthy(self):
        self.surface_builder.scene_mode = "healthy"
        self.last_result = None

    @rule()
    def set_limited(self):
        self.surface_builder.scene_mode = "limited"
        self.last_result = None

    @rule()
    def set_unavailable(self):
        self.surface_builder.scene_mode = "unavailable"
        self.last_result = None

    @rule()
    def request_plan(self):
        self.goal_counter += 1
        goal_id = self.executive.create_commitment_goal(
            title=f"Stateful planning goal {self.goal_counter}",
            intent="maintenance.review",
            source="stateful",
            goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
            goal_status=BrainGoalStatus.OPEN.value,
            details={"survive_restart": True},
        )
        self.last_result = asyncio.run(self.executive.request_plan_proposal(goal_id=goal_id))

    @invariant()
    def healthy_policy_keeps_auto_adopt(self):
        if self.surface_builder.scene_mode != "healthy" or self.last_result is None:
            return
        assert self.last_result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
        assert self.last_result.decision.executive_policy is not None
        assert self.last_result.decision.executive_policy["action_posture"] == "allow"

    @invariant()
    def limited_policy_downgrades_to_review(self):
        if self.surface_builder.scene_mode != "limited" or self.last_result is None:
            return
        assert self.last_result.outcome == BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value
        assert "policy_conservative_deferral" in self.last_result.decision.reason_codes
        assert self.last_result.proposal.details["procedural"]["policy"]["effect"] == "advisory_only"

    @invariant()
    def unavailable_policy_forces_operator_review(self):
        if self.surface_builder.scene_mode != "unavailable" or self.last_result is None:
            return
        assert self.last_result.outcome == BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value
        assert "policy_blocked_action" in self.last_result.decision.reason_codes
        assert self.last_result.proposal.details["procedural"]["policy"]["effect"] == "blocked"


TestPolicyCoupledPlanningStateMachine = PolicyCoupledPlanningStateMachine.TestCase
TestPolicyCoupledPlanningStateMachine.settings = settings(
    max_examples=12,
    stateful_step_count=6,
    deadline=None,
)
