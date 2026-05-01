from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from blink.brain.capabilities import CapabilityRegistry
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.projections import (
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainCommitmentWakeRouteKind,
    BrainWakeCondition,
    BrainWakeConditionKind,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language

pytestmark = pytest.mark.brain_stateful


class _WakeSurfaceBuilder:
    """Wrap the real context surface with one mutable degraded-scene override."""

    def __init__(self, *, store: BrainStore, session_ids):
        self.scene_mode = "healthy"
        self._builder = BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
            capability_registry=CapabilityRegistry(),
        )

    def build(self, **kwargs):
        base = self._builder.build(**kwargs)
        return replace(
            base,
            scene_world_state=replace(
                base.scene_world_state,
                degraded_mode=self.scene_mode,
                degraded_reason_codes=(
                    ["scene_stale"] if self.scene_mode == "limited" else []
                ),
            ),
        )


class PolicyCoupledWakeStateMachine(RuleBasedStateMachine):
    """Exercise keep-waiting vs release through the single executive wake path."""

    def __init__(self):
        super().__init__()
        self._tmpdir = TemporaryDirectory()
        self.store = BrainStore(path=Path(self._tmpdir.name) / "brain.db")
        self.session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="stateful-policy-wake",
        )
        self.surface_builder = _WakeSurfaceBuilder(
            store=self.store,
            session_ids=self.session_ids,
        )
        self.executive = BrainExecutive(
            store=self.store,
            session_resolver=lambda: self.session_ids,
            capability_registry=CapabilityRegistry(),
            context_surface_builder=self.surface_builder,
        )
        self.executive.create_commitment_goal(
            title="Stateful wake follow-up",
            intent="narrative.commitment",
            source="stateful",
            details={"summary": "Revisit once the thread is idle."},
        )
        self.commitment = self.store.list_executive_commitments(
            user_id=self.session_ids.user_id,
            limit=4,
        )[0]
        self.executive.defer_commitment(
            commitment_id=self.commitment.commitment_id,
            reason=BrainBlockedReason(
                kind=BrainBlockedReasonKind.WAITING_USER.value,
                summary="Wait for the thread to settle.",
            ),
            wake_conditions=[
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.THREAD_IDLE.value,
                    summary="Wake when the thread is idle.",
                )
            ],
        )
        self.last_result = None

    def teardown(self):
        self.store.close()
        self._tmpdir.cleanup()

    @rule()
    def set_limited(self):
        self.surface_builder.scene_mode = "limited"
        self.last_result = None

    @rule()
    def set_healthy(self):
        self.surface_builder.scene_mode = "healthy"
        self.last_result = None

    @rule()
    def run_wake_router(self):
        self.last_result = asyncio.run(
            self.executive.run_commitment_wake_router(boundary_kind="startup_recovery")
        )

    @invariant()
    def policy_holds_are_traceable(self):
        if self.surface_builder.scene_mode != "limited" or self.last_result is None:
            return
        if not self.last_result.progressed:
            return
        assert self.last_result.route_kind == BrainCommitmentWakeRouteKind.KEEP_WAITING.value
        assert "policy_conservative_deferral" in self.last_result.reason_codes
        assert "scene_limited" in self.last_result.reason_codes
        assert self.last_result.executive_policy is not None
        assert self.last_result.executive_policy["action_posture"] == "defer"

    @invariant()
    def healthy_release_stays_legacy_compatible(self):
        if self.surface_builder.scene_mode != "healthy" or self.last_result is None:
            return
        if not self.last_result.progressed:
            return
        if self.last_result.route_kind == BrainCommitmentWakeRouteKind.KEEP_WAITING.value:
            assert self.last_result.reason != "policy_conservative_deferral"
        if self.last_result.route_kind == BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value:
            wake_events = [
                event
                for event in self.store.recent_brain_events(
                    user_id=self.session_ids.user_id,
                    thread_id=self.session_ids.thread_id,
                    limit=8,
                )
                if event.event_type == BrainEventType.COMMITMENT_WAKE_TRIGGERED
            ]
            assert wake_events
            assert wake_events[0].payload["routing"]["route_kind"] == (
                BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value
            )


TestPolicyCoupledWakeStateMachine = PolicyCoupledWakeStateMachine.TestCase
TestPolicyCoupledWakeStateMachine.settings = settings(
    max_examples=12,
    stateful_step_count=6,
    deadline=None,
)
