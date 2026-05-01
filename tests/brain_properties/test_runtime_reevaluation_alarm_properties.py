from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from tempfile import TemporaryDirectory

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from blink.brain.autonomy import (
    BrainCandidateGoal,
    BrainCandidateGoalSource,
    BrainInitiativeClass,
    BrainReevaluationConditionKind,
)
from blink.brain.identity import base_brain_system_prompt
from blink.brain.runtime import BrainRuntime
from blink.brain.session import resolve_brain_session_ids
from blink.transcriptions.language import Language

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=6,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


class _DummyLLM:
    def register_function(self, function_name, handler):
        return None


@dataclass(frozen=True)
class _WakeCandidateSpec:
    expiry_offset: int | None
    maintenance_offset: int | None


@st.composite
def _wake_candidate_specs(draw) -> list[_WakeCandidateSpec]:
    count = draw(st.integers(min_value=1, max_value=4))
    specs = [
        _WakeCandidateSpec(
            expiry_offset=draw(st.one_of(st.none(), st.integers(min_value=5, max_value=90))),
            maintenance_offset=draw(st.one_of(st.none(), st.integers(min_value=10, max_value=120))),
        )
        for _ in range(count)
    ]
    assume(any(spec.expiry_offset is not None or spec.maintenance_offset is not None for spec in specs))
    return specs


def _build_runtime(tempdir: TemporaryDirectory[str]) -> BrainRuntime:
    return BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.EN),
        language=Language.EN,
        runtime_kind="browser",
        session_resolver=lambda: resolve_brain_session_ids(runtime_kind="browser", client_id="fuzz"),
        llm=_DummyLLM(),
        brain_db_path=f"{tempdir.name}/brain.db",
    )


def _append_candidate(runtime: BrainRuntime, *, candidate: BrainCandidateGoal) -> None:
    session_ids = runtime.session_resolver()
    runtime.store.append_candidate_goal_created(
        candidate_goal=candidate,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="property",
    )


@given(specs=_wake_candidate_specs())
@_SETTINGS
def test_runtime_next_reevaluation_wake_selects_and_reuses_the_nearest_alarm(specs):
    # The runtime must track the nearest reevaluation wake and avoid rearming the same alarm key.
    tempdir = TemporaryDirectory()
    runtime = _build_runtime(tempdir)
    try:
        base = datetime(2099, 1, 1, tzinfo=UTC)
        expected_deadlines: list[tuple[str, datetime]] = []
        for index, spec in enumerate(specs):
            candidate = BrainCandidateGoal(
                candidate_goal_id=f"candidate-{index}",
                candidate_type="presence_acknowledgement",
                source=BrainCandidateGoalSource.RUNTIME.value,
                summary=f"Candidate {index}",
                goal_family="environment",
                urgency=0.6 + (index * 0.01),
                confidence=0.9,
                initiative_class=BrainInitiativeClass.INSPECT_ONLY.value,
                expires_at=(
                    (base + timedelta(seconds=spec.expiry_offset)).isoformat()
                    if spec.expiry_offset is not None
                    else None
                ),
                expected_reevaluation_condition="after the maintenance window opens"
                if spec.maintenance_offset is not None
                else None,
                expected_reevaluation_condition_kind=(
                    BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value
                    if spec.maintenance_offset is not None
                    else None
                ),
                expected_reevaluation_condition_details=(
                    {
                        "not_before": (
                            base + timedelta(seconds=spec.maintenance_offset)
                        ).isoformat()
                    }
                    if spec.maintenance_offset is not None
                    else {}
                ),
                created_at=(base - timedelta(seconds=5)).isoformat(),
            )
            _append_candidate(runtime, candidate=candidate)
            if spec.expiry_offset is not None:
                expected_deadlines.append(
                    (
                        BrainReevaluationConditionKind.TIME_REACHED.value,
                        base + timedelta(seconds=spec.expiry_offset),
                    )
                )
            if spec.maintenance_offset is not None:
                expected_deadlines.append(
                    (
                        BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
                        base + timedelta(seconds=spec.maintenance_offset),
                    )
                )

        expected_kind, expected_deadline = min(expected_deadlines, key=lambda item: item[1])
        assert runtime._next_reevaluation_wake() == (expected_kind, expected_deadline)

        async def _assert_refresh_stability() -> None:
            runtime._refresh_reevaluation_alarm()
            assert runtime._reevaluation_alarm_key == (expected_kind, expected_deadline.isoformat())
            previous_task = runtime._reevaluation_alarm_task
            runtime._refresh_reevaluation_alarm()
            assert runtime._reevaluation_alarm_task is previous_task
            runtime._cancel_reevaluation_alarm()

        asyncio.run(_assert_refresh_stability())
    finally:
        runtime.stop_background_maintenance()
        runtime.close()
        tempdir.cleanup()


@given(
    maintenance_offset=st.integers(min_value=15, max_value=120),
    expired_count=st.integers(min_value=1, max_value=3),
)
@_SETTINGS
def test_runtime_expiry_cleanup_rearms_the_next_available_maintenance_wake(
    maintenance_offset,
    expired_count,
):
    # Expiry cleanup must remove expired candidates and rearm the next maintenance-window wake.
    tempdir = TemporaryDirectory()
    runtime = _build_runtime(tempdir)
    try:
        now = datetime.now(UTC)
        session_ids = runtime.session_resolver()
        maintenance_ts = (now + timedelta(seconds=maintenance_offset)).isoformat()
        for index in range(expired_count):
            _append_candidate(
                runtime,
                candidate=BrainCandidateGoal(
                    candidate_goal_id=f"expired-{index}",
                    candidate_type="presence_acknowledgement",
                    source=BrainCandidateGoalSource.RUNTIME.value,
                    summary=f"Expired {index}",
                    goal_family="environment",
                    urgency=0.7,
                    confidence=0.9,
                    initiative_class=BrainInitiativeClass.INSPECT_ONLY.value,
                    expires_at=(now - timedelta(seconds=index + 1)).isoformat(),
                    created_at=(now - timedelta(seconds=10)).isoformat(),
                ),
            )
        maintenance_candidate = BrainCandidateGoal(
            candidate_goal_id="maintenance-next",
            candidate_type="maintenance_review_findings",
            source=BrainCandidateGoalSource.TIMER.value,
            summary="Wait for the maintenance window.",
            goal_family="memory_maintenance",
            urgency=0.6,
            confidence=1.0,
            initiative_class=BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,
            expires_at=(now + timedelta(minutes=5)).isoformat(),
            expected_reevaluation_condition="after the maintenance window opens",
            expected_reevaluation_condition_kind=BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
            expected_reevaluation_condition_details={"not_before": maintenance_ts},
            created_at=(now - timedelta(seconds=5)).isoformat(),
        )
        _append_candidate(runtime, candidate=maintenance_candidate)

        asyncio.run(
            runtime._run_reevaluation_alarm(
                kind=BrainReevaluationConditionKind.TIME_REACHED.value,
                deadline=now,
            )
        )

        ledger = runtime.store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
        assert ledger.candidate(maintenance_candidate.candidate_goal_id) is not None
        for index in range(expired_count):
            assert ledger.candidate(f"expired-{index}") is None
        assert runtime._next_reevaluation_wake() == (
            BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
            datetime.fromisoformat(maintenance_ts),
        )

        async def _assert_rearm() -> None:
            runtime._refresh_reevaluation_alarm()
            assert runtime._reevaluation_alarm_key == (
                BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
                maintenance_ts,
            )
            runtime._cancel_reevaluation_alarm()

        asyncio.run(_assert_rearm())
        assert runtime._reevaluation_alarm_key is None
        assert runtime._next_reevaluation_wake() == (
            BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
            datetime.fromisoformat(maintenance_ts),
        )
    finally:
        runtime.stop_background_maintenance()
        runtime.close()
        tempdir.cleanup()
