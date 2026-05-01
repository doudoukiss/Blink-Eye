from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from blink.brain.projections import BrainPredictionResolutionKind
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from tests.test_brain_world_model import (
    _append_body_state,
    _append_goal_created,
    _append_goal_updated,
    _append_robot_action_outcome,
    _append_scene_changed,
    _ensure_blocks,
    _ts,
)

pytestmark = pytest.mark.brain_stateful


class PredictiveWorldModelStateMachine(RuleBasedStateMachine):
    """Exercise predictive lifecycle refreshes without duplicating active or terminal rows."""

    def __init__(self):
        super().__init__()
        self._tmpdir = TemporaryDirectory()
        self.store = BrainStore(path=Path(self._tmpdir.name) / "brain.db")
        _ensure_blocks(self.store)
        self.session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="stateful-predictive-world-model",
        )
        self.current_second = 0
        _append_body_state(self.store, self.session_ids, second=1)
        _append_scene_changed(self.store, self.session_ids, second=2)
        self.goal, _ = _append_goal_created(self.store, self.session_ids, second=3)
        self.current_second = 3

    def teardown(self):
        self.store.close()
        self._tmpdir.cleanup()

    def _tick(self, *, seconds: int = 1) -> int:
        self.current_second += seconds
        return self.current_second

    def _projection(self):
        return self.store.build_predictive_world_model_projection(
            user_id=self.session_ids.user_id,
            thread_id=self.session_ids.thread_id,
            reference_ts=_ts(self.current_second),
            agent_id=self.session_ids.agent_id,
            presence_scope_key="browser:presence",
        )

    @rule()
    def refresh_scene_stable(self):
        _append_scene_changed(self.store, self.session_ids, second=self._tick())

    @rule()
    def refresh_scene_blocked(self):
        _append_scene_changed(
            self.store,
            self.session_ids,
            second=self._tick(),
            include_person=False,
            affordance_availability="blocked",
            camera_fresh=False,
        )

    @rule()
    def refresh_goal_state(self):
        self.goal, _ = _append_goal_updated(
            self.store,
            self.session_ids,
            goal=self.goal,
            second=self._tick(),
        )
        _append_scene_changed(self.store, self.session_ids, second=self._tick())

    @rule(accepted=st.booleans())
    def record_robot_action_outcome(self, accepted):
        _append_robot_action_outcome(
            self.store,
            self.session_ids,
            second=self._tick(),
            accepted=accepted,
        )

    @rule()
    def expire_active_predictions(self):
        self.goal, _ = _append_goal_updated(
            self.store,
            self.session_ids,
            goal=self.goal,
            second=self._tick(seconds=40),
        )
        _append_scene_changed(
            self.store,
            self.session_ids,
            second=self._tick(),
            include_person=False,
            affordance_availability="blocked",
            camera_fresh=False,
        )

    @invariant()
    def active_predictions_stay_unique_by_subject(self):
        projection = self._projection()
        subject_keys = {
            (record.prediction_kind, record.subject_id) for record in projection.active_predictions
        }
        assert len(subject_keys) == len(projection.active_predictions)
        assert len(set(projection.active_prediction_ids)) == len(projection.active_prediction_ids)

    @invariant()
    def terminal_resolutions_do_not_duplicate_or_overlap_active_predictions(self):
        projection = self._projection()
        assert len(set(projection.recent_resolution_ids)) == len(projection.recent_resolution_ids)
        assert not (set(projection.active_prediction_ids) & set(projection.recent_resolution_ids))
        assert all(
            record.resolution_kind
            in {
                BrainPredictionResolutionKind.CONFIRMED.value,
                BrainPredictionResolutionKind.INVALIDATED.value,
                BrainPredictionResolutionKind.EXPIRED.value,
            }
            for record in projection.recent_resolutions
        )


TestPredictiveWorldModelStateMachine = PredictiveWorldModelStateMachine.TestCase
TestPredictiveWorldModelStateMachine.settings = settings(
    stateful_step_count=8,
    max_examples=20,
    deadline=None,
)
