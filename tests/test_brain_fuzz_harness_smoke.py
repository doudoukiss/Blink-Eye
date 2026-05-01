from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.brain_fuzz

_HARNESS_DIR = Path(__file__).resolve().parent / "fuzz" / "atheris"
_SEEDS = (
    b'{"session":{"agent_id":"agent-1","user_id":"user-1","session_id":"session-1","thread_id":"thread-1"},"events":[{"event_type":"memory.claim.recorded","payload":{"subject_entity_id":"user-1","predicate":"preference.like","object":{"value":"tea"}}}]}',
    b'{"projection":{"scope_type":"presence","scope_id":"browser:presence"},"private_working_memory":{"scope_type":"thread","scope_id":"thread-1","records":[{"record_id":"pwm-1","buffer_kind":"plan_assumption","summary":"Need review","state":"active","evidence_kind":"derived"}]},"active_situation_model":{"scope_type":"thread","scope_id":"thread-1","records":[{"record_id":"situation-1","record_kind":"plan_state","summary":"Need review","state":"active","evidence_kind":"derived"}]},"scene_world_state":{"scope_type":"presence","scope_id":"browser:presence","degraded_mode":"limited","degraded_reason_codes":["camera_occluded"],"entities":[{"entity_id":"person-1","entity_kind":"person","canonical_label":"operator","summary":"A person is visible near the desk.","state":"active","evidence_kind":"observed","source_event_ids":["evt-engagement-1"]},{"entity_id":"entity-1","entity_kind":"object","canonical_label":"cup","summary":"A cup is visible.","state":"active","evidence_kind":"observed","source_event_ids":["evt-scene-1"]}],"affordances":[{"affordance_id":"aff-1","entity_id":"entity-1","capability_family":"vision.inspect","summary":"Inspect the cup.","availability":"available","source_event_ids":["evt-scene-1"]}]},"multimodal_autobiography":[{"entry_id":"entry-1","scope_type":"presence","scope_id":"browser:presence","entry_kind":"scene_episode","modality":"scene_world","review_state":"requested","retention_class":"session","privacy_class":"sensitive","governance_reason_codes":["privacy_boundary","degraded_scene_evidence"],"source_presence_scope_key":"browser:presence","source_scene_entity_ids":["person-1","entity-1"],"source_scene_affordance_ids":["aff-1"],"rendered_summary":"limited; person near the desk; affordances: vision.inspect","content":{"summary":"limited; person near the desk; affordances: vision.inspect","degraded_mode":"limited","anchor_entity_ids":["person-1","entity-1"],"anchor_affordance_ids":["aff-1"],"supporting_event_ids":["evt-scene-1","evt-engagement-1"],"confidence_band":"medium","salience":1.65,"observed_at":"2026-01-01T00:00:02+00:00","updated_at":"2026-01-01T00:00:02+00:00"},"status":"current","salience":1.65,"source_event_ids":["evt-scene-1","evt-engagement-1"],"valid_from":"2026-01-01T00:00:02+00:00","created_at":"2026-01-01T00:00:02+00:00","updated_at":"2026-01-01T00:00:02+00:00"},{"entry_id":"entry-0","scope_type":"presence","scope_id":"browser:presence","entry_kind":"scene_episode","modality":"scene_world","review_state":"resolved","retention_class":"session","privacy_class":"redacted","governance_reason_codes":["privacy_boundary"],"redacted_at":"2026-01-01T00:00:03+00:00","source_presence_scope_key":"browser:presence","source_scene_entity_ids":["person-1"],"source_scene_affordance_ids":["aff-1"],"rendered_summary":"[redacted scene episode]","content":{"redacted":true,"redacted_summary":"[redacted scene episode]","degraded_mode":"limited","anchor_entity_ids":["person-1"],"anchor_affordance_ids":["aff-1"],"supporting_event_ids":["evt-scene-0"]},"status":"superseded","salience":1.2,"source_event_ids":["evt-scene-0"],"valid_from":"2025-12-31T23:59:59+00:00","valid_to":"2026-01-01T00:00:02+00:00","created_at":"2025-12-31T23:59:59+00:00","updated_at":"2026-01-01T00:00:03+00:00"}],"autonomy_ledger":{"current_candidates":[{"candidate_goal_id":"candidate-1","candidate_type":"presence_acknowledgement","source":"runtime","summary":"Ack presence","goal_family":"environment","urgency":0.7,"confidence":0.9,"initiative_class":"inspect_only"}]},"agenda":{"goals":[{"goal_id":"goal-1","title":"Ack presence","intent":"autonomy.presence_acknowledgement","source":"test"}]},"recent_events":[{"event_id":"evt-scene-1","event_type":"scene.changed","ts":"2026-01-01T00:00:01+00:00","payload":{"presence_scope_key":"browser:presence","degraded_mode":"limited"}},{"event_id":"evt-engagement-1","event_type":"engagement.changed","ts":"2026-01-01T00:00:02+00:00","payload":{"presence_scope_key":"browser:presence","focus_entity_id":"person-1"}}]}',
    b'{"predictive_world_model":{"scope_key":"thread-1","presence_scope_key":"browser:presence","active_predictions":[{"prediction_id":"prediction-1","prediction_kind":"scene_transition","subject_kind":"scene","subject_id":"browser:presence","scope_key":"thread-1","presence_scope_key":"browser:presence","summary":"Scene likely remains bounded.","predicted_state":{"scene_change_state":"stable","degraded_mode":"healthy"},"confidence":0.71,"confidence_band":"medium","supporting_event_ids":["evt-scene-1"],"backing_ids":["browser:presence"],"predicted_at":"2026-01-01T00:00:02+00:00","valid_from":"2026-01-01T00:00:02+00:00","valid_to":"2026-01-01T00:00:22+00:00","details":{"prediction_role":"opportunity"}}],"recent_resolutions":[{"prediction_id":"prediction-0","prediction_kind":"action_outcome","subject_kind":"action","subject_id":"goal-1","scope_key":"thread-1","presence_scope_key":"browser:presence","summary":"Next bounded action is likely safe to accept.","predicted_state":{"accepted":true},"confidence":0.72,"confidence_band":"medium","supporting_event_ids":["evt-scene-1"],"backing_ids":["goal-1"],"predicted_at":"2026-01-01T00:00:03+00:00","valid_from":"2026-01-01T00:00:03+00:00","valid_to":"2026-01-01T00:00:23+00:00","resolved_at":"2026-01-01T00:00:04+00:00","resolution_kind":"invalidated","resolution_event_ids":["evt-action-1"],"resolution_summary":"Observed state contradicted the short-horizon prediction.","details":{"prediction_role":"opportunity"}}]},"prediction":{"prediction_id":"prediction-2","prediction_kind":"engagement_drift","subject_kind":"engagement","subject_id":"engagement","scope_key":"thread-1","presence_scope_key":"browser:presence","summary":"User engagement likely stays attentive.","predicted_state":{"engagement_state":"engaged","person_present":"present"},"confidence":0.78,"confidence_band":"medium","supporting_event_ids":["evt-scene-1"],"backing_ids":["engagement"],"predicted_at":"2026-01-01T00:00:02+00:00","valid_from":"2026-01-01T00:00:02+00:00","valid_to":"2026-01-01T00:00:17+00:00","details":{"prediction_role":"opportunity"}},"active_situation_model":{"scope_type":"thread","scope_id":"thread-1","records":[{"record_id":"scene-1","record_kind":"scene_state","summary":"Scene is stable.","state":"active","evidence_kind":"observed","details":{"engagement_state":"engaged","person_present":"present","scene_change_state":"stable"}}]},"scene_world_state":{"scope_type":"presence","scope_id":"browser:presence","degraded_mode":"healthy","entities":[{"entity_id":"person-1","entity_kind":"person","canonical_label":"Ada","summary":"Ada is visible.","state":"active","evidence_kind":"observed","source_event_ids":["evt-scene-1"]}],"affordances":[{"affordance_id":"aff-1","entity_id":"person-1","capability_family":"inspect","summary":"Inspect the scene.","availability":"available","source_event_ids":["evt-scene-1"]}]},"events":[{"event_id":"evt-scene-1","event_type":"scene.changed","ts":"2026-01-01T00:00:02+00:00","payload":{"presence_scope_key":"browser:presence"}},{"event_id":"evt-action-1","event_type":"robot.action.outcome","ts":"2026-01-01T00:00:04+00:00","payload":{"accepted":false}}]}',
    b'{"session":{"agent_id":"agent-1","user_id":"user-1","session_id":"session-1","thread_id":"thread-1"},"agenda":{},"commitment_projection":{},"recent_events":[],"reflection_cycles":[{"cycle_id":"reflection-shell-1","trigger":"runtime_shell:manual","status":"completed"}],"memory_exports":[{"id":1,"export_kind":"continuity_audit","metadata":{"source":"runtime_shell","action_kind":"audit_export"}}]}',
)


class _StubFuzzedDataProvider:
    def __init__(self, data: bytes):
        self._data = data
        self._index = 0

    def _consume_byte(self) -> int:
        if self._index >= len(self._data):
            return 0
        value = self._data[self._index]
        self._index += 1
        return value

    def ConsumeIntInRange(self, minimum: int, maximum: int) -> int:
        if minimum >= maximum:
            return minimum
        span = (maximum - minimum) + 1
        return minimum + (self._consume_byte() % span)

    def ConsumeUnicodeNoSurrogates(self, max_length: int) -> str:
        length = min(max_length, self.ConsumeIntInRange(0, max_length))
        chars = []
        for _ in range(length):
            chars.append(chr(32 + (self._consume_byte() % 95)))
        return "".join(chars)


class _InstrumentImports:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_atheris_stub() -> types.ModuleType:
    module = types.ModuleType("atheris")
    module.FuzzedDataProvider = _StubFuzzedDataProvider
    module.instrument_imports = lambda: _InstrumentImports()
    module.Setup = lambda argv, test_one_input: (argv, test_one_input)
    module.Fuzz = lambda: None
    return module


def _import_harness(path: Path):
    module_name = f"_blink_harness_smoke_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("harness_path", sorted(_HARNESS_DIR.glob("fuzz_*.py")))
def test_atheris_harness_modules_import_and_exercise_testoneinput(monkeypatch, harness_path: Path):
    monkeypatch.syspath_prepend(str(_HARNESS_DIR))
    monkeypatch.setitem(sys.modules, "atheris", _make_atheris_stub())

    module = _import_harness(harness_path)

    assert hasattr(module, "TestOneInput")
    for seed in _SEEDS:
        module.TestOneInput(seed)
