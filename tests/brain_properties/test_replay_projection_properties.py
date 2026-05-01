from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from blink.brain.core import BrainCoreStore
from blink.brain.replay import BrainReplayHarness
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language
from tests.brain_properties._replay_property_helpers import (
    PRESENCE_SCOPE_KEY,
    build_core_replay_scenario,
    load_replay_scenario_from_artifact,
    materialize_core_events,
    materialize_duplicate_noise_events,
    normalized_artifact_payload,
    normalized_context_surface_snapshot,
    normalized_core_projection_bundle_from_store,
    replay_case_strategy,
)

pytestmark = pytest.mark.brain_property

_COMMON_HEALTH_CHECKS = [
    HealthCheck.function_scoped_fixture,
    HealthCheck.too_slow,
]
_CORE_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=_COMMON_HEALTH_CHECKS,
)
_ROUNDTRIP_SETTINGS = settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=_COMMON_HEALTH_CHECKS,
)


@given(case=replay_case_strategy())
@_CORE_SETTINGS
def test_normalized_equivalent_streams_rebuild_identical_core_projections(case):
    # This proves normalized-equivalent streams rebuild the same projected state.
    first_events = materialize_core_events(case, stream_name="baseline")
    second_events = materialize_core_events(
        case,
        stream_name="equivalent",
        equivalent_stream=True,
    )

    with TemporaryDirectory() as tmpdir:
        first_store = BrainCoreStore(path=Path(tmpdir) / "first.db")
        second_store = BrainCoreStore(path=Path(tmpdir) / "second.db")
        try:
            for event in first_events:
                first_store.import_brain_event(event)
            for event in second_events:
                second_store.import_brain_event(event)
            first_store.rebuild_brain_projections()
            second_store.rebuild_brain_projections()

            assert normalized_core_projection_bundle_from_store(
                first_store
            ) == normalized_core_projection_bundle_from_store(second_store)
        finally:
            first_store.close()
            second_store.close()


@given(
    case=replay_case_strategy(),
    duplicate_kinds=st.lists(
        st.sampled_from(("reevaluation", "wake")),
        min_size=1,
        max_size=3,
    ),
)
@_CORE_SETTINGS
def test_duplicate_noop_events_do_not_change_semantic_core_projections(case, duplicate_kinds):
    # Heartbeat is excluded because it is defined by the last seen event and changes on any append.
    base_events = materialize_core_events(case, stream_name="base")
    duplicate_events = base_events + materialize_duplicate_noise_events(
        case,
        duplicate_kinds=duplicate_kinds,
    )

    with TemporaryDirectory() as tmpdir:
        base_store = BrainCoreStore(path=Path(tmpdir) / "base.db")
        duplicate_store = BrainCoreStore(path=Path(tmpdir) / "duplicate.db")
        try:
            for event in base_events:
                base_store.import_brain_event(event)
            for event in duplicate_events:
                duplicate_store.import_brain_event(event)
            base_store.rebuild_brain_projections()
            duplicate_store.rebuild_brain_projections()

            assert normalized_core_projection_bundle_from_store(
                base_store,
                include_heartbeat=False,
            ) == normalized_core_projection_bundle_from_store(
                duplicate_store,
                include_heartbeat=False,
            )
        finally:
            base_store.close()
            duplicate_store.close()


@given(case=replay_case_strategy())
@_ROUNDTRIP_SETTINGS
def test_replay_artifacts_roundtrip_without_semantic_drift(case):
    # This proves exported replay artifacts preserve replay semantics after reload.
    scenario = build_core_replay_scenario(case, stream_name="roundtrip")

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        source_store = BrainStore(path=tmpdir_path / "source.db")
        harness = BrainReplayHarness(store=source_store)
        try:
            first = harness.replay(
                scenario,
                output_store_path=tmpdir_path / "first.db",
                artifact_path=tmpdir_path / "first.json",
                presence_scope_key=PRESENCE_SCOPE_KEY,
                language=Language.EN,
            )
            loaded_scenario = load_replay_scenario_from_artifact(first.artifact_path)
            second = harness.replay(
                loaded_scenario,
                output_store_path=tmpdir_path / "second.db",
                artifact_path=tmpdir_path / "second.json",
                presence_scope_key=PRESENCE_SCOPE_KEY,
                language=Language.EN,
            )

            first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
            second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

            assert normalized_context_surface_snapshot(
                first.context_surface
            ) == normalized_context_surface_snapshot(second.context_surface)
            assert normalized_artifact_payload(first_payload) == normalized_artifact_payload(
                second_payload
            )

            expected_event_ids = [event.event_id for event in scenario.events]
            expected_event_types = [event.event_type for event in scenario.events]
            assert first_payload["scenario"]["event_count"] == len(scenario.events)
            assert second_payload["scenario"]["event_count"] == len(scenario.events)
            assert [event["event_id"] for event in first_payload["events"]] == expected_event_ids
            assert [event["event_id"] for event in second_payload["events"]] == expected_event_ids
            assert (
                [event["event_type"] for event in first_payload["events"]]
                == expected_event_types
            )
            assert (
                [event["event_type"] for event in second_payload["events"]]
                == expected_event_types
            )
            assert [
                (event["event_id"], event["event_type"], event["ts"])
                for event in first_payload["events"]
            ] == [
                (event["event_id"], event["event_type"], event["ts"])
                for event in second_payload["events"]
            ]
            assert first_payload["qa"]["matched"] is True
            assert second_payload["qa"]["matched"] is True
        finally:
            source_store.close()
