from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.memory_v2 import (
    BrainAutobiographyEntryKind,
    BrainContinuityDossierKind,
    BrainMultimodalAutobiographyPrivacyClass,
    build_multimodal_autobiography_digest,
    parse_multimodal_autobiography_record,
)
from blink.brain.projections import (
    BrainClaimReviewState,
    BrainGovernanceReasonCode,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldAffordanceRecord,
    BrainSceneWorldEntityKind,
    BrainSceneWorldEntityRecord,
    BrainSceneWorldEvidenceKind,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=16,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-01-01T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def _projection(
    *,
    scope_id: str,
    source_event_ids: list[str],
    updated_at: str,
    include_person: bool,
    degraded: bool,
    blocked: bool,
) -> BrainSceneWorldProjection:
    projection = BrainSceneWorldProjection(
        scope_type="presence",
        scope_id=scope_id,
        degraded_mode="limited" if degraded else "healthy",
        degraded_reason_codes=(["scene_stale"] if degraded else []),
        updated_at=updated_at,
        entities=[
            *(
                [
                    BrainSceneWorldEntityRecord(
                        entity_id="scene-person-1",
                        entity_kind=BrainSceneWorldEntityKind.PERSON.value,
                        canonical_label="Ada",
                        summary="Ada is in view.",
                        state=BrainSceneWorldRecordState.ACTIVE.value,
                        evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                        confidence=0.9,
                        source_event_ids=list(source_event_ids),
                        observed_at=updated_at,
                        updated_at=updated_at,
                    )
                ]
                if include_person
                else []
            ),
            BrainSceneWorldEntityRecord(
                entity_id="scene-desk-1",
                entity_kind=BrainSceneWorldEntityKind.OBJECT.value,
                canonical_label="Desk",
                summary="A desk is visible.",
                state=BrainSceneWorldRecordState.ACTIVE.value,
                evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                confidence=0.7,
                source_event_ids=list(source_event_ids),
                observed_at=updated_at,
                updated_at=updated_at,
            ),
        ],
        affordances=[
            BrainSceneWorldAffordanceRecord(
                affordance_id="scene-aff-1",
                entity_id="scene-desk-1",
                capability_family="inspect",
                summary="The desk can be inspected.",
                availability=(
                    BrainSceneWorldAffordanceAvailability.BLOCKED.value
                    if blocked
                    else BrainSceneWorldAffordanceAvailability.AVAILABLE.value
                ),
                confidence=0.65,
                source_event_ids=list(source_event_ids),
                observed_at=updated_at,
                updated_at=updated_at,
            )
        ],
    )
    projection.sync_lists()
    return projection


def _import_scene_events(store: BrainStore, session_ids, *, start_second: int, include_attention: bool):
    scene_event = BrainEventRecord(
        id=0,
        event_id=f"evt-scene-{start_second}",
        event_type=BrainEventType.SCENE_CHANGED,
        ts=_ts(start_second),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="property",
        correlation_id=f"scene-{start_second}",
        causal_parent_id=None,
        confidence=1.0,
        payload_json=json.dumps(
            {"presence_scope_key": "browser:presence", "summary": "scene"},
            ensure_ascii=False,
            sort_keys=True,
        ),
        tags_json="[]",
    )
    store.import_brain_event(scene_event)
    if include_attention:
        store.import_brain_event(
            BrainEventRecord(
                id=0,
                event_id=f"evt-attention-{start_second}",
                event_type=BrainEventType.ATTENTION_CHANGED,
                ts=_ts(start_second + 1),
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="property",
                correlation_id=f"scene-{start_second}",
                causal_parent_id=scene_event.event_id,
                confidence=1.0,
                payload_json=json.dumps(
                    {"presence_scope_key": "browser:presence", "attention": "camera"},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                tags_json="[]",
            )
        )
    return scene_event


def _entry_context(store: BrainStore, session_ids, *, correlation_id: str) -> dict[str, str]:
    return store._memory_event_context(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source="property",
        correlation_id=correlation_id,
    )


def _build_store_state(
    *,
    include_person: bool,
    degraded: bool,
    blocked: bool,
    include_attention: bool,
):
    tmpdir = TemporaryDirectory()
    store = BrainStore(path=Path(tmpdir.name) / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="phase17-prop")
    scene_event = _import_scene_events(
        store,
        session_ids,
        start_second=10,
        include_attention=include_attention,
    )
    projection = _projection(
        scope_id="browser:presence",
        source_event_ids=[scene_event.event_id],
        updated_at=_ts(10),
        include_person=include_person,
        degraded=degraded,
        blocked=blocked,
    )
    entry = store.refresh_scene_episode_autobiography(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        session_id=session_ids.session_id,
        agent_id=session_ids.agent_id,
        presence_scope_key="browser:presence",
        scene_world_state=projection,
        recent_events=store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=8,
        ),
        source_event_id=scene_event.event_id,
        updated_at=projection.updated_at,
        event_context=_entry_context(store, session_ids, correlation_id="scene-prop"),
    )
    assert entry is not None
    return tmpdir, store, session_ids, projection


@given(
    include_person=st.booleans(),
    degraded=st.booleans(),
    blocked=st.booleans(),
    include_attention=st.booleans(),
)
@_SETTINGS
def test_multimodal_autobiography_rebuilds_identically_for_equivalent_inputs(
    include_person,
    degraded,
    blocked,
    include_attention,
):
    left_tmp, left_store, left_session_ids, left_projection = _build_store_state(
        include_person=include_person,
        degraded=degraded,
        blocked=blocked,
        include_attention=include_attention,
    )
    right_tmp, right_store, right_session_ids, right_projection = _build_store_state(
        include_person=include_person,
        degraded=degraded,
        blocked=blocked,
        include_attention=include_attention,
    )
    try:
        left_entries = [
            record.as_dict()
            for entry in left_store.autobiographical_entries(
                scope_type="presence",
                scope_id="browser:presence",
                entry_kinds=(BrainAutobiographyEntryKind.SCENE_EPISODE.value,),
                statuses=("current", "superseded"),
                modalities=("scene_world",),
                limit=8,
            )
            if (record := parse_multimodal_autobiography_record(entry)) is not None
        ]
        right_entries = [
            record.as_dict()
            for entry in right_store.autobiographical_entries(
                scope_type="presence",
                scope_id="browser:presence",
                entry_kinds=(BrainAutobiographyEntryKind.SCENE_EPISODE.value,),
                statuses=("current", "superseded"),
                modalities=("scene_world",),
                limit=8,
            )
            if (record := parse_multimodal_autobiography_record(entry)) is not None
        ]
        assert left_entries == right_entries
        assert left_projection.as_dict() == right_projection.as_dict()
    finally:
        left_store.close()
        left_tmp.cleanup()
        right_store.close()
        right_tmp.cleanup()


@given(
    include_person=st.booleans(),
    degraded=st.booleans(),
    blocked=st.booleans(),
    include_attention=st.booleans(),
)
@_SETTINGS
def test_multimodal_autobiography_stays_single_current_and_never_stores_raw_firehoses(
    include_person,
    degraded,
    blocked,
    include_attention,
):
    tmpdir, store, session_ids, projection = _build_store_state(
        include_person=include_person,
        degraded=degraded,
        blocked=blocked,
        include_attention=include_attention,
    )
    try:
        duplicate = store.refresh_scene_episode_autobiography(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            session_id=session_ids.session_id,
            agent_id=session_ids.agent_id,
            presence_scope_key="browser:presence",
            scene_world_state=projection,
            recent_events=store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=8,
            ),
            source_event_id="evt-duplicate",
            updated_at=projection.updated_at,
            event_context=_entry_context(store, session_ids, correlation_id="scene-duplicate"),
        )
        assert duplicate is not None
        entries = store.autobiographical_entries(
            scope_type="presence",
            scope_id="browser:presence",
            entry_kinds=(BrainAutobiographyEntryKind.SCENE_EPISODE.value,),
            statuses=("current", "superseded"),
            modalities=("scene_world",),
            limit=8,
        )
        assert sum(1 for entry in entries if entry.status == "current") == 1
        encoded = json.dumps([entry.content for entry in entries], ensure_ascii=False, sort_keys=True)
        assert "frame_bytes" not in encoded
        assert "vision_payload" not in encoded
        assert "audio_archive" not in encoded
    finally:
        store.close()
        tmpdir.cleanup()


@given(include_person=st.booleans(), degraded=st.booleans(), blocked=st.booleans())
@_SETTINGS
def test_multimodal_redaction_never_leaks_prior_content_and_graph_evidence_resolves(
    include_person,
    degraded,
    blocked,
):
    tmpdir, store, session_ids, projection = _build_store_state(
        include_person=include_person,
        degraded=degraded,
        blocked=blocked,
        include_attention=True,
    )
    try:
        current = store.latest_autobiographical_entry(
            scope_type="presence",
            scope_id="browser:presence",
            entry_kind=BrainAutobiographyEntryKind.SCENE_EPISODE.value,
        )
        assert current is not None
        original_summary = current.rendered_summary
        redacted = store.redact_autobiographical_entry(
            current.entry_id,
            redacted_summary="Redacted scene episode.",
            source_event_id="evt-redact",
            reason_codes=[BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value],
            event_context=_entry_context(store, session_ids, correlation_id="scene-redact"),
        )
        typed = parse_multimodal_autobiography_record(redacted)
        assert typed is not None
        assert typed.privacy_class == BrainMultimodalAutobiographyPrivacyClass.REDACTED.value
        assert typed.review_state == BrainClaimReviewState.RESOLVED.value
        assert original_summary not in json.dumps(typed.as_dict(), ensure_ascii=False, sort_keys=True)

        graph = store.build_continuity_graph(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            scope_type="user",
            scope_id=session_ids.user_id,
            scene_world_state=projection,
            presence_scope_key="browser:presence",
        )
        dossiers = store.build_continuity_dossiers(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            scope_type="user",
            scope_id=session_ids.user_id,
            continuity_graph=graph,
            scene_world_state=projection,
            presence_scope_key="browser:presence",
        )
        scene_world = next(
            record
            for record in dossiers.dossiers
            if record.kind == BrainContinuityDossierKind.SCENE_WORLD.value
        )
        graph_nodes = {node.node_id for node in graph.nodes}
        graph_edges = {edge.edge_id for edge in graph.edges}
        for evidence in [scene_world.summary_evidence, *(record.evidence for record in scene_world.recent_changes)]:
            assert set(evidence.graph_node_ids) <= graph_nodes
            assert set(evidence.graph_edge_ids) <= graph_edges

        digest = build_multimodal_autobiography_digest(
            store.autobiographical_entries(
                scope_type="presence",
                scope_id="browser:presence",
                entry_kinds=(BrainAutobiographyEntryKind.SCENE_EPISODE.value,),
                statuses=("current", "superseded"),
                limit=8,
            )
        )
        assert digest["entry_counts"]["privacy"].get("redacted", 0) >= 1
        assert digest["recent_redacted_rows"]
        assert original_summary not in json.dumps(digest, ensure_ascii=False, sort_keys=True)
    finally:
        store.close()
        tmpdir.cleanup()
