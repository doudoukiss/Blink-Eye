from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.memory_v2 import (
    BrainAutobiographyEntryKind,
    BrainMultimodalAutobiographyPrivacyClass,
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
from blink.brain.replay_support import materialize_replayed_events
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore

pytestmark = pytest.mark.brain_stateful


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-01-01T{hour:02d}:{minute:02d}:{second:02d}+00:00"


class MultimodalMemoryRetentionStateMachine(RuleBasedStateMachine):
    """Exercise scene distillation, retention, and redaction with replay-safe rebuilds."""

    def __init__(self):
        super().__init__()
        self._tmpdir = TemporaryDirectory()
        self.store = BrainStore(path=Path(self._tmpdir.name) / "brain.db")
        self.session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="stateful-multimodal",
        )
        self.current_entry_id: str | None = None
        self.known_entry_ids: set[str] = set()
        self.counter = 0

    def teardown(self):
        self.store.close()
        self._tmpdir.cleanup()

    def _event_context(self, correlation_id: str):
        return self.store._memory_event_context(
            user_id=self.session_ids.user_id,
            thread_id=self.session_ids.thread_id,
            agent_id=self.session_ids.agent_id,
            session_id=self.session_ids.session_id,
            source="stateful",
            correlation_id=correlation_id,
        )

    def _import_scene_event(self, *, second: int, event_type: str) -> BrainEventRecord:
        event = BrainEventRecord(
            id=0,
            event_id=f"evt-stateful-{second}-{event_type.replace('.', '-')}",
            event_type=event_type,
            ts=_ts(second),
            agent_id=self.session_ids.agent_id,
            user_id=self.session_ids.user_id,
            session_id=self.session_ids.session_id,
            thread_id=self.session_ids.thread_id,
            source="stateful",
            correlation_id=f"scene-{second}",
            causal_parent_id=None,
            confidence=1.0,
            payload_json=json.dumps(
                {"presence_scope_key": "browser:presence", "summary": f"scene-{second}"},
                ensure_ascii=False,
                sort_keys=True,
            ),
            tags_json="[]",
        )
        self.store.import_brain_event(event)
        return event

    def _projection(
        self,
        *,
        second: int,
        include_person: bool,
        degraded: bool,
        blocked: bool,
        object_state: str,
    ) -> BrainSceneWorldProjection:
        updated_at = _ts(second)
        projection = BrainSceneWorldProjection(
            scope_type="presence",
            scope_id="browser:presence",
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
                            summary="Ada is present.",
                            state=BrainSceneWorldRecordState.ACTIVE.value,
                            evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                            confidence=0.9,
                            source_event_ids=[f"evt-stateful-{second}-scene-changed"],
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
                    summary="Desk visible.",
                    state=object_state,
                    evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                    confidence=0.72,
                    source_event_ids=[f"evt-stateful-{second}-scene-changed"],
                    observed_at=updated_at,
                    updated_at=updated_at,
                ),
            ],
            affordances=[
                BrainSceneWorldAffordanceRecord(
                    affordance_id="scene-aff-1",
                    entity_id="scene-desk-1",
                    capability_family="inspect",
                    summary="Desk can be inspected.",
                    availability=(
                        BrainSceneWorldAffordanceAvailability.BLOCKED.value
                        if blocked
                        else BrainSceneWorldAffordanceAvailability.AVAILABLE.value
                    ),
                    confidence=0.68,
                    source_event_ids=[f"evt-stateful-{second}-scene-changed"],
                    observed_at=updated_at,
                    updated_at=updated_at,
                )
            ],
        )
        projection.sync_lists()
        return projection

    def _entries(self):
        return self.store.autobiographical_entries(
            scope_type="presence",
            scope_id="browser:presence",
            entry_kinds=(BrainAutobiographyEntryKind.SCENE_EPISODE.value,),
            statuses=("current", "superseded"),
            modalities=("scene_world",),
            limit=16,
        )

    @rule()
    def distill_initial_scene(self):
        self.counter += 1
        scene_event = self._import_scene_event(second=10 * self.counter, event_type=BrainEventType.SCENE_CHANGED)
        projection = self._projection(
            second=10 * self.counter,
            include_person=True,
            degraded=False,
            blocked=False,
            object_state=BrainSceneWorldRecordState.ACTIVE.value,
        )
        entry = self.store.refresh_scene_episode_autobiography(
            user_id=self.session_ids.user_id,
            thread_id=self.session_ids.thread_id,
            session_id=self.session_ids.session_id,
            agent_id=self.session_ids.agent_id,
            presence_scope_key="browser:presence",
            scene_world_state=projection,
            recent_events=self.store.recent_brain_events(
                user_id=self.session_ids.user_id,
                thread_id=self.session_ids.thread_id,
                limit=8,
            ),
            source_event_id=scene_event.event_id,
            updated_at=projection.updated_at,
            event_context=self._event_context(f"scene-{self.counter}"),
        )
        assert entry is not None
        self.current_entry_id = entry.entry_id
        self.known_entry_ids.add(entry.entry_id)

    @rule()
    @precondition(lambda self: self.current_entry_id is not None)
    def distill_meaningful_scene_change(self):
        self.counter += 1
        scene_event = self._import_scene_event(second=10 * self.counter, event_type=BrainEventType.SCENE_CHANGED)
        projection = self._projection(
            second=10 * self.counter,
            include_person=True,
            degraded=True,
            blocked=True,
            object_state=BrainSceneWorldRecordState.STALE.value,
        )
        entry = self.store.refresh_scene_episode_autobiography(
            user_id=self.session_ids.user_id,
            thread_id=self.session_ids.thread_id,
            session_id=self.session_ids.session_id,
            agent_id=self.session_ids.agent_id,
            presence_scope_key="browser:presence",
            scene_world_state=projection,
            recent_events=self.store.recent_brain_events(
                user_id=self.session_ids.user_id,
                thread_id=self.session_ids.thread_id,
                limit=8,
            ),
            source_event_id=scene_event.event_id,
            updated_at=projection.updated_at,
            event_context=self._event_context(f"scene-change-{self.counter}"),
        )
        assert entry is not None
        self.current_entry_id = entry.entry_id
        self.known_entry_ids.add(entry.entry_id)

    @rule()
    @precondition(lambda self: self.current_entry_id is not None)
    def reclassify_retention(self):
        entry = self.store.reclassify_autobiographical_entry_retention(
            self.current_entry_id,
            retention_class="durable",
            source_event_id=f"evt-retention-{self.counter}",
            reason_codes=[BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value],
            event_context=self._event_context(f"retention-{self.counter}"),
        )
        assert entry.retention_class == "durable"

    @rule()
    @precondition(lambda self: self.current_entry_id is not None)
    def redact_current_scene(self):
        entry = self.store.redact_autobiographical_entry(
            self.current_entry_id,
            redacted_summary="Redacted scene episode.",
            source_event_id=f"evt-redact-{self.counter}",
            reason_codes=[BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value],
            event_context=self._event_context(f"redact-{self.counter}"),
        )
        typed = parse_multimodal_autobiography_record(entry)
        assert typed is not None
        assert typed.privacy_class == BrainMultimodalAutobiographyPrivacyClass.REDACTED.value

    @invariant()
    def only_one_current_scene_episode_exists(self):
        assert sum(1 for entry in self._entries() if entry.status == "current") <= 1

    @invariant()
    def known_entries_remain_queryable(self):
        current_ids = {entry.entry_id for entry in self._entries()}
        assert self.known_entry_ids >= (self.known_entry_ids & current_ids)
        for entry_id in self.known_entry_ids:
            assert any(entry.entry_id == entry_id for entry in self._entries())

    @invariant()
    def redacted_entries_never_leak_original_summary(self):
        for entry in self._entries():
            typed = parse_multimodal_autobiography_record(entry)
            if typed is None or typed.privacy_class != BrainMultimodalAutobiographyPrivacyClass.REDACTED.value:
                continue
            encoded = json.dumps(typed.as_dict(), ensure_ascii=False, sort_keys=True)
            assert "Ada is present." not in encoded
            assert typed.review_state == BrainClaimReviewState.RESOLVED.value

    @invariant()
    def replay_rebuild_matches_multimodal_state(self):
        replay_tmpdir = TemporaryDirectory()
        replay_store = BrainStore(path=Path(replay_tmpdir.name) / "brain.db")
        try:
            events = list(reversed(self.store.recent_brain_events(
                user_id=self.session_ids.user_id,
                thread_id=self.session_ids.thread_id,
                limit=128,
            )))
            for event in events:
                replay_store.import_brain_event(event)
            materialize_replayed_events(
                store=replay_store,
                session_ids=self.session_ids,
                events=events,
            )
            left = [
                record.as_dict()
                for entry in self._entries()
                if (record := parse_multimodal_autobiography_record(entry)) is not None
            ]
            right = [
                record.as_dict()
                for entry in replay_store.autobiographical_entries(
                    scope_type="presence",
                    scope_id="browser:presence",
                    entry_kinds=(BrainAutobiographyEntryKind.SCENE_EPISODE.value,),
                    statuses=("current", "superseded"),
                    modalities=("scene_world",),
                    limit=16,
                )
                if (record := parse_multimodal_autobiography_record(entry)) is not None
            ]
            assert left == right
        finally:
            replay_store.close()
            replay_tmpdir.cleanup()


TestMultimodalMemoryRetentionStateMachine = MultimodalMemoryRetentionStateMachine.TestCase
TestMultimodalMemoryRetentionStateMachine.settings = settings(
    stateful_step_count=5,
    max_examples=12,
    deadline=None,
)
