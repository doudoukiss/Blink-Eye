"""SQLite-backed local Blink brain store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import ValidationError

from blink.brain.active_situation_model import (
    build_active_situation_model_projection as derive_active_situation_model_projection,
)
from blink.brain.adapters.cards import BrainAdapterCard, build_default_adapter_cards
from blink.brain.adapters.world_model import (
    LocalDeterministicWorldModelAdapter,
    WorldModelAdapter,
    WorldModelAdapterRequest,
)
from blink.brain.autonomy import (
    BrainAutonomyLedgerEntry,
    BrainAutonomyLedgerProjection,
    BrainCandidateGoal,
    autonomy_decision_kind_for_event_type,
)
from blink.brain.bounded_json import (
    MAX_BRAIN_EVENT_PAYLOAD_JSON_CHARS,
    MAX_CLAIM_OBJECT_JSON_CHARS,
    dumps_bounded_json,
)
from blink.brain.core.store import BrainCoreStore
from blink.brain.embodied_executive import (
    append_embodied_action_envelope,
    append_embodied_execution_trace,
    append_embodied_intent,
    append_embodied_recovery,
    embodied_event_types,
)
from blink.brain.evals.adapter_promotion import (
    BrainAdapterBenchmarkComparisonReport,
    BrainAdapterGovernanceProjection,
    BrainAdapterPromotionDecision,
    append_adapter_benchmark_report,
    append_adapter_card,
    append_adapter_promotion_decision,
    apply_promotion_decision_to_card,
    with_card_benchmark_summary,
)
from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.memory_layers.episodic import BrainEpisodicMemoryRecord
from blink.brain.memory_layers.narrative import (
    BrainNarrativeMemoryRecord,
    narrative_default_staleness,
)
from blink.brain.memory_layers.retrieval import BrainMemorySearchResult
from blink.brain.memory_layers.semantic import (
    BrainSemanticMemoryRecord,
    semantic_contradiction_key,
    semantic_default_staleness,
)
from blink.brain.memory_v2 import (
    AutobiographyService,
    BrainAutobiographyEntryKind,
    BrainClaimRecord,
    BrainClaimSupersessionRecord,
    BrainClaimTemporalMode,
    BrainContinuityDossierProjection,
    BrainContinuityGraphProjection,
    BrainContinuityQuery,
    BrainCoreMemoryBlockKind,
    BrainCoreMemoryBlockRecord,
    BrainEntityRecord,
    BrainMemoryHealthReportRecord,
    BrainMemoryUseTrace,
    BrainProceduralSkillProjection,
    BrainProceduralSkillRecord,
    BrainProceduralTraceProjection,
    BrainReflectionCycleRecord,
    ClaimLedger,
    ContinuityRetriever,
    CoreMemoryBlockService,
    DiscourseEpisode,
    EntityRegistry,
    MemoryContinuityTrace,
    MemoryHealthService,
    build_continuity_dossier_projection,
    build_continuity_graph_projection,
    build_procedural_trace_projection,
    render_claim_summary,
    stamp_memory_continuity_trace,
    stamp_memory_use_trace,
)
from blink.brain.memory_v2 import (
    build_procedural_skill_projection as derive_procedural_skill_projection,
)
from blink.brain.memory_v2.governance import (
    build_claim_governance_projection as derive_claim_governance_projection,
)
from blink.brain.memory_v2.governance import (
    seed_claim_governance,
)
from blink.brain.memory_v2.multimodal_autobiography import (
    BrainMultimodalAutobiographyModality,
    BrainMultimodalAutobiographyPrivacyClass,
    distill_scene_episode,
    parse_multimodal_autobiography_record,
)
from blink.brain.memory_v2.skill_evidence import (
    BrainSkillEvidenceLedger,
    apply_skill_evidence_update,
)
from blink.brain.memory_v2.skill_promotion import (
    BrainSkillDemotionProposal,
    BrainSkillGovernanceProjection,
    BrainSkillGovernanceStatus,
    BrainSkillPromotionProposal,
    append_skill_demotion_proposal,
    append_skill_promotion_proposal,
)
from blink.brain.persona.compiler import (
    compile_relationship_style_state,
    compile_self_persona_core,
    compile_teaching_profile_state,
)
from blink.brain.persona.policy import PERSONA_RELATIONSHIP_MEMORY_NAMESPACES
from blink.brain.persona.schema import RelationshipStyleStateSpec, TeachingProfileStateSpec
from blink.brain.planning_digest import build_planning_digest
from blink.brain.practice_director import (
    BrainPracticeDirectorProjection,
    BrainPracticePlan,
    append_practice_plan,
    practice_event_types,
)
from blink.brain.presence import BrainPresenceSnapshot, normalize_presence_snapshot
from blink.brain.private_working_memory import build_private_working_memory_projection
from blink.brain.projections import (
    ACTIVE_SITUATION_MODEL_PROJECTION,
    ADAPTER_GOVERNANCE_PROJECTION,
    AGENDA_PROJECTION,
    AUTONOMY_LEDGER_PROJECTION,
    BODY_STATE_PROJECTION,
    CLAIM_GOVERNANCE_PROJECTION,
    COMMITMENT_PROJECTION,
    COUNTERFACTUAL_REHEARSAL_PROJECTION,
    EMBODIED_EXECUTIVE_PROJECTION,
    ENGAGEMENT_STATE_PROJECTION,
    HEARTBEAT_PROJECTION,
    PRACTICE_DIRECTOR_PROJECTION,
    PREDICTIVE_WORLD_MODEL_PROJECTION,
    PRIVATE_WORKING_MEMORY_PROJECTION,
    RELATIONSHIP_STATE_PROJECTION,
    SCENE_STATE_PROJECTION,
    SCENE_WORLD_STATE_PROJECTION,
    SKILL_EVIDENCE_PROJECTION,
    SKILL_GOVERNANCE_PROJECTION,
    WORKING_CONTEXT_PROJECTION,
    BrainActionOutcomeComparisonRecord,
    BrainActionRehearsalRequest,
    BrainActionRehearsalResult,
    BrainActiveSituationProjection,
    BrainAgendaProjection,
    BrainBlockedReason,
    BrainClaimGovernanceProjection,
    BrainCommitmentProjection,
    BrainCommitmentRecord,
    BrainCommitmentScopeType,
    BrainCommitmentStatus,
    BrainCounterfactualCalibrationSummary,
    BrainCounterfactualRehearsalProjection,
    BrainEmbodiedActionEnvelope,
    BrainEmbodiedExecutionTrace,
    BrainEmbodiedExecutiveProjection,
    BrainEmbodiedIntent,
    BrainEmbodiedRecoveryRecord,
    BrainEngagementStateProjection,
    BrainGoal,
    BrainGoalFamily,
    BrainGoalStatus,
    BrainHeartbeatProjection,
    BrainObservedActionOutcomeKind,
    BrainPredictionCalibrationSummary,
    BrainPredictionKind,
    BrainPredictionRecord,
    BrainPredictionResolutionKind,
    BrainPredictiveWorldModelProjection,
    BrainPrivateWorkingMemoryProjection,
    BrainRelationshipStateProjection,
    BrainSceneStateProjection,
    BrainSceneWorldProjection,
    BrainWakeCondition,
    BrainWorkingContextProjection,
)
from blink.brain.scene_world_state import (
    build_scene_world_state_projection as derive_scene_world_state_projection,
)
from blink.brain.world_model import (
    append_prediction_generation,
    append_prediction_resolution,
    build_prediction_records_from_proposals,
    build_prediction_resolution,
    is_predictive_event_type,
    prediction_event_types,
    prediction_generation_event_type,
    resolve_prediction_against_state,
    should_refresh_predictive_world_model,
)
from blink.project_identity import PROJECT_IDENTITY, cache_dir

DEFAULT_BRAIN_DB_PATH = cache_dir("brain", "brain.db")
_SCHEMA_VERSION = 8
_DEFAULT_RELATIONSHIP_AGENT_ID = "blink/main"
_PERSONA_RELATIONSHIP_MEMORY_NAMESPACE_SET = set(PERSONA_RELATIONSHIP_MEMORY_NAMESPACES)
_OVERSIZED_BRAIN_PAYLOAD_MARKER = "brain_event_payload_too_large"
_OVERSIZED_CLAIM_OBJECT_MARKER = "claim_object_too_large"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_ts(value: str | None) -> datetime | None:
    """Parse one optional ISO timestamp into UTC."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _optional_text(value: Any) -> str | None:
    """Normalize one optional stored text value."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stable_commitment_id(
    *,
    scope_type: str,
    scope_id: str,
    goal_family: str,
    intent: str,
    title: str,
) -> str:
    """Return a deterministic durable commitment id."""
    normalized = "|".join(
        part.strip()
        for part in (scope_type, scope_id, goal_family, intent, " ".join(title.split()).strip())
    )
    return f"commitment_{uuid5(NAMESPACE_URL, f'blink:commitment:{normalized}').hex}"


def _event_payload_dict(event: BrainEventRecord) -> dict[str, Any]:
    payload = event.payload
    return payload if isinstance(payload, dict) else {}


def _event_presence_scope_key(event: BrainEventRecord) -> str | None:
    payload = _event_payload_dict(event)
    value = str(payload.get("presence_scope_key") or payload.get("scope_key") or "").strip()
    return value or None


def _autobiography_sort_key(record) -> tuple[str, str, str]:
    return (
        str(getattr(record, "updated_at", "") or ""),
        str(getattr(record, "created_at", "") or ""),
        str(getattr(record, "entry_id", "") or ""),
    )


@dataclass(frozen=True)
class BrainFactRecord:
    """One typed semantic memory record."""

    id: int
    user_id: str
    namespace: str
    subject: str
    value_json: str
    rendered_text: str
    confidence: float
    status: str
    source_episode_id: int | None
    supersedes_fact_id: int | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class BrainEpisodeRecord:
    """One persisted turn episode."""

    id: int
    agent_id: str
    user_id: str
    session_id: str
    thread_id: str
    user_text: str
    assistant_text: str
    assistant_summary: str
    tool_calls_json: str
    created_at: str


@dataclass(frozen=True)
class BrainActionEventRecord:
    """One executed embodied action event."""

    id: int
    agent_id: str
    user_id: str
    thread_id: str
    action_id: str
    source: str
    accepted: bool
    preview_only: bool
    summary: str
    metadata_json: str
    created_at: str

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the decoded action-event metadata."""
        return dict(json.loads(self.metadata_json or "{}"))


@dataclass(frozen=True)
class BrainMemoryExportRecord:
    """One persisted operator artifact export."""

    id: int
    user_id: str
    thread_id: str
    export_kind: str
    path: str
    payload_json: str
    metadata_json: str
    generated_at: str

    @property
    def payload(self) -> dict[str, Any]:
        """Return the decoded export payload."""
        return dict(json.loads(self.payload_json or "{}"))

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the decoded export metadata."""
        return dict(json.loads(self.metadata_json or "{}"))

    def as_dict(self) -> dict[str, Any]:
        """Serialize the export record."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "export_kind": self.export_kind,
            "path": self.path,
            "payload": self.payload,
            "metadata": self.metadata,
            "generated_at": self.generated_at,
        }


class BrainStore(BrainCoreStore):
    """Compatibility superset over the provider-free Blink brain core store."""

    SQLITE_BUSY_TIMEOUT_MS = 30_000

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        world_model_adapter: WorldModelAdapter | None = None,
    ):
        """Initialize the store and ensure the schema exists.

        Args:
            path: Optional SQLite path override. Defaults to Blink cache storage.
            world_model_adapter: Optional proposal-only predictive backend.
        """
        self.path = Path(path) if path else DEFAULT_BRAIN_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=self.SQLITE_BUSY_TIMEOUT_MS / 1000,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure_sqlite_connection()
        self._fts_enabled = False
        self._world_model_adapter = world_model_adapter or LocalDeterministicWorldModelAdapter()
        self._initialize()

    def close(self):
        """Close the SQLite connection."""
        self._conn.close()

    def _configure_sqlite_connection(self):
        """Configure SQLite for local multi-surface runtime access."""
        self._conn.execute(f"PRAGMA busy_timeout = {self.SQLITE_BUSY_TIMEOUT_MS}")
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass

    def _utc_now_for_memory_v2(self) -> str:
        """Expose the store clock to continuity helpers."""
        return _utc_now()

    @property
    def world_model_adapter(self) -> WorldModelAdapter:
        """Expose the proposal-only predictive backend used by the store."""
        return self._world_model_adapter

    def _core_blocks(self) -> CoreMemoryBlockService:
        """Return a continuity core-block service bound to this store."""
        return CoreMemoryBlockService(store=self)

    def _entities(self) -> EntityRegistry:
        """Return a continuity entity registry bound to this store."""
        return EntityRegistry(store=self)

    def _claims(self) -> ClaimLedger:
        """Return a continuity claim ledger bound to this store."""
        return ClaimLedger(store=self)

    def _autobiography(self) -> AutobiographyService:
        """Return an autobiographical-entry service bound to this store."""
        return AutobiographyService(store=self)

    def _health(self) -> MemoryHealthService:
        """Return a memory-health service bound to this store."""
        return MemoryHealthService(store=self)

    def _table_columns(self, table_name: str) -> set[str]:
        """Return the existing column names for one SQLite table."""
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        columns: set[str] = set()
        for row in rows:
            if isinstance(row, sqlite3.Row):
                columns.add(str(row["name"]))
            else:
                columns.add(str(row[1]))
        return columns

    def _ensure_table_column(
        self,
        *,
        table_name: str,
        column_name: str,
        column_sql: str,
        cursor: sqlite3.Cursor | None = None,
    ):
        """Lazily add one missing SQLite column."""
        if column_name in self._table_columns(table_name):
            return
        executor = cursor or self._conn
        executor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")

    def _ensure_claim_governance_schema(self, cursor: sqlite3.Cursor):
        """Lazily add explicit governance columns to the claim table."""
        self._ensure_table_column(
            table_name="claims",
            column_name="currentness_status",
            column_sql="TEXT",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="claims",
            column_name="review_state",
            column_sql="TEXT",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="claims",
            column_name="retention_class",
            column_sql="TEXT",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="claims",
            column_name="governance_reason_codes_json",
            column_sql="TEXT",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="claims",
            column_name="last_governance_event_id",
            column_sql="TEXT",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="claims",
            column_name="governance_updated_at",
            column_sql="TEXT",
            cursor=cursor,
        )

    def _ensure_autobiography_schema(self, cursor: sqlite3.Cursor):
        """Lazily add explicit governance/privacy columns to autobiographical entries."""
        self._ensure_table_column(
            table_name="autobiographical_entries",
            column_name="modality",
            column_sql="TEXT",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="autobiographical_entries",
            column_name="review_state",
            column_sql="TEXT",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="autobiographical_entries",
            column_name="retention_class",
            column_sql="TEXT",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="autobiographical_entries",
            column_name="privacy_class",
            column_sql="TEXT",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="autobiographical_entries",
            column_name="governance_reason_codes_json",
            column_sql="TEXT NOT NULL DEFAULT '[]'",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="autobiographical_entries",
            column_name="last_governance_event_id",
            column_sql="TEXT",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="autobiographical_entries",
            column_name="source_presence_scope_key",
            column_sql="TEXT",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="autobiographical_entries",
            column_name="source_scene_entity_ids_json",
            column_sql="TEXT NOT NULL DEFAULT '[]'",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="autobiographical_entries",
            column_name="source_scene_affordance_ids_json",
            column_sql="TEXT NOT NULL DEFAULT '[]'",
            cursor=cursor,
        )
        self._ensure_table_column(
            table_name="autobiographical_entries",
            column_name="redacted_at",
            column_sql="TEXT",
            cursor=cursor,
        )

    def _ensure_memory_export_schema(self, cursor: sqlite3.Cursor):
        """Lazily add metadata to persisted memory exports."""
        self._ensure_table_column(
            table_name="memory_exports",
            column_name="metadata_json",
            column_sql="TEXT NOT NULL DEFAULT '{}'",
            cursor=cursor,
        )

    def _initialize(self):
        """Create tables and schema metadata if they do not already exist."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_blocks (
                name TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT,
                language TEXT,
                trust_level TEXT NOT NULL,
                last_seen TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_text TEXT NOT NULL,
                assistant_text TEXT NOT NULL,
                assistant_summary TEXT NOT NULL,
                tool_calls_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                subject TEXT NOT NULL,
                value_json TEXT NOT NULL,
                rendered_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                source_episode_id INTEGER,
                supersedes_fact_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                details_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS executive_commitments (
                commitment_id TEXT PRIMARY KEY,
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                title TEXT NOT NULL,
                goal_family TEXT NOT NULL,
                intent TEXT NOT NULL,
                status TEXT NOT NULL,
                details_json TEXT NOT NULL,
                current_goal_id TEXT,
                blocked_reason_json TEXT NOT NULL,
                wake_conditions_json TEXT NOT NULL,
                plan_revision INTEGER NOT NULL,
                resume_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_executive_commitments_scope_status
            ON executive_commitments (scope_type, scope_id, status, updated_at)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS presence_snapshots (
                scope_key TEXT PRIMARY KEY,
                snapshot_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS action_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                source TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                preview_only INTEGER NOT NULL,
                summary TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS brain_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                ts TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                source TEXT NOT NULL,
                correlation_id TEXT,
                causal_parent_id TEXT,
                confidence REAL NOT NULL,
                payload_json TEXT NOT NULL,
                tags_json TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS brain_projections (
                projection_name TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                projection_json TEXT NOT NULL,
                source_event_id TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (projection_name, scope_key)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_semantic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                subject TEXT NOT NULL,
                value_json TEXT NOT NULL,
                rendered_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                source_event_id TEXT,
                source_episode_id INTEGER,
                provenance_json TEXT NOT NULL,
                contradiction_key TEXT,
                supersedes_memory_id INTEGER,
                observed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                stale_after_seconds INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_narrative (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                details_json TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_event_id TEXT,
                provenance_json TEXT NOT NULL,
                contradiction_key TEXT,
                supersedes_memory_id INTEGER,
                observed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                stale_after_seconds INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_episodic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                summary TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_event_id TEXT,
                provenance_json TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                stale_after_seconds INTEGER,
                status TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                layer TEXT NOT NULL,
                memory_id INTEGER NOT NULL,
                provider_name TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (layer, memory_id, provider_name)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                export_kind TEXT NOT NULL,
                path TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                generated_at TEXT NOT NULL
            )
            """
        )
        self._ensure_memory_export_schema(cursor)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS core_memory_blocks (
                block_id TEXT PRIMARY KEY,
                block_kind TEXT NOT NULL,
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                content_json TEXT NOT NULL,
                status TEXT NOT NULL,
                source_event_id TEXT,
                supersedes_block_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_core_memory_blocks_current
            ON core_memory_blocks (block_kind, scope_type, scope_id)
            WHERE status = 'current'
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                attributes_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                subject_entity_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_entity_id TEXT,
                object_json TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                source_event_id TEXT,
                scope_type TEXT,
                scope_id TEXT,
                claim_key TEXT,
                stale_after_seconds INTEGER,
                currentness_status TEXT,
                review_state TEXT,
                retention_class TEXT,
                governance_reason_codes_json TEXT,
                last_governance_event_id TEXT,
                governance_updated_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_claims_scope_updated
            ON claims (scope_type, scope_id, updated_at DESC, created_at DESC)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_claims_subject_updated
            ON claims (subject_entity_id, predicate, updated_at DESC, created_at DESC)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_brain_events_user_thread_id
            ON brain_events (user_id, thread_id, id DESC)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_brain_events_user_thread_type_id
            ON brain_events (user_id, thread_id, event_type, id DESC)
            """
        )
        self._ensure_claim_governance_schema(cursor)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_evidence (
                evidence_id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL,
                source_event_id TEXT,
                source_episode_id INTEGER,
                evidence_summary TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_supersessions (
                supersession_id TEXT PRIMARY KEY,
                prior_claim_id TEXT NOT NULL,
                new_claim_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                source_event_id TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS autobiographical_entries (
                entry_id TEXT PRIMARY KEY,
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                entry_kind TEXT NOT NULL,
                rendered_summary TEXT NOT NULL,
                content_json TEXT NOT NULL,
                status TEXT NOT NULL,
                salience REAL NOT NULL,
                source_episode_ids_json TEXT NOT NULL,
                source_claim_ids_json TEXT NOT NULL,
                source_event_ids_json TEXT NOT NULL,
                supersedes_entry_id TEXT,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._ensure_autobiography_schema(cursor)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_health_reports (
                report_id TEXT PRIMARY KEY,
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                score REAL NOT NULL,
                status TEXT NOT NULL,
                findings_json TEXT NOT NULL,
                stats_json TEXT NOT NULL,
                artifact_path TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reflection_cycles (
                cycle_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                trigger TEXT NOT NULL,
                status TEXT NOT NULL,
                input_episode_cursor INTEGER NOT NULL,
                input_event_cursor INTEGER NOT NULL,
                terminal_episode_cursor INTEGER NOT NULL,
                terminal_event_cursor INTEGER NOT NULL,
                draft_artifact_path TEXT,
                result_stats_json TEXT NOT NULL,
                skip_reason TEXT,
                error_json TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reflection_cycles_scope_status
            ON reflection_cycles (user_id, thread_id, status, started_at DESC)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS procedural_skills (
                skill_id TEXT PRIMARY KEY,
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                skill_family_key TEXT NOT NULL,
                template_fingerprint TEXT NOT NULL,
                title TEXT NOT NULL,
                purpose TEXT NOT NULL,
                goal_family TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                activation_conditions_json TEXT NOT NULL,
                invariants_json TEXT NOT NULL,
                step_template_json TEXT NOT NULL,
                required_capability_ids_json TEXT NOT NULL,
                effects_json TEXT NOT NULL,
                termination_conditions_json TEXT NOT NULL,
                failure_signatures_json TEXT NOT NULL,
                stats_json TEXT NOT NULL,
                supporting_trace_ids_json TEXT NOT NULL,
                supporting_outcome_ids_json TEXT NOT NULL,
                supporting_plan_proposal_ids_json TEXT NOT NULL,
                supporting_commitment_ids_json TEXT NOT NULL,
                supersedes_skill_id TEXT,
                superseded_by_skill_id TEXT,
                retirement_reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                retired_at TEXT,
                details_json TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_procedural_skills_scope_status
            ON procedural_skills (scope_type, scope_id, status, updated_at DESC)
            """
        )
        try:
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_semantic_fts USING fts5(
                    memory_id UNINDEXED,
                    user_id UNINDEXED,
                    rendered_text,
                    namespace,
                    subject
                )
                """
            )
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_narrative_fts USING fts5(
                    memory_id UNINDEXED,
                    user_id UNINDEXED,
                    thread_id UNINDEXED,
                    title,
                    summary,
                    kind
                )
                """
            )
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_claims_fts USING fts5(
                    claim_id UNINDEXED,
                    status UNINDEXED,
                    scope_type UNINDEXED,
                    scope_id UNINDEXED,
                    rendered_text,
                    predicate,
                    subject_name,
                    object_text
                )
                """
            )
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_episodic_fts USING fts5(
                    memory_id UNINDEXED,
                    user_id UNINDEXED,
                    thread_id UNINDEXED,
                    summary,
                    kind
                )
                """
            )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False
        cursor.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES ('schema_version', ?)
            """,
            (str(_SCHEMA_VERSION),),
        )
        cursor.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES ('memory:fts_enabled', ?)
            """,
            ("1" if self._fts_enabled else "0",),
        )
        self._quarantine_oversized_brain_payloads(cursor)
        self._conn.commit()

    def _quarantine_oversized_brain_payloads(self, cursor: sqlite3.Cursor):
        """Replace pathological historical JSON blobs with bounded repair markers."""
        now = _utc_now()
        event_rows = cursor.execute(
            """
            SELECT id, event_type, source, length(payload_json) AS payload_chars
            FROM brain_events
            WHERE length(payload_json) > ?
            """,
            (MAX_BRAIN_EVENT_PAYLOAD_JSON_CHARS,),
        ).fetchall()
        for row in event_rows:
            marker_payload = {
                "_payload_status": "quarantined_oversize",
                "event_type": str(row["event_type"]),
                "payload_chars": int(row["payload_chars"] or 0),
                "reason_codes": [_OVERSIZED_BRAIN_PAYLOAD_MARKER],
                "repaired_at": now,
                "source": str(row["source"]),
            }
            cursor.execute(
                "UPDATE brain_events SET payload_json = ? WHERE id = ?",
                (
                    json.dumps(marker_payload, ensure_ascii=False, sort_keys=True),
                    int(row["id"]),
                ),
            )

        claim_rows = cursor.execute(
            """
            SELECT claim_id, predicate, status, length(object_json) AS object_chars,
                   governance_reason_codes_json
            FROM claims
            WHERE length(object_json) > ?
            """,
            (MAX_CLAIM_OBJECT_JSON_CHARS,),
        ).fetchall()
        for row in claim_rows:
            existing_reason_codes: list[str]
            try:
                existing_reason_codes = list(
                    json.loads(str(row["governance_reason_codes_json"] or "[]"))
                )
            except (TypeError, ValueError):
                existing_reason_codes = []
            reason_codes = sorted(
                {
                    *[str(code) for code in existing_reason_codes if str(code).strip()],
                    "low_support",
                    "requires_confirmation",
                }
            )
            marker_object = {
                "_object_status": "quarantined_oversize",
                "object_chars": int(row["object_chars"] or 0),
                "predicate": str(row["predicate"]),
                "reason_codes": [_OVERSIZED_CLAIM_OBJECT_MARKER],
                "repaired_at": now,
            }
            cursor.execute(
                """
                UPDATE claims
                SET object_json = ?, status = 'revoked', currentness_status = 'historical',
                    review_state = 'resolved', governance_reason_codes_json = ?, updated_at = ?
                WHERE claim_id = ?
                """,
                (
                    json.dumps(marker_object, ensure_ascii=False, sort_keys=True),
                    json.dumps(reason_codes, ensure_ascii=False, sort_keys=True),
                    now,
                    str(row["claim_id"]),
                ),
            )

        if event_rows or claim_rows:
            cursor.execute(
                """
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES (?, ?)
                """,
                ("brain:oversized_payload_quarantine", now),
            )
            cursor.execute(
                """
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES (?, ?)
                """,
                (
                    "brain:oversized_payload_quarantine_counts",
                    json.dumps(
                        {
                            "brain_events": len(event_rows),
                            "claims": len(claim_rows),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            )

    def ensure_default_blocks(self, blocks: dict[str, str]):
        """Insert default pinned agent blocks on first use."""
        now = _utc_now()
        for name, content in blocks.items():
            self._conn.execute(
                """
                INSERT INTO agent_blocks (name, content, source, updated_at)
                VALUES (?, ?, 'default', ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (name, content, now),
            )
        self._conn.commit()
        self._refresh_self_core_block(
            agent_id=_DEFAULT_RELATIONSHIP_AGENT_ID,
            source_event_id=None,
        )
        self._refresh_self_persona_core_block(
            agent_id=_DEFAULT_RELATIONSHIP_AGENT_ID,
            source_event_id=None,
        )

    def ensure_default_adapter_cards(
        self,
        *,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str = "adapter_governance",
        updated_at: str | None = None,
    ) -> None:
        """Seed the deterministic local baseline adapter cards through explicit events."""
        projection = self.get_adapter_governance_projection(scope_key=thread_id)
        existing = {
            (record.adapter_family, record.backend_id, record.backend_version)
            for record in projection.adapter_cards
        }
        resolved_updated_at = str(updated_at or _utc_now())
        for card in build_default_adapter_cards(updated_at=resolved_updated_at):
            key = (card.adapter_family, card.backend_id, card.backend_version)
            if key in existing:
                continue
            self.append_brain_event(
                event_type=BrainEventType.ADAPTER_CARD_UPSERTED,
                agent_id=agent_id,
                user_id=user_id,
                session_id=session_id,
                thread_id=thread_id,
                source=source,
                payload={
                    "scope_key": thread_id,
                    "adapter_card": card.as_dict(),
                },
                ts=resolved_updated_at,
            )
            existing.add(key)

    def get_agent_blocks(self) -> dict[str, str]:
        """Return the current pinned agent blocks."""
        rows = self._conn.execute("SELECT name, content FROM agent_blocks ORDER BY name").fetchall()
        return {str(row["name"]): str(row["content"]) for row in rows}

    def _build_self_core_content(self) -> dict[str, Any]:
        """Return structured agent identity continuity without embedding raw block prose."""
        agent_blocks = self.get_agent_blocks()
        return {
            "project_identity": {
                "display_name": PROJECT_IDENTITY.display_name,
                "distribution_name": PROJECT_IDENTITY.distribution_name,
                "import_namespace": PROJECT_IDENTITY.import_namespace,
                "cli_prefix": PROJECT_IDENTITY.cli_prefix,
                "env_prefix": PROJECT_IDENTITY.env_prefix,
                "homepage_url": PROJECT_IDENTITY.homepage_url,
                "documentation_url": PROJECT_IDENTITY.documentation_url,
                "source_url": PROJECT_IDENTITY.source_url,
                "issues_url": PROJECT_IDENTITY.issues_url,
                "changelog_url": PROJECT_IDENTITY.changelog_url,
            },
            "pinned_blocks": {
                name: {
                    "fingerprint": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "length": len(content),
                }
                for name, content in agent_blocks.items()
            },
            "pinned_block_names": sorted(agent_blocks),
        }

    def _refresh_self_persona_core_block(
        self,
        *,
        agent_id: str,
        source_event_id: str | None,
        user_id: str | None = None,
        thread_id: str | None = None,
        session_id: str | None = None,
    ) -> BrainCoreMemoryBlockRecord | None:
        """Refresh the durable self-persona core block when defaults are available."""
        try:
            payload = compile_self_persona_core(self.get_agent_blocks()).model_dump(mode="json")
        except (ValidationError, ValueError):
            return None
        event_context = None
        if user_id and thread_id and session_id:
            event_context = self._memory_event_context(
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id,
                session_id=session_id,
                source="memory",
            )
        return self.upsert_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.SELF_PERSONA_CORE.value,
            scope_type="agent",
            scope_id=agent_id,
            content=payload,
            source_event_id=source_event_id,
            event_context=event_context,
        )

    def _refresh_relationship_persona_blocks(
        self,
        *,
        user_id: str,
        thread_id: str,
        agent_id: str | None,
        session_id: str | None,
        source_event_id: str | None,
    ) -> tuple[BrainCoreMemoryBlockRecord | None, BrainCoreMemoryBlockRecord | None]:
        """Refresh relationship persona blocks from defaults plus user-scoped memory."""
        agent_blocks = self.get_agent_blocks()
        try:
            relationship_style = compile_relationship_style_state(
                agent_blocks,
                store=self,
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id,
            )
            teaching_profile = compile_teaching_profile_state(
                agent_blocks,
                store=self,
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id,
            )
        except (ValidationError, ValueError):
            return None, None

        relationship_scope_id = self._relationship_scope_id(agent_id=agent_id, user_id=user_id)
        event_context = None
        if session_id:
            event_context = self._memory_event_context(
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id,
                session_id=session_id,
                source="memory",
            )
        relationship_record = self.upsert_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
            scope_type="relationship",
            scope_id=relationship_scope_id,
            content=relationship_style.model_dump(mode="json"),
            source_event_id=source_event_id,
            event_context=event_context,
        )
        teaching_record = self.upsert_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
            scope_type="relationship",
            scope_id=relationship_scope_id,
            content=teaching_profile.model_dump(mode="json"),
            source_event_id=source_event_id,
            event_context=event_context,
        )
        return relationship_record, teaching_record

    def _refresh_self_core_block(
        self,
        *,
        agent_id: str,
        source_event_id: str | None,
        user_id: str | None = None,
        thread_id: str | None = None,
        session_id: str | None = None,
    ):
        """Refresh the structured agent-scoped self-core continuity block."""
        event_context = None
        if user_id and thread_id and session_id:
            event_context = self._memory_event_context(
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id,
                session_id=session_id,
                source="memory",
            )
        self.upsert_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.SELF_CORE.value,
            scope_type="agent",
            scope_id=agent_id,
            content=self._build_self_core_content(),
            source_event_id=source_event_id,
            event_context=event_context,
        )

    def get_metadata(self, key: str) -> str | None:
        """Return one metadata value if present."""
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return str(row["value"]) if row is not None else None

    def set_metadata(self, key: str, value: str):
        """Upsert one metadata value."""
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def ensure_user(
        self,
        *,
        user_id: str,
        language: str | None = None,
        display_name: str | None = None,
        trust_level: str = "local-default",
    ):
        """Create or refresh the current user."""
        now = _utc_now()
        existing = self._conn.execute(
            "SELECT display_name, language, trust_level FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        self._conn.execute(
            """
            INSERT INTO users (user_id, display_name, language, trust_level, last_seen)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                display_name = COALESCE(excluded.display_name, users.display_name),
                language = COALESCE(excluded.language, users.language),
                trust_level = COALESCE(excluded.trust_level, users.trust_level),
                last_seen = excluded.last_seen
            """,
            (
                user_id,
                display_name
                if display_name is not None
                else (existing["display_name"] if existing else None),
                language if language is not None else (existing["language"] if existing else None),
                trust_level
                if trust_level is not None
                else (existing["trust_level"] if existing else "local-default"),
                now,
            ),
        )
        self._conn.commit()

    def add_episode(
        self,
        *,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        user_text: str,
        assistant_text: str,
        assistant_summary: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> int:
        """Append one episode and return its id."""
        cursor = self._conn.execute(
            """
            INSERT INTO episodes (
                agent_id, user_id, session_id, thread_id, user_text,
                assistant_text, assistant_summary, tool_calls_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                user_id,
                session_id,
                thread_id,
                user_text,
                assistant_text,
                assistant_summary,
                json.dumps(tool_calls or [], ensure_ascii=False),
                _utc_now(),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def recent_episodes(
        self,
        *,
        user_id: str,
        thread_id: str,
        limit: int = 4,
    ) -> list[BrainEpisodeRecord]:
        """Return recent episodes for one user thread."""
        rows = self._conn.execute(
            """
            SELECT * FROM episodes
            WHERE user_id = ? AND thread_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, thread_id, limit),
        ).fetchall()
        return [
            BrainEpisodeRecord(
                id=int(row["id"]),
                agent_id=str(row["agent_id"]),
                user_id=str(row["user_id"]),
                session_id=str(row["session_id"]),
                thread_id=str(row["thread_id"]),
                user_text=str(row["user_text"]),
                assistant_text=str(row["assistant_text"]),
                assistant_summary=str(row["assistant_summary"]),
                tool_calls_json=str(row["tool_calls_json"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def episodes_since(
        self,
        *,
        user_id: str,
        thread_id: str,
        after_id: int = 0,
        limit: int = 32,
    ) -> list[BrainEpisodeRecord]:
        """Return episodes newer than `after_id` in ascending order."""
        rows = self._conn.execute(
            """
            SELECT * FROM episodes
            WHERE user_id = ? AND thread_id = ? AND id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (user_id, thread_id, after_id, limit),
        ).fetchall()
        return [
            BrainEpisodeRecord(
                id=int(row["id"]),
                agent_id=str(row["agent_id"]),
                user_id=str(row["user_id"]),
                session_id=str(row["session_id"]),
                thread_id=str(row["thread_id"]),
                user_text=str(row["user_text"]),
                assistant_text=str(row["assistant_text"]),
                assistant_summary=str(row["assistant_summary"]),
                tool_calls_json=str(row["tool_calls_json"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def remember_fact(
        self,
        *,
        user_id: str,
        namespace: str,
        subject: str,
        value: dict[str, Any],
        rendered_text: str,
        confidence: float,
        singleton: bool,
        source_event_id: str | None = None,
        source_episode_id: int | None = None,
        provenance: dict[str, Any] | None = None,
        stale_after_seconds: int | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        thread_id: str | None = None,
    ) -> BrainFactRecord:
        """Insert or update one semantic fact via the continuity layer and semantic mirror."""
        semantic_record = self.remember_semantic_memory(
            user_id=user_id,
            namespace=namespace,
            subject=subject,
            value=value,
            rendered_text=rendered_text,
            confidence=confidence,
            singleton=singleton,
            source_event_id=source_event_id,
            source_episode_id=source_episode_id,
            provenance=provenance,
            stale_after_seconds=stale_after_seconds,
        )
        self._remember_continuity_claim_for_fact(
            user_id=user_id,
            namespace=namespace,
            subject=subject,
            value=value,
            confidence=semantic_record.confidence,
            singleton=singleton,
            source_event_id=source_event_id,
            source_episode_id=source_episode_id,
            provenance=provenance,
            stale_after_seconds=stale_after_seconds,
            agent_id=agent_id,
            session_id=session_id,
            thread_id=thread_id,
            rendered_text=rendered_text,
        )

        now = _utc_now()
        normalized_confidence = max(0.0, min(1.0, float(confidence)))
        value_json = json.dumps(value, ensure_ascii=False, sort_keys=True)
        existing_same = self._conn.execute(
            """
            SELECT * FROM facts
            WHERE user_id = ? AND namespace = ? AND subject = ? AND rendered_text = ? AND status = 'active'
            ORDER BY id DESC LIMIT 1
            """,
            (user_id, namespace, subject, rendered_text),
        ).fetchone()
        if existing_same is not None:
            merged_confidence = min(1.0, float(existing_same["confidence"]) + 0.1)
            self._conn.execute(
                """
                UPDATE facts
                SET confidence = ?, value_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (merged_confidence, value_json, now, int(existing_same["id"])),
            )
            self._conn.commit()
            if agent_id is not None and thread_id is not None:
                if namespace in _PERSONA_RELATIONSHIP_MEMORY_NAMESPACE_SET:
                    self._refresh_relationship_persona_blocks(
                        user_id=user_id,
                        thread_id=thread_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        source_event_id=source_event_id,
                    )
            return self.get_fact(int(existing_same["id"]))

        superseded_fact_id: int | None = None
        if singleton:
            previous = self._conn.execute(
                """
                SELECT * FROM facts
                WHERE user_id = ? AND namespace = ? AND subject = ? AND status = 'active'
                ORDER BY id DESC LIMIT 1
                """,
                (user_id, namespace, subject),
            ).fetchone()
            if previous is not None:
                superseded_fact_id = int(previous["id"])
                self._conn.execute(
                    """
                    UPDATE facts
                    SET status = 'superseded', updated_at = ?
                    WHERE id = ?
                    """,
                    (now, superseded_fact_id),
                )
        else:
            conflicting_namespace = self._conflicting_namespace(namespace)
            if conflicting_namespace is not None:
                conflicting = self._conn.execute(
                    """
                    SELECT * FROM facts
                    WHERE user_id = ? AND namespace = ? AND subject = ? AND status = 'active'
                    ORDER BY id DESC
                    """,
                    (user_id, conflicting_namespace, subject),
                ).fetchall()
                for row in conflicting:
                    self._conn.execute(
                        """
                        UPDATE facts
                        SET status = 'superseded', updated_at = ?
                        WHERE id = ?
                        """,
                        (now, int(row["id"])),
                    )

        cursor = self._conn.execute(
            """
            INSERT INTO facts (
                user_id, namespace, subject, value_json, rendered_text,
                confidence, status, source_episode_id, supersedes_fact_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
            """,
            (
                user_id,
                namespace,
                subject,
                value_json,
                rendered_text,
                normalized_confidence,
                source_episode_id,
                superseded_fact_id,
                now,
                now,
            ),
        )
        self._conn.commit()
        if agent_id is not None and thread_id is not None:
            if namespace in _PERSONA_RELATIONSHIP_MEMORY_NAMESPACE_SET:
                self._refresh_relationship_persona_blocks(
                    user_id=user_id,
                    thread_id=thread_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    source_event_id=source_event_id,
                )
        return BrainFactRecord(
            id=int(cursor.lastrowid),
            user_id=user_id,
            namespace=namespace,
            subject=subject,
            value_json=value_json,
            rendered_text=rendered_text,
            confidence=semantic_record.confidence,
            status="active",
            source_episode_id=source_episode_id,
            supersedes_fact_id=superseded_fact_id,
            created_at=now,
            updated_at=now,
        )

    def _conflicting_namespace(self, namespace: str) -> str | None:
        """Return the opposite namespace for one mutually exclusive fact type."""
        if namespace == "preference.like":
            return "preference.dislike"
        if namespace == "preference.dislike":
            return "preference.like"
        return None

    def _relationship_scope_id(self, *, agent_id: str | None, user_id: str) -> str:
        """Return the canonical relationship-scope id for one agent/user pair."""
        return f"{agent_id or _DEFAULT_RELATIONSHIP_AGENT_ID}:{user_id}"

    def _agent_scope_id(self, *, agent_id: str | None) -> str:
        """Return the canonical agent scope id."""
        return agent_id or _DEFAULT_RELATIONSHIP_AGENT_ID

    def _thread_scope_id(self, *, thread_id: str) -> str:
        """Return the canonical thread scope id."""
        return thread_id

    def _session_commitment_scopes(
        self,
        *,
        agent_id: str | None,
        user_id: str,
        thread_id: str,
    ) -> tuple[tuple[str, str], ...]:
        """Return the canonical durable commitment scopes for one live session."""
        return (
            (
                BrainCommitmentScopeType.RELATIONSHIP.value,
                self._relationship_scope_id(agent_id=agent_id, user_id=user_id),
            ),
            (
                BrainCommitmentScopeType.THREAD.value,
                self._thread_scope_id(thread_id=thread_id),
            ),
            (
                BrainCommitmentScopeType.AGENT.value,
                self._agent_scope_id(agent_id=agent_id),
            ),
        )

    @staticmethod
    def _commitment_family_priority(goal_family: str) -> int:
        """Return the family-aware sort priority for durable commitments."""
        return {
            BrainGoalFamily.CONVERSATION.value: 0,
            BrainGoalFamily.ENVIRONMENT.value: 1,
            BrainGoalFamily.MEMORY_MAINTENANCE.value: 2,
        }.get(goal_family, 99)

    def _build_commitment_projection(
        self,
        *,
        records: list[BrainCommitmentRecord],
        updated_at: str | None = None,
    ) -> BrainCommitmentProjection:
        """Build one durable commitment projection from a record set."""
        ordered = sorted(
            records,
            key=lambda record: (
                self._commitment_family_priority(record.goal_family),
                record.updated_at,
                record.created_at,
                record.title,
            ),
        )
        active = [
            record for record in ordered if record.status == BrainCommitmentStatus.ACTIVE.value
        ]
        deferred = [
            record for record in ordered if record.status == BrainCommitmentStatus.DEFERRED.value
        ]
        blocked = [
            record for record in ordered if record.status == BrainCommitmentStatus.BLOCKED.value
        ]
        terminal = [
            record
            for record in sorted(
                records,
                key=lambda record: (record.updated_at, record.created_at, record.title),
                reverse=True,
            )
            if record.status
            in {
                BrainCommitmentStatus.COMPLETED.value,
                BrainCommitmentStatus.CANCELLED.value,
            }
        ][:6]
        current_active = active[0] if active else None
        return BrainCommitmentProjection(
            active_commitments=active,
            deferred_commitments=deferred,
            blocked_commitments=blocked,
            recent_terminal_commitments=terminal,
            current_active_summary=(
                f"{current_active.goal_family}: {current_active.title}" if current_active else None
            ),
            updated_at=updated_at or _utc_now(),
        )

    def _memory_event_context(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        agent_id: str | None,
        session_id: str | None,
        source: str,
        correlation_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return append-event context if the required session ids are available."""
        if not thread_id or not session_id:
            return None
        return {
            "agent_id": agent_id or _DEFAULT_RELATIONSHIP_AGENT_ID,
            "user_id": user_id,
            "session_id": session_id,
            "thread_id": thread_id,
            "source": source,
            "correlation_id": correlation_id,
        }

    def _commitment_from_row(self, row) -> BrainCommitmentRecord | None:
        """Hydrate one durable commitment row."""
        if row is None:
            return None
        return BrainCommitmentRecord(
            commitment_id=str(row["commitment_id"]),
            scope_type=str(row["scope_type"]),
            scope_id=str(row["scope_id"]),
            title=str(row["title"]),
            goal_family=str(row["goal_family"]),
            intent=str(row["intent"]),
            status=str(row["status"]),
            details=json.loads(str(row["details_json"])),
            current_goal_id=str(row["current_goal_id"])
            if row["current_goal_id"] is not None
            else None,
            blocked_reason=BrainBlockedReason.from_dict(
                json.loads(str(row["blocked_reason_json"]))
            ),
            wake_conditions=[
                item
                for item in (
                    BrainWakeCondition.from_dict(entry)
                    for entry in json.loads(str(row["wake_conditions_json"]))
                )
                if item is not None
            ],
            plan_revision=int(row["plan_revision"]),
            resume_count=int(row["resume_count"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            completed_at=str(row["completed_at"]) if row["completed_at"] is not None else None,
        )

    def get_executive_commitment(self, *, commitment_id: str) -> BrainCommitmentRecord | None:
        """Return one durable executive commitment by id."""
        row = self._conn.execute(
            """
            SELECT * FROM executive_commitments
            WHERE commitment_id = ?
            """,
            (commitment_id,),
        ).fetchone()
        return self._commitment_from_row(row)

    def list_executive_commitments(
        self,
        *,
        scope_type: str | None = None,
        scope_id: str | None = None,
        user_id: str | None = None,
        statuses: tuple[str, ...] | None = None,
        limit: int = 24,
    ) -> list[BrainCommitmentRecord]:
        """Return durable executive commitments for one optional scope filter."""
        clauses: list[str] = []
        params: list[Any] = []
        if scope_type is not None:
            clauses.append("scope_type = ?")
            params.append(scope_type)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(statuses)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT * FROM executive_commitments
            {where}
            ORDER BY updated_at ASC, created_at ASC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [
            record
            for record in (self._commitment_from_row(row) for row in rows)
            if record is not None
        ]

    def list_session_commitments(
        self,
        *,
        agent_id: str | None,
        user_id: str,
        thread_id: str,
        statuses: tuple[str, ...] | None = None,
        limit: int = 64,
    ) -> list[BrainCommitmentRecord]:
        """Return durable commitments relevant to one active session."""
        scope_pairs = set(
            self._session_commitment_scopes(
                agent_id=agent_id,
                user_id=user_id,
                thread_id=thread_id,
            )
        )
        records = self.list_executive_commitments(
            user_id=user_id,
            statuses=statuses,
            limit=max(limit * 3, 64),
        )
        filtered = [
            record for record in records if (record.scope_type, record.scope_id) in scope_pairs
        ]
        ordered = sorted(
            filtered,
            key=lambda record: (
                self._commitment_family_priority(record.goal_family),
                record.updated_at,
                record.created_at,
                record.title,
            ),
        )
        return ordered[:limit]

    def find_executive_commitment(
        self,
        *,
        scope_type: str,
        scope_id: str,
        goal_family: str,
        intent: str,
        title: str,
    ) -> BrainCommitmentRecord | None:
        """Return a stable commitment by its deterministic identity tuple."""
        return self.get_executive_commitment(
            commitment_id=_stable_commitment_id(
                scope_type=scope_type,
                scope_id=scope_id,
                goal_family=goal_family,
                intent=intent,
                title=title,
            )
        )

    def upsert_executive_commitment(
        self,
        *,
        scope_type: str,
        scope_id: str,
        user_id: str,
        thread_id: str,
        title: str,
        goal_family: str,
        intent: str,
        status: str,
        details: dict[str, Any] | None = None,
        current_goal_id: str | None = None,
        blocked_reason: BrainBlockedReason | None = None,
        wake_conditions: list[BrainWakeCondition] | None = None,
        plan_revision: int | None = None,
        resume_count: int | None = None,
        commitment_id: str | None = None,
        source_event_id: str | None = None,
    ) -> BrainCommitmentRecord:
        """Insert or update one durable executive commitment row."""
        normalized_title = " ".join((title or "").split()).strip()
        if not normalized_title:
            raise ValueError("Commitment title must not be empty")
        resolved_commitment_id = commitment_id or _stable_commitment_id(
            scope_type=scope_type,
            scope_id=scope_id,
            goal_family=goal_family,
            intent=intent,
            title=normalized_title,
        )
        now = _utc_now()
        existing = self.get_executive_commitment(commitment_id=resolved_commitment_id)
        if existing is None:
            record = BrainCommitmentRecord(
                commitment_id=resolved_commitment_id,
                scope_type=scope_type,
                scope_id=scope_id,
                title=normalized_title,
                goal_family=goal_family,
                intent=intent,
                status=status,
                details=dict(details or {}),
                current_goal_id=current_goal_id,
                blocked_reason=blocked_reason,
                wake_conditions=list(wake_conditions or []),
                plan_revision=plan_revision or 1,
                resume_count=resume_count or 0,
                created_at=now,
                updated_at=now,
                completed_at=now
                if status
                in {
                    BrainCommitmentStatus.COMPLETED.value,
                    BrainCommitmentStatus.CANCELLED.value,
                }
                else None,
            )
            self._conn.execute(
                """
                INSERT INTO executive_commitments (
                    commitment_id, scope_type, scope_id, user_id, thread_id, title,
                    goal_family, intent, status, details_json, current_goal_id,
                    blocked_reason_json, wake_conditions_json, plan_revision, resume_count,
                    created_at, updated_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.commitment_id,
                    record.scope_type,
                    record.scope_id,
                    user_id,
                    thread_id,
                    record.title,
                    record.goal_family,
                    record.intent,
                    record.status,
                    json.dumps(record.details, ensure_ascii=False, sort_keys=True),
                    record.current_goal_id,
                    json.dumps(
                        record.blocked_reason.as_dict() if record.blocked_reason else {},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    json.dumps(
                        [item.as_dict() for item in record.wake_conditions],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    record.plan_revision,
                    record.resume_count,
                    record.created_at,
                    record.updated_at,
                    record.completed_at,
                ),
            )
        else:
            record = BrainCommitmentRecord(
                commitment_id=existing.commitment_id,
                scope_type=existing.scope_type,
                scope_id=existing.scope_id,
                title=normalized_title,
                goal_family=goal_family or existing.goal_family,
                intent=intent or existing.intent,
                status=status,
                details=dict(details if details is not None else existing.details),
                current_goal_id=current_goal_id
                if current_goal_id is not None
                else existing.current_goal_id,
                blocked_reason=blocked_reason,
                wake_conditions=list(
                    wake_conditions if wake_conditions is not None else existing.wake_conditions
                ),
                plan_revision=plan_revision
                if plan_revision is not None
                else existing.plan_revision,
                resume_count=resume_count if resume_count is not None else existing.resume_count,
                created_at=existing.created_at,
                updated_at=now,
                completed_at=(
                    now
                    if status
                    in {
                        BrainCommitmentStatus.COMPLETED.value,
                        BrainCommitmentStatus.CANCELLED.value,
                    }
                    else None
                ),
            )
            self._conn.execute(
                """
                UPDATE executive_commitments
                SET user_id = ?, thread_id = ?, title = ?, goal_family = ?, intent = ?, status = ?,
                    details_json = ?, current_goal_id = ?, blocked_reason_json = ?,
                    wake_conditions_json = ?, plan_revision = ?, resume_count = ?,
                    updated_at = ?, completed_at = ?
                WHERE commitment_id = ?
                """,
                (
                    user_id,
                    thread_id,
                    record.title,
                    record.goal_family,
                    record.intent,
                    record.status,
                    json.dumps(record.details, ensure_ascii=False, sort_keys=True),
                    record.current_goal_id,
                    json.dumps(
                        record.blocked_reason.as_dict() if record.blocked_reason else {},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    json.dumps(
                        [item.as_dict() for item in record.wake_conditions],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    record.plan_revision,
                    record.resume_count,
                    record.updated_at,
                    record.completed_at,
                    record.commitment_id,
                ),
            )
        self._conn.commit()
        self._refresh_commitment_state_surfaces(
            scope_type=scope_type,
            scope_id=scope_id,
            user_id=user_id,
            thread_id=thread_id,
            source_event_id=source_event_id,
            updated_at=record.updated_at,
        )
        refreshed = self.get_executive_commitment(commitment_id=resolved_commitment_id)
        if refreshed is None:
            raise RuntimeError("Failed to persist executive commitment.")
        return refreshed

    def _refresh_commitment_state_surfaces(
        self,
        *,
        scope_type: str,
        scope_id: str,
        user_id: str,
        thread_id: str,
        source_event_id: str | None,
        updated_at: str,
    ):
        """Refresh durable commitment projections and compatibility mirrors."""
        records = self.list_executive_commitments(
            scope_type=scope_type,
            scope_id=scope_id,
            limit=64,
        )
        projection = self._build_commitment_projection(records=records, updated_at=updated_at)
        self._upsert_projection(
            projection_name=COMMITMENT_PROJECTION,
            scope_key=scope_id,
            projection=projection.as_dict(),
            source_event_id=source_event_id,
            updated_at=updated_at,
            commit=False,
        )
        if scope_type == "relationship":
            self._refresh_active_commitments_block(
                user_id=user_id,
                agent_id=scope_id.split(":", 1)[0],
                session_id=None,
                thread_id=thread_id,
                source_event_id=source_event_id,
            )

    def get_commitment_projection(self, *, scope_key: str) -> BrainCommitmentProjection:
        """Return the durable commitment projection for one persisted scope id."""
        projection = self._get_projection_dict(
            projection_name=COMMITMENT_PROJECTION,
            scope_key=scope_key,
        )
        if projection is not None:
            return BrainCommitmentProjection.from_dict(projection)
        records = self.list_executive_commitments(
            scope_type="relationship",
            scope_id=scope_key,
            limit=64,
        )
        return self._build_commitment_projection(records=records)

    def get_session_commitment_projection(
        self,
        *,
        agent_id: str | None,
        user_id: str,
        thread_id: str,
    ) -> BrainCommitmentProjection:
        """Return the aggregate durable commitment projection for one session."""
        return self._build_commitment_projection(
            records=self.list_session_commitments(
                agent_id=agent_id,
                user_id=user_id,
                thread_id=thread_id,
                limit=64,
            )
        )

    def _remember_continuity_claim_for_fact(
        self,
        *,
        user_id: str,
        namespace: str,
        subject: str,
        value: dict[str, Any],
        confidence: float,
        singleton: bool,
        source_event_id: str | None,
        source_episode_id: int | None,
        provenance: dict[str, Any] | None,
        stale_after_seconds: int | None,
        agent_id: str | None,
        session_id: str | None,
        thread_id: str | None,
        rendered_text: str,
    ):
        """Mirror a semantic fact into the continuity entity/claim ledger."""
        registry = self._entities()
        ledger = self._claims()
        user_entity = registry.ensure_entity(
            entity_type="user",
            canonical_name=user_id,
            aliases=[user_id],
            attributes={"user_id": user_id},
        )
        object_value = str(value.get("value", "")).strip()
        object_entity_id: str | None = None
        claim_key = f"{namespace}:{user_id}" if singleton else None
        if namespace.startswith("preference.") and object_value:
            topic_entity = registry.ensure_entity(
                entity_type="topic",
                canonical_name=object_value,
                aliases=[subject],
                attributes={"normalized_subject": subject},
            )
            object_entity_id = topic_entity.entity_id
            claim_key = f"preference:{topic_entity.canonical_name.lower()}"
        event_context = self._memory_event_context(
            user_id=user_id,
            thread_id=thread_id,
            agent_id=agent_id,
            session_id=session_id,
            source=str((provenance or {}).get("source", "memory")),
            correlation_id=str((provenance or {}).get("tool_name", "")) or None,
        )

        existing_current = [
            claim
            for claim in ledger.query_claims(
                temporal_mode="current",
                subject_entity_id=user_entity.entity_id,
                predicate=namespace,
                scope_type="user",
                scope_id=user_id,
                limit=12,
            )
            if (
                (object_entity_id is not None and claim.object_entity_id == object_entity_id)
                or str(claim.object.get("value", "")).strip() == object_value
                or claim.object == value
            )
        ]
        if existing_current:
            if namespace in {"profile.name", "profile.role", "profile.origin"}:
                self._refresh_user_core_block(
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    thread_id=thread_id,
                    source_event_id=source_event_id,
                )
            return

        replacement_payload = {
            "subject_entity_id": user_entity.entity_id,
            "predicate": namespace,
            "object_entity_id": object_entity_id,
            "object_value": object_value,
            "object_data": value,
            "status": "active",
            "confidence": confidence,
            "scope_type": "user",
            "scope_id": user_id,
            "claim_key": claim_key,
            "stale_after_seconds": stale_after_seconds,
            "evidence_summary": rendered_text,
            "evidence_json": {
                "provenance": provenance or {},
                "source": (provenance or {}).get("source", "memory"),
            },
            "source_episode_id": source_episode_id,
        }
        prior_claim: BrainClaimRecord | None = None
        current_claims = ledger.query_claims(
            temporal_mode="current",
            subject_entity_id=user_entity.entity_id,
            scope_type="user",
            scope_id=user_id,
            limit=24,
        )
        if singleton:
            for claim in current_claims:
                if claim.predicate == namespace:
                    prior_claim = claim
                    break
        else:
            conflicting_namespace = self._conflicting_namespace(namespace)
            for claim in current_claims:
                if claim.predicate != conflicting_namespace:
                    continue
                same_topic = (
                    object_entity_id is not None and claim.object_entity_id == object_entity_id
                ) or str(claim.object.get("value", "")).strip() == object_value
                if same_topic:
                    prior_claim = claim
                    break

        if prior_claim is not None:
            ledger.supersede_claim(
                prior_claim.claim_id,
                replacement_payload,
                reason="explicit_correction"
                if (provenance or {}).get("source") == "tool"
                else "continuity_update",
                source_event_id=source_event_id,
                event_context=event_context,
            )
        else:
            ledger.record_claim(
                source_event_id=source_event_id,
                event_context=event_context,
                **replacement_payload,
            )

        if namespace in {"profile.name", "profile.role", "profile.origin"}:
            self._refresh_user_core_block(
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                thread_id=thread_id,
                source_event_id=source_event_id,
            )

    def _refresh_user_core_block(
        self,
        *,
        user_id: str,
        agent_id: str | None,
        session_id: str | None,
        thread_id: str | None,
        source_event_id: str | None,
    ):
        """Refresh the user-core block from current profile claims."""
        registry = self._entities()
        user_entity = registry.ensure_entity(
            entity_type="user",
            canonical_name=user_id,
            aliases=[user_id],
            attributes={"user_id": user_id},
        )
        block_content: dict[str, Any] = {}
        for claim in self.query_claims(
            temporal_mode="current",
            subject_entity_id=user_entity.entity_id,
            scope_type="user",
            scope_id=user_id,
            limit=12,
        ):
            if claim.predicate == "profile.name":
                block_content["name"] = claim.object.get("value")
            elif claim.predicate == "profile.role":
                block_content["role"] = claim.object.get("value")
            elif claim.predicate == "profile.origin":
                block_content["origin"] = claim.object.get("value")
        self.upsert_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.USER_CORE.value,
            scope_type="user",
            scope_id=user_id,
            content=block_content,
            source_event_id=source_event_id,
            event_context=self._memory_event_context(
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id,
                session_id=session_id,
                source="memory",
            ),
        )

    def _refresh_active_commitments_block(
        self,
        *,
        user_id: str,
        agent_id: str | None,
        session_id: str | None,
        thread_id: str | None,
        source_event_id: str | None,
    ):
        """Refresh the active-commitments block from durable executive commitments."""
        scope_id = self._relationship_scope_id(agent_id=agent_id, user_id=user_id)
        commitment_rows = self.list_executive_commitments(
            scope_type="relationship",
            scope_id=scope_id,
            statuses=(
                BrainCommitmentStatus.ACTIVE.value,
                BrainCommitmentStatus.DEFERRED.value,
                BrainCommitmentStatus.BLOCKED.value,
            ),
            limit=32,
        )
        if not commitment_rows:
            narrative_rows = self.narrative_memories(
                user_id=user_id,
                thread_id=thread_id,
                kinds=("commitment",),
                statuses=("open",),
                limit=32,
            )
            commitments = [
                {
                    "title": record.title,
                    "summary": record.summary,
                    "details": record.details,
                    "status": record.status,
                    "created_at": record.observed_at,
                    "updated_at": record.updated_at,
                }
                for record in narrative_rows
            ]
        else:
            commitments = [
                {
                    "commitment_id": record.commitment_id,
                    "title": record.title,
                    "summary": str(record.details.get("summary") or record.title),
                    "details": record.details,
                    "status": record.status,
                    "goal_family": record.goal_family,
                    "current_goal_id": record.current_goal_id,
                    "blocked_reason": (
                        record.blocked_reason.as_dict() if record.blocked_reason else None
                    ),
                    "wake_conditions": [item.as_dict() for item in record.wake_conditions],
                    "plan_revision": record.plan_revision,
                    "resume_count": record.resume_count,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                }
                for record in commitment_rows
            ]
        self.upsert_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.ACTIVE_COMMITMENTS.value,
            scope_type="relationship",
            scope_id=scope_id,
            content={"commitments": commitments},
            source_event_id=source_event_id,
            event_context=self._memory_event_context(
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id,
                session_id=session_id,
                source="memory",
            ),
        )

    def _refresh_relationship_core_block(
        self,
        *,
        user_id: str,
        thread_id: str,
        agent_id: str | None,
        session_id: str | None,
        source_event_id: str | None,
        summary: str | None = None,
    ):
        """Refresh the relationship-core block from current summary and commitments."""
        scope_id = self._relationship_scope_id(agent_id=agent_id, user_id=user_id)
        commitments = [task["title"] for task in self.active_tasks(user_id=user_id, limit=8)]
        content = {
            "thread_id": thread_id,
            "last_session_summary": summary or "",
            "open_commitments": commitments,
        }
        self.upsert_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_CORE.value,
            scope_type="relationship",
            scope_id=scope_id,
            content=content,
            source_event_id=source_event_id,
            event_context=self._memory_event_context(
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id,
                session_id=session_id,
                source="memory",
            ),
        )
        self._refresh_relationship_persona_blocks(
            user_id=user_id,
            thread_id=thread_id,
            agent_id=agent_id,
            session_id=session_id,
            source_event_id=source_event_id,
        )

    def get_fact(self, fact_id: int) -> BrainFactRecord:
        """Return one fact record by id."""
        row = self._conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if row is None:
            raise KeyError(f"Missing fact id {fact_id}")
        return BrainFactRecord(
            id=int(row["id"]),
            user_id=str(row["user_id"]),
            namespace=str(row["namespace"]),
            subject=str(row["subject"]),
            value_json=str(row["value_json"]),
            rendered_text=str(row["rendered_text"]),
            confidence=float(row["confidence"]),
            status=str(row["status"]),
            source_episode_id=int(row["source_episode_id"])
            if row["source_episode_id"] is not None
            else None,
            supersedes_fact_id=int(row["supersedes_fact_id"])
            if row["supersedes_fact_id"] is not None
            else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _rendered_text_for_claim(self, claim: BrainClaimRecord) -> str:
        """Return compatibility fact text for one continuity claim."""
        if claim.predicate == "session.summary":
            return str(claim.object.get("summary", "")).strip()
        return render_claim_summary(claim)

    def _subject_for_claim_fact(self, claim: BrainClaimRecord) -> str:
        """Return a compatibility fact subject for one continuity claim."""
        if claim.predicate.startswith("profile."):
            return "user"
        if claim.predicate == "session.summary":
            return str(claim.object.get("thread_id") or claim.scope_id or "user").strip()
        value = str(claim.object.get("value", "")).strip()
        return value.lower() if value else claim.predicate

    def _fact_record_from_claim(
        self,
        *,
        claim: BrainClaimRecord,
        user_id: str,
        fallback_id: int,
    ) -> BrainFactRecord:
        """Project one continuity claim into the legacy fact record shape."""
        evidence = self._claims().claim_evidence(claim.claim_id)
        source_episode_id = next(
            (
                record.source_episode_id
                for record in evidence
                if record.source_episode_id is not None
            ),
            None,
        )
        return BrainFactRecord(
            id=fallback_id,
            user_id=user_id,
            namespace=claim.predicate,
            subject=self._subject_for_claim_fact(claim),
            value_json=claim.object_json,
            rendered_text=self._rendered_text_for_claim(claim),
            confidence=claim.confidence,
            status=claim.status,
            source_episode_id=source_episode_id,
            supersedes_fact_id=None,
            created_at=claim.created_at,
            updated_at=claim.updated_at,
        )

    def _continuity_fact_records(
        self,
        *,
        user_id: str,
        claims: list[BrainClaimRecord],
    ) -> list[BrainFactRecord]:
        """Convert continuity claims into compatibility fact records."""
        return [
            self._fact_record_from_claim(
                claim=claim,
                user_id=user_id,
                fallback_id=index + 1,
            )
            for index, claim in enumerate(claims)
        ]

    def active_facts(self, *, user_id: str, limit: int = 24) -> list[BrainFactRecord]:
        """Return active facts for one user, preferring continuity claims over legacy mirrors."""
        current_claims = [
            claim
            for claim in self.query_claims(
                temporal_mode="current",
                scope_type="user",
                scope_id=user_id,
                limit=limit * 3,
            )
            if not claim.is_stale
        ]
        if current_claims:
            return self._continuity_fact_records(
                user_id=user_id,
                claims=current_claims[:limit],
            )

        semantic_records = self.semantic_memories(user_id=user_id, limit=limit)
        if semantic_records:
            return [
                BrainFactRecord(
                    id=record.id,
                    user_id=record.user_id,
                    namespace=record.namespace,
                    subject=record.subject,
                    value_json=record.value_json,
                    rendered_text=record.rendered_text,
                    confidence=record.confidence,
                    status=record.status,
                    source_episode_id=record.source_episode_id,
                    supersedes_fact_id=record.supersedes_memory_id,
                    created_at=record.observed_at,
                    updated_at=record.updated_at,
                )
                for record in semantic_records[:limit]
            ]

        rows = self._conn.execute(
            """
            SELECT * FROM facts
            WHERE user_id = ? AND status = 'active'
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [self.get_fact(int(row["id"])) for row in rows]

    def relevant_facts(self, *, user_id: str, query: str, limit: int = 6) -> list[BrainFactRecord]:
        """Return the most relevant active facts, preferring continuity retrieval."""
        continuity_hits = [
            result
            for result in ContinuityRetriever(store=self).retrieve(
                BrainContinuityQuery(
                    text=query,
                    scope_type="user",
                    scope_id=user_id,
                    temporal_mode="current",
                    limit=limit,
                )
            )
            if not result.claim.is_stale
        ]
        if continuity_hits:
            return self._continuity_fact_records(
                user_id=user_id,
                claims=[result.claim for result in continuity_hits[:limit]],
            )

        results = self.search_semantic_memories(
            user_id=user_id,
            text=query,
            limit=limit,
            include_stale=False,
        )
        if results:
            records = {
                record.id: record
                for record in self.semantic_memories(
                    user_id=user_id,
                    limit=max(limit * 4, limit),
                    include_inactive=False,
                    include_stale=True,
                )
            }
            return [
                BrainFactRecord(
                    id=record.id,
                    user_id=record.user_id,
                    namespace=record.namespace,
                    subject=record.subject,
                    value_json=record.value_json,
                    rendered_text=record.rendered_text,
                    confidence=record.confidence,
                    status=record.status,
                    source_episode_id=record.source_episode_id,
                    supersedes_fact_id=record.supersedes_memory_id,
                    created_at=record.observed_at,
                    updated_at=record.updated_at,
                )
                for result in results
                if (record := records.get(result.record_id)) is not None
            ]
        return self.active_facts(user_id=user_id, limit=limit)

    def upsert_session_summary(
        self,
        *,
        user_id: str,
        thread_id: str,
        summary: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        source_event_id: str | None = None,
    ):
        """Persist the current thread summary in narrative memory and compatibility facts."""
        self.upsert_narrative_memory(
            user_id=user_id,
            thread_id=thread_id,
            kind="session_summary",
            title=thread_id,
            summary=summary,
            details={"thread_id": thread_id},
            status="active",
            confidence=0.9,
            source_event_id=source_event_id,
            contradiction_key=f"session_summary:{thread_id}",
            stale_after_seconds=semantic_default_staleness("session.summary"),
        )
        self.remember_fact(
            user_id=user_id,
            namespace="session.summary",
            subject=thread_id,
            value={"summary": summary},
            rendered_text=summary,
            confidence=0.9,
            singleton=True,
            source_event_id=source_event_id,
            source_episode_id=None,
            agent_id=agent_id,
            session_id=session_id,
            thread_id=thread_id,
        )
        self._refresh_relationship_core_block(
            user_id=user_id,
            thread_id=thread_id,
            agent_id=agent_id,
            session_id=session_id,
            source_event_id=source_event_id,
            summary=summary,
        )

    def get_session_summary(self, *, user_id: str, thread_id: str) -> str:
        """Return the current stored session summary if present."""
        relationship_scope_id = self._relationship_scope_id(
            agent_id=_DEFAULT_RELATIONSHIP_AGENT_ID,
            user_id=user_id,
        )
        relationship_core = self.get_current_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_CORE.value,
            scope_type="relationship",
            scope_id=relationship_scope_id,
        )
        if relationship_core is not None:
            summary = str(relationship_core.content.get("last_session_summary", "")).strip()
            if summary:
                return summary
        narrative = self.narrative_memories(
            user_id=user_id,
            thread_id=thread_id,
            kinds=("session_summary",),
            statuses=("active",),
            limit=1,
        )
        if narrative:
            return narrative[0].summary
        row = self._conn.execute(
            """
            SELECT rendered_text FROM facts
            WHERE user_id = ? AND namespace = 'session.summary' AND subject = ? AND status = 'active'
            ORDER BY id DESC LIMIT 1
            """,
            (user_id, thread_id),
        ).fetchone()
        return str(row["rendered_text"]) if row is not None else ""

    def get_current_core_memory_block(
        self,
        *,
        block_kind: str,
        scope_type: str,
        scope_id: str,
    ) -> BrainCoreMemoryBlockRecord | None:
        """Return the current continuity block for one scope."""
        return self._core_blocks().get_current_block(block_kind, scope_type, scope_id)

    def list_core_memory_block_versions(
        self,
        *,
        block_kind: str,
        scope_type: str,
        scope_id: str,
    ) -> list[BrainCoreMemoryBlockRecord]:
        """Return all recorded versions for one continuity block."""
        return self._core_blocks().list_block_versions(block_kind, scope_type, scope_id)

    def list_current_core_memory_blocks(
        self,
        *,
        scope_type: str | None = None,
        scope_id: str | None = None,
        block_kinds: tuple[str, ...] | None = None,
        limit: int = 16,
    ) -> list[BrainCoreMemoryBlockRecord]:
        """Return current continuity blocks for one optional scope filter."""
        return self._core_blocks().list_current_blocks(
            scope_type=scope_type,
            scope_id=scope_id,
            block_kinds=block_kinds,
            limit=limit,
        )

    def upsert_core_memory_block(
        self,
        *,
        block_kind: str,
        scope_type: str,
        scope_id: str,
        content: dict[str, Any],
        source_event_id: str | None = None,
        expected_version: int | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainCoreMemoryBlockRecord:
        """Append a new continuity block version and supersede the prior current version."""
        return self._core_blocks().upsert_block(
            block_kind=block_kind,
            scope_type=scope_type,
            scope_id=scope_id,
            content=content,
            source_event_id=source_event_id,
            expected_version=expected_version,
            event_context=event_context,
        )

    def ensure_entity(
        self,
        *,
        entity_type: str,
        canonical_name: str,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> BrainEntityRecord:
        """Ensure a continuity entity exists and return it."""
        return self._entities().ensure_entity(
            entity_type=entity_type,
            canonical_name=canonical_name,
            aliases=aliases,
            attributes=attributes,
        )

    def query_claims(
        self,
        *,
        temporal_mode: str = "current",
        subject_entity_id: str | None = None,
        predicate: str | None = None,
        entity_type: str | None = None,
        scope_type: str | None = None,
        scope_id: str | None = None,
        currentness_states: Iterable[str] | None = None,
        review_states: Iterable[str] | None = None,
        retention_classes: Iterable[str] | None = None,
        limit: int | None = 24,
    ) -> list[BrainClaimRecord]:
        """Query continuity claims with temporal filtering."""
        return self._claims().query_claims(
            temporal_mode=temporal_mode,
            subject_entity_id=subject_entity_id,
            predicate=predicate,
            entity_type=entity_type,
            scope_type=scope_type,
            scope_id=scope_id,
            currentness_states=currentness_states,
            review_states=review_states,
            retention_classes=retention_classes,
            limit=limit,
        )

    def build_claim_governance_projection(
        self,
        *,
        scope_type: str,
        scope_id: str,
        source_event_id: str | None = None,
        updated_at: str | None = None,
        commit: bool = True,
    ) -> BrainClaimGovernanceProjection:
        """Build and persist the scoped claim-governance projection."""
        claims = self.query_claims(
            temporal_mode=BrainClaimTemporalMode.ALL.value,
            scope_type=scope_type,
            scope_id=scope_id,
            limit=None,
        )
        projection = derive_claim_governance_projection(
            scope_type=scope_type,
            scope_id=scope_id,
            claims=claims,
            updated_at=updated_at,
        )
        self._upsert_projection(
            projection_name=CLAIM_GOVERNANCE_PROJECTION,
            scope_key=f"{scope_type}:{scope_id}",
            projection=projection.as_dict(),
            source_event_id=source_event_id,
            updated_at=projection.updated_at,
            commit=commit,
        )
        return projection

    def get_claim_governance_projection(
        self,
        *,
        scope_type: str,
        scope_id: str,
    ) -> BrainClaimGovernanceProjection:
        """Return one scoped claim-governance projection, building it on cache miss."""
        projection = BrainClaimGovernanceProjection.from_dict(
            self._get_projection_dict(
                projection_name=CLAIM_GOVERNANCE_PROJECTION,
                scope_key=f"{scope_type}:{scope_id}",
            )
        )
        if projection.scope_type and projection.scope_id:
            return projection
        return self.build_claim_governance_projection(
            scope_type=scope_type,
            scope_id=scope_id,
            commit=True,
        )

    def preview_claim_governance_projection(
        self,
        *,
        scope_type: str,
        scope_id: str,
        updated_at: str | None = None,
    ) -> BrainClaimGovernanceProjection:
        """Derive one scoped claim-governance projection without persisting it."""
        claims = self.query_claims(
            temporal_mode=BrainClaimTemporalMode.ALL.value,
            scope_type=scope_type,
            scope_id=scope_id,
            limit=None,
        )
        return derive_claim_governance_projection(
            scope_type=scope_type,
            scope_id=scope_id,
            claims=claims,
            updated_at=updated_at,
        )

    def claim_history(
        self,
        *,
        subject_entity_id: str,
        predicate: str | None = None,
        limit: int = 24,
    ) -> list[BrainClaimRecord]:
        """Return current and historical claims for one subject."""
        return self._claims().claim_history(subject_entity_id, predicate, limit=limit)

    def claim_supersessions(
        self,
        *,
        claim_id: str | None = None,
    ) -> list[BrainClaimSupersessionRecord]:
        """Return recorded claim correction links."""
        return self._claims().claim_supersessions(claim_id)

    def _list_continuity_core_blocks(
        self,
        *,
        user_id: str,
        agent_id: str | None,
    ) -> list[BrainCoreMemoryBlockRecord]:
        """Return the current continuity block versions relevant to dossier compilation."""
        resolved_agent_id = agent_id or _DEFAULT_RELATIONSHIP_AGENT_ID
        relationship_scope_id = f"{resolved_agent_id}:{user_id}"
        current_blocks = (
            self.list_current_core_memory_blocks(
                scope_type="user",
                scope_id=user_id,
                block_kinds=(BrainCoreMemoryBlockKind.USER_CORE.value,),
                limit=8,
            )
            + self.list_current_core_memory_blocks(
                scope_type="agent",
                scope_id=resolved_agent_id,
                block_kinds=(
                    BrainCoreMemoryBlockKind.SELF_CORE.value,
                    BrainCoreMemoryBlockKind.SELF_PERSONA_CORE.value,
                    BrainCoreMemoryBlockKind.SELF_CURRENT_ARC.value,
                ),
                limit=8,
            )
            + self.list_current_core_memory_blocks(
                scope_type="relationship",
                scope_id=relationship_scope_id,
                block_kinds=(
                    BrainCoreMemoryBlockKind.RELATIONSHIP_CORE.value,
                    BrainCoreMemoryBlockKind.ACTIVE_COMMITMENTS.value,
                    BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
                    BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
                ),
                limit=8,
            )
        )
        versioned_blocks: dict[str, BrainCoreMemoryBlockRecord] = {}
        for record in current_blocks:
            for version in self.list_core_memory_block_versions(
                block_kind=record.block_kind,
                scope_type=record.scope_type,
                scope_id=record.scope_id,
            ):
                versioned_blocks[version.block_id] = version
        return sorted(
            versioned_blocks.values(),
            key=lambda record: (
                record.scope_type,
                record.scope_id,
                record.block_kind,
                -int(record.version),
                record.block_id,
            ),
        )

    def _continuity_autobiography_entries(
        self,
        *,
        user_id: str,
        agent_id: str,
        presence_scope_key: str | None,
        autobiography_limit: int,
    ):
        """Return relationship and scene-linked autobiography entries for continuity reads."""
        relationship_scope_id = f"{agent_id}:{user_id}"
        combined = {
            record.entry_id: record
            for record in self.autobiographical_entries(
                scope_type="relationship",
                scope_id=relationship_scope_id,
                statuses=("current", "superseded"),
                limit=autobiography_limit,
            )
        }
        resolved_presence_scope_key = str(presence_scope_key or "").strip()
        if resolved_presence_scope_key:
            combined.update(
                {
                    record.entry_id: record
                    for record in self.autobiographical_entries(
                        scope_type="presence",
                        scope_id=resolved_presence_scope_key,
                        entry_kinds=(BrainAutobiographyEntryKind.SCENE_EPISODE.value,),
                        statuses=("current", "superseded"),
                        modalities=(BrainMultimodalAutobiographyModality.SCENE_WORLD.value,),
                        limit=max(8, autobiography_limit),
                    )
                }
            )
        return sorted(combined.values(), key=_autobiography_sort_key, reverse=True)

    def build_continuity_graph(
        self,
        *,
        user_id: str,
        thread_id: str,
        agent_id: str | None = None,
        scope_type: str = "user",
        scope_id: str | None = None,
        reference_ts: str | None = None,
        core_blocks: Iterable[BrainCoreMemoryBlockRecord] | None = None,
        commitment_projection: BrainCommitmentProjection | None = None,
        agenda: BrainAgendaProjection | None = None,
        procedural_skills: BrainProceduralSkillProjection | None = None,
        scene_world_state: BrainSceneWorldProjection | None = None,
        presence_scope_key: str | None = None,
        recent_event_limit: int = 192,
        current_claim_limit: int = 48,
        historical_claim_limit: int = 48,
        autobiography_limit: int = 24,
    ) -> BrainContinuityGraphProjection:
        """Build the derived temporal continuity graph for one thread/user scope."""
        resolved_scope_id = scope_id or (user_id if scope_type == "user" else thread_id)
        resolved_agent_id = agent_id or _DEFAULT_RELATIONSHIP_AGENT_ID
        current_claims = self.query_claims(
            temporal_mode="current",
            scope_type="user",
            scope_id=user_id,
            limit=current_claim_limit,
        )
        historical_claims = self.query_claims(
            temporal_mode="historical",
            scope_type="user",
            scope_id=user_id,
            limit=historical_claim_limit,
        )
        claim_evidence_by_id = {
            claim.claim_id: self._claims().claim_evidence(claim.claim_id)
            for claim in {*current_claims, *historical_claims}
        }
        recent_events = self.recent_brain_events(
            user_id=user_id,
            thread_id=thread_id,
            limit=recent_event_limit,
        )
        resolved_commitment_projection = (
            commitment_projection
            or self.get_session_commitment_projection(
                agent_id=resolved_agent_id,
                user_id=user_id,
                thread_id=thread_id,
            )
        )
        resolved_core_blocks = list(
            core_blocks
            or self._list_continuity_core_blocks(
                user_id=user_id,
                agent_id=resolved_agent_id,
            )
        )
        resolved_procedural_skills = procedural_skills or self.build_procedural_skill_projection(
            scope_type="thread",
            scope_id=thread_id,
        )
        resolved_scene_world_state = scene_world_state or self.build_scene_world_state_projection(
            user_id=user_id,
            thread_id=thread_id,
            reference_ts=reference_ts,
            recent_event_limit=recent_event_limit,
            agent_id=resolved_agent_id,
            presence_scope_key=presence_scope_key,
        )
        continuity_autobiography = self._continuity_autobiography_entries(
            user_id=user_id,
            agent_id=resolved_agent_id,
            presence_scope_key=presence_scope_key or resolved_scene_world_state.scope_id,
            autobiography_limit=autobiography_limit,
        )
        del agenda

        def _entity_lookup(entity_id: str) -> BrainEntityRecord | None:
            try:
                return self._entities().get_entity(entity_id)
            except KeyError:
                return None

        return build_continuity_graph_projection(
            scope_type=scope_type,
            scope_id=resolved_scope_id,
            current_claims=current_claims,
            historical_claims=historical_claims,
            claim_supersessions=[
                record
                for record in self.claim_supersessions()
                if record.prior_claim_id in claim_evidence_by_id
                or record.new_claim_id in claim_evidence_by_id
            ],
            autobiography=continuity_autobiography,
            commitment_projection=resolved_commitment_projection,
            core_blocks=resolved_core_blocks,
            procedural_skills=resolved_procedural_skills,
            scene_world_state=resolved_scene_world_state,
            recent_events=recent_events,
            claim_evidence_by_id=claim_evidence_by_id,
            entity_lookup=_entity_lookup,
            now=reference_ts,
        )

    def build_continuity_dossiers(
        self,
        *,
        user_id: str,
        thread_id: str,
        agent_id: str | None = None,
        scope_type: str = "user",
        scope_id: str | None = None,
        continuity_graph: BrainContinuityGraphProjection | None = None,
        reference_ts: str | None = None,
        core_blocks: Iterable[BrainCoreMemoryBlockRecord] | None = None,
        commitment_projection: BrainCommitmentProjection | None = None,
        agenda: BrainAgendaProjection | None = None,
        procedural_skills: BrainProceduralSkillProjection | None = None,
        scene_world_state: BrainSceneWorldProjection | None = None,
        presence_scope_key: str | None = None,
        recent_event_limit: int = 192,
        current_claim_limit: int = 48,
        historical_claim_limit: int = 48,
        autobiography_limit: int = 48,
    ) -> BrainContinuityDossierProjection:
        """Build the derived compiled dossier projection for one thread/user scope."""
        resolved_scope_id = scope_id or (user_id if scope_type == "user" else thread_id)
        resolved_agent_id = agent_id or _DEFAULT_RELATIONSHIP_AGENT_ID
        resolved_commitment_projection = (
            commitment_projection
            or self.get_session_commitment_projection(
                agent_id=resolved_agent_id,
                user_id=user_id,
                thread_id=thread_id,
            )
        )
        resolved_agenda = agenda or self.get_agenda_projection(scope_key=thread_id, user_id=user_id)
        resolved_core_blocks = list(
            core_blocks
            or self._list_continuity_core_blocks(
                user_id=user_id,
                agent_id=resolved_agent_id,
            )
        )
        resolved_procedural_skills = procedural_skills or self.build_procedural_skill_projection(
            scope_type="thread",
            scope_id=thread_id,
        )
        resolved_scene_world_state = scene_world_state or self.build_scene_world_state_projection(
            user_id=user_id,
            thread_id=thread_id,
            reference_ts=reference_ts,
            recent_event_limit=recent_event_limit,
            agent_id=resolved_agent_id,
            presence_scope_key=presence_scope_key,
        )
        continuity_autobiography = self._continuity_autobiography_entries(
            user_id=user_id,
            agent_id=resolved_agent_id,
            presence_scope_key=presence_scope_key or resolved_scene_world_state.scope_id,
            autobiography_limit=autobiography_limit,
        )
        graph = continuity_graph or self.build_continuity_graph(
            agent_id=resolved_agent_id,
            user_id=user_id,
            thread_id=thread_id,
            scope_type=scope_type,
            scope_id=resolved_scope_id,
            reference_ts=reference_ts,
            core_blocks=resolved_core_blocks,
            commitment_projection=resolved_commitment_projection,
            agenda=resolved_agenda,
            procedural_skills=resolved_procedural_skills,
            scene_world_state=resolved_scene_world_state,
            presence_scope_key=presence_scope_key,
            recent_event_limit=recent_event_limit,
            current_claim_limit=current_claim_limit,
            historical_claim_limit=historical_claim_limit,
            autobiography_limit=autobiography_limit,
        )
        return build_continuity_dossier_projection(
            scope_type=scope_type,
            scope_id=resolved_scope_id,
            thread_id=thread_id,
            current_claims=self.query_claims(
                temporal_mode="current",
                scope_type="user",
                scope_id=user_id,
                limit=current_claim_limit,
            ),
            historical_claims=self.query_claims(
                temporal_mode="historical",
                scope_type="user",
                scope_id=user_id,
                limit=historical_claim_limit,
            ),
            claim_supersessions=self.claim_supersessions(),
            autobiography=continuity_autobiography,
            continuity_graph=graph,
            core_blocks=resolved_core_blocks,
            commitment_projection=resolved_commitment_projection,
            agenda=resolved_agenda,
            procedural_skills=resolved_procedural_skills,
            scene_world_state=resolved_scene_world_state,
            recent_events=self.recent_brain_events(
                user_id=user_id,
                thread_id=thread_id,
                limit=recent_event_limit,
            ),
        )

    def build_procedural_trace_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        scope_type: str = "thread",
        scope_id: str | None = None,
        event_limit: int | None = None,
    ) -> BrainProceduralTraceProjection:
        """Build the derived procedural trace projection for one thread scope."""
        resolved_scope_id = scope_id or thread_id
        relevant_event_types = (
            BrainEventType.GOAL_CREATED,
            BrainEventType.GOAL_UPDATED,
            BrainEventType.GOAL_DEFERRED,
            BrainEventType.GOAL_RESUMED,
            BrainEventType.GOAL_CANCELLED,
            BrainEventType.GOAL_REPAIRED,
            BrainEventType.GOAL_COMPLETED,
            BrainEventType.GOAL_FAILED,
            BrainEventType.PLANNING_PROPOSED,
            BrainEventType.PLANNING_ADOPTED,
            BrainEventType.PLANNING_REJECTED,
            BrainEventType.CAPABILITY_REQUESTED,
            BrainEventType.CAPABILITY_COMPLETED,
            BrainEventType.CAPABILITY_FAILED,
            BrainEventType.CRITIC_FEEDBACK,
        )
        placeholders = ",".join("?" for _ in relevant_event_types)
        params: list[Any] = [user_id, thread_id, *relevant_event_types]
        if event_limit is None:
            rows = self._conn.execute(
                f"""
                SELECT * FROM brain_events
                WHERE user_id = ? AND thread_id = ? AND event_type IN ({placeholders})
                ORDER BY id ASC
                """,
                params,
            ).fetchall()
        else:
            params.append(int(event_limit))
            rows = self._conn.execute(
                f"""
                SELECT * FROM (
                    SELECT * FROM brain_events
                    WHERE user_id = ? AND thread_id = ? AND event_type IN ({placeholders})
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                params,
            ).fetchall()
        return build_procedural_trace_projection(
            scope_type=scope_type,
            scope_id=resolved_scope_id,
            events=[
                event for row in rows if (event := self._brain_event_from_row(row)) is not None
            ],
        )

    def consolidate_procedural_skills(
        self,
        *,
        user_id: str,
        thread_id: str,
        scope_type: str = "thread",
        scope_id: str | None = None,
    ) -> BrainProceduralSkillProjection:
        """Rebuild and persist the thread-scoped procedural skill set."""
        resolved_scope_id = scope_id or thread_id
        procedural_traces = self.build_procedural_trace_projection(
            user_id=user_id,
            thread_id=thread_id,
            scope_type=scope_type,
            scope_id=resolved_scope_id,
        )
        projection = derive_procedural_skill_projection(
            scope_type=scope_type,
            scope_id=resolved_scope_id,
            procedural_traces=procedural_traces,
        )
        self._conn.execute(
            """
            DELETE FROM procedural_skills
            WHERE scope_type = ? AND scope_id = ?
            """,
            (scope_type, resolved_scope_id),
        )
        for record in projection.skills:
            self._conn.execute(
                """
                INSERT INTO procedural_skills (
                    skill_id, scope_type, scope_id, skill_family_key, template_fingerprint,
                    title, purpose, goal_family, status, confidence,
                    activation_conditions_json, invariants_json, step_template_json,
                    required_capability_ids_json, effects_json, termination_conditions_json,
                    failure_signatures_json, stats_json, supporting_trace_ids_json,
                    supporting_outcome_ids_json, supporting_plan_proposal_ids_json,
                    supporting_commitment_ids_json, supersedes_skill_id, superseded_by_skill_id,
                    retirement_reason, created_at, updated_at, retired_at, details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.skill_id,
                    record.scope_type,
                    record.scope_id,
                    record.skill_family_key,
                    record.template_fingerprint,
                    record.title,
                    record.purpose,
                    record.goal_family,
                    record.status,
                    float(record.confidence),
                    json.dumps(
                        [item.as_dict() for item in record.activation_conditions],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    json.dumps(
                        [item.as_dict() for item in record.invariants],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    json.dumps(
                        [item.as_dict() for item in record.step_template],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    json.dumps(record.required_capability_ids, ensure_ascii=False, sort_keys=True),
                    json.dumps(
                        [item.as_dict() for item in record.effects],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    json.dumps(record.termination_conditions, ensure_ascii=False, sort_keys=True),
                    json.dumps(
                        [item.as_dict() for item in record.failure_signatures],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    json.dumps(record.stats.as_dict(), ensure_ascii=False, sort_keys=True),
                    json.dumps(record.supporting_trace_ids, ensure_ascii=False, sort_keys=True),
                    json.dumps(record.supporting_outcome_ids, ensure_ascii=False, sort_keys=True),
                    json.dumps(
                        record.supporting_plan_proposal_ids,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    json.dumps(
                        record.supporting_commitment_ids,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    record.supersedes_skill_id,
                    record.superseded_by_skill_id,
                    record.retirement_reason,
                    record.created_at,
                    record.updated_at,
                    record.retired_at,
                    json.dumps(record.details, ensure_ascii=False, sort_keys=True),
                ),
            )
        self._conn.commit()
        return projection

    def list_procedural_skills(
        self,
        *,
        scope_type: str,
        scope_id: str,
        statuses: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[BrainProceduralSkillRecord]:
        """Return persisted procedural skills for one scope."""
        query = [
            """
            SELECT * FROM procedural_skills
            WHERE scope_type = ? AND scope_id = ?
            """
        ]
        params: list[Any] = [scope_type, scope_id]
        normalized_statuses = [str(item).strip() for item in statuses or () if str(item).strip()]
        if normalized_statuses:
            placeholders = ",".join("?" for _ in normalized_statuses)
            query.append(f"AND status IN ({placeholders})")
            params.extend(normalized_statuses)
        query.append("ORDER BY updated_at DESC, skill_id ASC")
        if limit is not None:
            query.append("LIMIT ?")
            params.append(int(limit))
        rows = self._conn.execute("\n".join(query), params).fetchall()
        records = [
            record for row in rows if (record := self._procedural_skill_from_row(row)) is not None
        ]
        return sorted(
            records,
            key=lambda record: (
                record.status,
                record.goal_family,
                record.template_fingerprint,
                record.skill_id,
            ),
        )

    def build_procedural_skill_projection(
        self,
        *,
        scope_type: str,
        scope_id: str,
    ) -> BrainProceduralSkillProjection:
        """Return the persisted procedural skill projection for one scope."""
        skills = self.list_procedural_skills(scope_type=scope_type, scope_id=scope_id)
        skill_counts: dict[str, int] = {}
        confidence_band_counts: dict[str, int] = {}
        candidate_skill_ids: list[str] = []
        active_skill_ids: list[str] = []
        superseded_skill_ids: list[str] = []
        retired_skill_ids: list[str] = []
        for record in skills:
            skill_counts[record.status] = skill_counts.get(record.status, 0) + 1
            if float(record.confidence) < 0.5:
                band = "low"
            elif float(record.confidence) < 0.75:
                band = "medium"
            else:
                band = "high"
            confidence_band_counts[band] = confidence_band_counts.get(band, 0) + 1
            if record.status == "candidate":
                candidate_skill_ids.append(record.skill_id)
            elif record.status == "active":
                active_skill_ids.append(record.skill_id)
            elif record.status == "superseded":
                superseded_skill_ids.append(record.skill_id)
            elif record.status == "retired":
                retired_skill_ids.append(record.skill_id)
        return BrainProceduralSkillProjection(
            scope_type=scope_type,
            scope_id=scope_id,
            skill_counts=dict(sorted(skill_counts.items())),
            confidence_band_counts=dict(sorted(confidence_band_counts.items())),
            skills=skills,
            candidate_skill_ids=sorted(candidate_skill_ids),
            active_skill_ids=sorted(active_skill_ids),
            superseded_skill_ids=sorted(superseded_skill_ids),
            retired_skill_ids=sorted(retired_skill_ids),
        )

    def upsert_autobiographical_entry(
        self,
        *,
        scope_type: str,
        scope_id: str,
        entry_kind: str,
        rendered_summary: str,
        content: dict[str, Any],
        salience: float,
        source_episode_ids: list[int] | None = None,
        source_claim_ids: list[str] | None = None,
        source_event_ids: list[str] | None = None,
        source_event_id: str | None = None,
        valid_from: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        append_only: bool = False,
        identity_key: str | None = None,
        modality: str | None = None,
        review_state: str | None = None,
        retention_class: str | None = None,
        privacy_class: str | None = None,
        governance_reason_codes: list[str] | None = None,
        last_governance_event_id: str | None = None,
        source_presence_scope_key: str | None = None,
        source_scene_entity_ids: list[str] | None = None,
        source_scene_affordance_ids: list[str] | None = None,
        redacted_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ):
        """Insert or version one autobiographical entry."""
        return self._autobiography().upsert_entry(
            scope_type=scope_type,
            scope_id=scope_id,
            entry_kind=entry_kind,
            rendered_summary=rendered_summary,
            content=content,
            salience=salience,
            source_episode_ids=source_episode_ids,
            source_claim_ids=source_claim_ids,
            source_event_ids=source_event_ids,
            source_event_id=source_event_id,
            valid_from=valid_from,
            created_at=created_at,
            updated_at=updated_at,
            append_only=append_only,
            identity_key=identity_key,
            modality=modality,
            review_state=review_state,
            retention_class=retention_class,
            privacy_class=privacy_class,
            governance_reason_codes=governance_reason_codes,
            last_governance_event_id=last_governance_event_id,
            source_presence_scope_key=source_presence_scope_key,
            source_scene_entity_ids=source_scene_entity_ids,
            source_scene_affordance_ids=source_scene_affordance_ids,
            redacted_at=redacted_at,
            event_context=event_context,
        )

    def autobiographical_entries(
        self,
        *,
        scope_type: str | None = None,
        scope_id: str | None = None,
        entry_kinds: tuple[str, ...] | None = None,
        statuses: tuple[str, ...] | None = ("current",),
        review_states: tuple[str, ...] | None = None,
        retention_classes: tuple[str, ...] | None = None,
        privacy_classes: tuple[str, ...] | None = None,
        modalities: tuple[str, ...] | None = None,
        limit: int = 16,
    ):
        """Return autobiographical entries for the requested scope."""
        return self._autobiography().list_entries(
            scope_type=scope_type,
            scope_id=scope_id,
            entry_kinds=entry_kinds,
            statuses=statuses,
            review_states=review_states,
            retention_classes=retention_classes,
            privacy_classes=privacy_classes,
            modalities=modalities,
            limit=limit,
        )

    def latest_autobiographical_entry(
        self,
        *,
        scope_type: str,
        scope_id: str,
        entry_kind: str,
    ):
        """Return the latest current autobiographical entry for one scope/kind."""
        return self._autobiography().latest_entry(
            scope_type=scope_type,
            scope_id=scope_id,
            entry_kind=entry_kind,
        )

    def request_autobiographical_entry_review(
        self,
        entry_id: str,
        *,
        source_event_id: str | None,
        reason_codes: list[str] | None = None,
        summary: str | None = None,
        notes: str | None = None,
        updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ):
        """Move one autobiographical entry into explicit requested review."""
        return self._autobiography().request_entry_review(
            entry_id,
            source_event_id=source_event_id,
            reason_codes=reason_codes,
            summary=summary,
            notes=notes,
            updated_at=updated_at,
            event_context=event_context,
        )

    def reclassify_autobiographical_entry_retention(
        self,
        entry_id: str,
        *,
        retention_class: str,
        source_event_id: str | None,
        reason_codes: list[str] | None = None,
        summary: str | None = None,
        notes: str | None = None,
        updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ):
        """Reclassify autobiographical retention without bypassing the event spine."""
        return self._autobiography().reclassify_entry_retention(
            entry_id,
            retention_class=retention_class,
            source_event_id=source_event_id,
            reason_codes=reason_codes,
            summary=summary,
            notes=notes,
            updated_at=updated_at,
            event_context=event_context,
        )

    def redact_autobiographical_entry(
        self,
        entry_id: str,
        *,
        redacted_summary: str,
        source_event_id: str | None,
        reason_codes: list[str] | None = None,
        summary: str | None = None,
        notes: str | None = None,
        updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ):
        """Redact rendered autobiographical content without deleting the canonical row."""
        return self._autobiography().redact_entry(
            entry_id,
            redacted_summary=redacted_summary,
            source_event_id=source_event_id,
            reason_codes=reason_codes,
            summary=summary,
            notes=notes,
            updated_at=updated_at,
            event_context=event_context,
        )

    def refresh_scene_episode_autobiography(
        self,
        *,
        user_id: str,
        thread_id: str,
        session_id: str | None,
        agent_id: str | None,
        presence_scope_key: str,
        scene_world_state: BrainSceneWorldProjection | None = None,
        recent_events: Iterable[BrainEventRecord] | None = None,
        source_event_id: str | None = None,
        updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ):
        """Refresh the bounded current scene episode for one presence scope."""
        resolved_agent_id = agent_id or _DEFAULT_RELATIONSHIP_AGENT_ID
        projection = scene_world_state or self.build_scene_world_state_projection(
            user_id=user_id,
            thread_id=thread_id,
            reference_ts=updated_at,
            recent_event_limit=96,
            agent_id=resolved_agent_id,
            presence_scope_key=presence_scope_key,
        )
        resolved_events = list(
            recent_events
            or self.recent_brain_events(
                user_id=user_id,
                thread_id=thread_id,
                limit=48,
            )
        )
        relevant_events = [
            event
            for event in resolved_events
            if event.event_type
            in {
                BrainEventType.PERCEPTION_OBSERVED,
                BrainEventType.SCENE_CHANGED,
                BrainEventType.ENGAGEMENT_CHANGED,
                BrainEventType.ATTENTION_CHANGED,
            }
            and (_event_presence_scope_key(event) in {None, presence_scope_key})
        ]
        distilled = distill_scene_episode(
            scene_world_state=projection,
            recent_events=relevant_events,
            presence_scope_key=presence_scope_key,
            reference_ts=updated_at or projection.updated_at,
        )
        if distilled is None:
            return None
        current = self.latest_autobiographical_entry(
            scope_type="presence",
            scope_id=presence_scope_key,
            entry_kind=BrainAutobiographyEntryKind.SCENE_EPISODE.value,
        )
        current_multimodal = parse_multimodal_autobiography_record(current)
        if current_multimodal is not None and (
            current_multimodal.content.get("semantic_fingerprint") == distilled.semantic_fingerprint
        ):
            return current
        return self.upsert_autobiographical_entry(
            scope_type="presence",
            scope_id=presence_scope_key,
            entry_kind=BrainAutobiographyEntryKind.SCENE_EPISODE.value,
            rendered_summary=distilled.rendered_summary,
            content=distilled.content,
            salience=distilled.salience,
            source_event_ids=list(distilled.source_event_ids),
            source_event_id=source_event_id,
            identity_key=presence_scope_key,
            created_at=distilled.valid_from,
            updated_at=distilled.valid_from,
            modality=BrainMultimodalAutobiographyModality.SCENE_WORLD.value,
            review_state=distilled.review_state,
            retention_class=distilled.retention_class,
            privacy_class=distilled.privacy_class,
            governance_reason_codes=list(distilled.governance_reason_codes),
            source_presence_scope_key=distilled.source_presence_scope_key,
            source_scene_entity_ids=list(distilled.source_scene_entity_ids),
            source_scene_affordance_ids=list(distilled.source_scene_affordance_ids),
            valid_from=distilled.valid_from,
            event_context=event_context
            or (
                {
                    "agent_id": resolved_agent_id,
                    "user_id": user_id,
                    "session_id": session_id or thread_id,
                    "thread_id": thread_id,
                    "source": "memory_v2",
                }
                if session_id is not None
                else None
            ),
        )

    def record_memory_health_report(
        self,
        *,
        scope_type: str,
        scope_id: str,
        cycle_id: str,
        score: float,
        status: str,
        findings: list[dict[str, Any]],
        stats: dict[str, Any],
        artifact_path: str | None = None,
        source_event_id: str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainMemoryHealthReportRecord:
        """Persist one inspectable memory-health report."""
        return self._health().record_report(
            scope_type=scope_type,
            scope_id=scope_id,
            cycle_id=cycle_id,
            score=score,
            status=status,
            findings=findings,
            stats=stats,
            artifact_path=artifact_path,
            source_event_id=source_event_id,
            event_context=event_context,
        )

    def latest_memory_health_report(
        self,
        *,
        scope_type: str,
        scope_id: str,
    ) -> BrainMemoryHealthReportRecord | None:
        """Return the latest memory-health report for one scope."""
        return self._health().latest_report(scope_type=scope_type, scope_id=scope_id)

    def start_reflection_cycle(
        self,
        *,
        cycle_id: str,
        user_id: str,
        thread_id: str,
        trigger: str,
        input_episode_cursor: int,
        input_event_cursor: int,
    ) -> BrainReflectionCycleRecord:
        """Insert one running reflection-cycle row."""
        started_at = _utc_now()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO reflection_cycles (
                cycle_id, user_id, thread_id, trigger, status,
                input_episode_cursor, input_event_cursor,
                terminal_episode_cursor, terminal_event_cursor,
                draft_artifact_path, result_stats_json, skip_reason, error_json,
                started_at, completed_at
            )
            VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, NULL, '{}', NULL, NULL, ?, NULL)
            """,
            (
                cycle_id,
                user_id,
                thread_id,
                trigger,
                int(input_episode_cursor),
                int(input_event_cursor),
                int(input_episode_cursor),
                int(input_event_cursor),
                started_at,
            ),
        )
        self._conn.commit()
        record = self.get_reflection_cycle(cycle_id=cycle_id)
        if record is None:
            raise RuntimeError("Failed to start reflection cycle.")
        return record

    def complete_reflection_cycle(
        self,
        *,
        cycle_id: str,
        terminal_episode_cursor: int,
        terminal_event_cursor: int,
        draft_artifact_path: str | None,
        result_stats: dict[str, Any],
    ) -> BrainReflectionCycleRecord:
        """Mark one running reflection cycle as completed."""
        completed_at = _utc_now()
        self._conn.execute(
            """
            UPDATE reflection_cycles
            SET status = 'completed',
                terminal_episode_cursor = ?,
                terminal_event_cursor = ?,
                draft_artifact_path = ?,
                result_stats_json = ?,
                completed_at = ?
            WHERE cycle_id = ?
            """,
            (
                int(terminal_episode_cursor),
                int(terminal_event_cursor),
                draft_artifact_path,
                json.dumps(result_stats or {}, ensure_ascii=False, sort_keys=True),
                completed_at,
                cycle_id,
            ),
        )
        self._conn.commit()
        record = self.get_reflection_cycle(cycle_id=cycle_id)
        if record is None:
            raise RuntimeError("Failed to complete reflection cycle.")
        return record

    def skip_reflection_cycle(
        self,
        *,
        cycle_id: str,
        user_id: str,
        thread_id: str,
        trigger: str,
        input_episode_cursor: int,
        input_event_cursor: int,
        terminal_episode_cursor: int,
        terminal_event_cursor: int,
        skip_reason: str,
        result_stats: dict[str, Any] | None = None,
    ) -> BrainReflectionCycleRecord:
        """Insert one skipped reflection-cycle row."""
        started_at = _utc_now()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO reflection_cycles (
                cycle_id, user_id, thread_id, trigger, status,
                input_episode_cursor, input_event_cursor,
                terminal_episode_cursor, terminal_event_cursor,
                draft_artifact_path, result_stats_json, skip_reason, error_json,
                started_at, completed_at
            )
            VALUES (?, ?, ?, ?, 'skipped', ?, ?, ?, ?, NULL, ?, ?, NULL, ?, ?)
            """,
            (
                cycle_id,
                user_id,
                thread_id,
                trigger,
                int(input_episode_cursor),
                int(input_event_cursor),
                int(terminal_episode_cursor),
                int(terminal_event_cursor),
                json.dumps(result_stats or {}, ensure_ascii=False, sort_keys=True),
                skip_reason,
                started_at,
                started_at,
            ),
        )
        self._conn.commit()
        record = self.get_reflection_cycle(cycle_id=cycle_id)
        if record is None:
            raise RuntimeError("Failed to persist skipped reflection cycle.")
        return record

    def fail_reflection_cycle(
        self,
        *,
        cycle_id: str,
        error_payload: dict[str, Any],
        draft_artifact_path: str | None = None,
        terminal_episode_cursor: int | None = None,
        terminal_event_cursor: int | None = None,
        result_stats: dict[str, Any] | None = None,
    ) -> BrainReflectionCycleRecord:
        """Mark one reflection cycle as failed."""
        completed_at = _utc_now()
        record = self.get_reflection_cycle(cycle_id=cycle_id)
        if record is None:
            raise KeyError(f"Missing reflection cycle {cycle_id}")
        self._conn.execute(
            """
            UPDATE reflection_cycles
            SET status = 'failed',
                terminal_episode_cursor = ?,
                terminal_event_cursor = ?,
                draft_artifact_path = COALESCE(?, draft_artifact_path),
                result_stats_json = ?,
                error_json = ?,
                completed_at = ?
            WHERE cycle_id = ?
            """,
            (
                int(
                    terminal_episode_cursor
                    if terminal_episode_cursor is not None
                    else record.terminal_episode_cursor
                ),
                int(
                    terminal_event_cursor
                    if terminal_event_cursor is not None
                    else record.terminal_event_cursor
                ),
                draft_artifact_path,
                json.dumps(result_stats or record.result_stats, ensure_ascii=False, sort_keys=True),
                json.dumps(error_payload or {}, ensure_ascii=False, sort_keys=True),
                completed_at,
                cycle_id,
            ),
        )
        self._conn.commit()
        failed = self.get_reflection_cycle(cycle_id=cycle_id)
        if failed is None:
            raise RuntimeError("Failed to mark reflection cycle as failed.")
        return failed

    def get_reflection_cycle(self, *, cycle_id: str) -> BrainReflectionCycleRecord | None:
        """Return one reflection-cycle row by id."""
        row = self._conn.execute(
            "SELECT * FROM reflection_cycles WHERE cycle_id = ?",
            (cycle_id,),
        ).fetchone()
        if row is None:
            return None
        return self._reflection_cycle_from_row(row)

    def list_reflection_cycles(
        self,
        *,
        user_id: str | None = None,
        thread_id: str | None = None,
        statuses: tuple[str, ...] | None = None,
        limit: int = 16,
    ) -> list[BrainReflectionCycleRecord]:
        """Return reflection-cycle rows for one optional scope and status filter."""
        clauses: list[str] = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(statuses)
        query = "SELECT * FROM reflection_cycles"
        if clauses:
            query += f" WHERE {' AND '.join(clauses)}"
        query += " ORDER BY started_at DESC LIMIT ?"
        rows = self._conn.execute(query, (*params, limit)).fetchall()
        return [self._reflection_cycle_from_row(row) for row in rows]

    def latest_reflection_cursors(self, *, user_id: str, thread_id: str) -> tuple[int, int]:
        """Return the terminal cursors from the latest completed reflection cycle."""
        rows = self.list_reflection_cycles(
            user_id=user_id,
            thread_id=thread_id,
            statuses=("completed",),
            limit=1,
        )
        if not rows:
            return 0, 0
        return rows[0].terminal_episode_cursor, rows[0].terminal_event_cursor

    def has_pending_reflection_work(self, *, user_id: str, thread_id: str) -> bool:
        """Return whether new episodes or events exist beyond the last completed cycle."""
        episode_cursor, event_cursor = self.latest_reflection_cursors(
            user_id=user_id,
            thread_id=thread_id,
        )
        latest_episode_row = self._conn.execute(
            """
            SELECT COALESCE(MAX(id), 0) AS max_id
            FROM episodes
            WHERE user_id = ? AND thread_id = ?
            """,
            (user_id, thread_id),
        ).fetchone()
        latest_event_row = self._conn.execute(
            """
            SELECT COALESCE(MAX(id), 0) AS max_id
            FROM brain_events
            WHERE user_id = ? AND thread_id = ?
            """,
            (user_id, thread_id),
        ).fetchone()
        return (
            int(latest_episode_row["max_id"]) > episode_cursor
            or int(latest_event_row["max_id"]) > event_cursor
        )

    def backfill_continuity_from_legacy(
        self,
        *,
        user_id: str,
        thread_id: str,
        agent_id: str = _DEFAULT_RELATIONSHIP_AGENT_ID,
    ):
        """Idempotently backfill continuity tables from legacy semantic and narrative state."""
        registry = self._entities()
        ledger = self._claims()
        user_entity = registry.ensure_entity(
            entity_type="user",
            canonical_name=user_id,
            aliases=[user_id],
            attributes={"user_id": user_id},
        )
        semantic_rows = self._conn.execute(
            """
            SELECT * FROM memory_semantic
            WHERE user_id = ?
            ORDER BY id ASC
            """,
            (user_id,),
        ).fetchall()
        for row in semantic_rows:
            record = self._semantic_memory_from_row(row)
            object_value = str(record.value.get("value", "")).strip()
            object_entity_id = None
            if record.namespace.startswith("preference.") and object_value:
                object_entity_id = registry.ensure_entity(
                    entity_type="topic",
                    canonical_name=object_value,
                    aliases=[record.subject],
                ).entity_id
            ledger.record_claim(
                subject_entity_id=user_entity.entity_id,
                predicate=record.namespace,
                object_entity_id=object_entity_id,
                object_value=object_value,
                object_data=record.value,
                status="active" if record.status == "active" else "superseded",
                confidence=record.confidence,
                valid_from=record.observed_at,
                valid_to=record.updated_at if record.status != "active" else None,
                source_event_id=record.source_event_id,
                scope_type="user",
                scope_id=user_id,
                claim_key=record.contradiction_key,
                stale_after_seconds=record.stale_after_seconds,
                evidence_summary=record.rendered_text,
                evidence_json={
                    "source": "migration",
                    "legacy_table": "memory_semantic",
                    "legacy_memory_id": record.id,
                    "provenance": record.provenance,
                },
                source_episode_id=record.source_episode_id,
            )

        current_profile: dict[str, Any] = {}
        for claim in self.query_claims(
            temporal_mode="current",
            subject_entity_id=user_entity.entity_id,
            scope_type="user",
            scope_id=user_id,
            limit=24,
        ):
            if claim.predicate == "profile.name":
                current_profile["name"] = claim.object.get("value")
            elif claim.predicate == "profile.role":
                current_profile["role"] = claim.object.get("value")
            elif claim.predicate == "profile.origin":
                current_profile["origin"] = claim.object.get("value")
        if current_profile:
            self.upsert_core_memory_block(
                block_kind=BrainCoreMemoryBlockKind.USER_CORE.value,
                scope_type="user",
                scope_id=user_id,
                content=current_profile,
                source_event_id=None,
            )

        self._refresh_self_core_block(
            agent_id=agent_id,
            source_event_id=None,
        )
        self._refresh_self_persona_core_block(
            agent_id=agent_id,
            source_event_id=None,
        )

        commitments = self.narrative_memories(
            user_id=user_id,
            thread_id=thread_id,
            kinds=("commitment",),
            statuses=("open",),
            limit=32,
        )
        relationship_scope_id = self._relationship_scope_id(agent_id=agent_id, user_id=user_id)
        self.upsert_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.ACTIVE_COMMITMENTS.value,
            scope_type="relationship",
            scope_id=relationship_scope_id,
            content={
                "commitments": [
                    {
                        "title": record.title,
                        "summary": record.summary,
                        "details": record.details,
                        "status": record.status,
                        "updated_at": record.updated_at,
                    }
                    for record in commitments
                ]
            },
            source_event_id=None,
        )
        relationship_core = {
            "thread_id": thread_id,
            "last_session_summary": self.get_session_summary(user_id=user_id, thread_id=thread_id),
            "open_commitments": [record.title for record in commitments],
        }
        self.upsert_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_CORE.value,
            scope_type="relationship",
            scope_id=relationship_scope_id,
            content=relationship_core,
            source_event_id=None,
        )
        self._refresh_relationship_persona_blocks(
            user_id=user_id,
            thread_id=thread_id,
            agent_id=agent_id,
            session_id=None,
            source_event_id=None,
        )

    def apply_memory_event(self, event: BrainEventRecord):
        """Apply one replayed continuity-memory mutation event to the local store."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        if event.event_type == BrainEventType.MEMORY_BLOCK_UPSERTED:
            block_kind = str(payload.get("block_kind", "")).strip()
            scope_type = str(payload.get("scope_type", "")).strip()
            scope_id = str(payload.get("scope_id", "")).strip()
            content = dict(payload.get("content", {}))
            if block_kind and scope_type and scope_id:
                self.upsert_core_memory_block(
                    block_kind=block_kind,
                    scope_type=scope_type,
                    scope_id=scope_id,
                    content=content,
                    source_event_id=str(payload.get("source_event_id", "")).strip()
                    or event.causal_parent_id,
                )
            return

        if event.event_type == BrainEventType.MEMORY_CLAIM_RECORDED:
            explicit_claim_id = str(payload.get("claim_id", "")).strip()
            if explicit_claim_id:
                source_event_id = (
                    str(payload.get("source_event_id", "")).strip() or event.causal_parent_id
                )
                source_episode_id = payload.get("source_episode_id")
                evidence_summary = str(payload.get("evidence_summary", "")).strip()
                evidence_json = dict(payload.get("evidence_json", {}))
                governance_state = seed_claim_governance(
                    scope_type=str(payload.get("scope_type", "")).strip() or None,
                    predicate=str(payload.get("predicate", "")).strip(),
                    truth_status=str(payload.get("status", "active")),
                    updated_at=event.ts,
                    currentness_status=str(payload.get("currentness_status", "")).strip() or None,
                    review_state=str(payload.get("review_state", "")).strip() or None,
                    retention_class=str(payload.get("retention_class", "")).strip() or None,
                    reason_codes=list(payload.get("reason_codes") or []),
                    last_governance_event_id=event.event_id,
                )
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO claims (
                        claim_id, subject_entity_id, predicate, object_entity_id, object_json,
                        status, confidence, valid_from, valid_to, source_event_id, scope_type,
                        scope_id, claim_key, stale_after_seconds, currentness_status,
                        review_state, retention_class, governance_reason_codes_json,
                        last_governance_event_id, governance_updated_at, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        explicit_claim_id,
                        str(payload.get("subject_entity_id", "")).strip(),
                        str(payload.get("predicate", "")).strip(),
                        str(payload.get("object_entity_id", "")).strip() or None,
                        json.dumps(
                            dict(payload.get("object", {})), ensure_ascii=False, sort_keys=True
                        ),
                        str(payload.get("status", "active")),
                        float(payload.get("confidence", 1.0)),
                        str(payload.get("valid_from") or event.ts),
                        str(payload.get("valid_to")) if payload.get("valid_to") else None,
                        source_event_id,
                        str(payload.get("scope_type", "")).strip() or None,
                        str(payload.get("scope_id", "")).strip() or None,
                        str(payload.get("claim_key", "")).strip() or None,
                        (
                            int(payload.get("stale_after_seconds"))
                            if payload.get("stale_after_seconds") is not None
                            else None
                        ),
                        governance_state.currentness_status,
                        governance_state.review_state,
                        governance_state.retention_class,
                        json.dumps(
                            list(governance_state.reason_codes),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        governance_state.last_governance_event_id,
                        governance_state.governance_updated_at,
                        event.ts,
                        event.ts,
                    ),
                )
                if any(
                    value is not None and value != ""
                    for value in (
                        source_event_id,
                        source_episode_id,
                        evidence_summary,
                        evidence_json,
                    )
                ):
                    evidence_id = f"replay:{event.event_id}:evidence"
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO claim_evidence (
                            evidence_id, claim_id, source_event_id, source_episode_id,
                            evidence_summary, evidence_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            evidence_id,
                            explicit_claim_id,
                            source_event_id,
                            int(source_episode_id) if source_episode_id is not None else None,
                            evidence_summary,
                            json.dumps(evidence_json, ensure_ascii=False, sort_keys=True),
                            event.ts,
                        ),
                    )
                self._claims()._upsert_claim_fts(explicit_claim_id)
                self._claims()._refresh_claim_governance_projection_for_scope(
                    scope_type=str(payload.get("scope_type", "")).strip() or None,
                    scope_id=str(payload.get("scope_id", "")).strip() or None,
                    source_event_id=event.event_id,
                    updated_at=event.ts,
                    commit=False,
                )
                self._conn.commit()
            else:
                self._claims().record_claim(
                    subject_entity_id=str(payload.get("subject_entity_id", "")).strip(),
                    predicate=str(payload.get("predicate", "")).strip(),
                    object_entity_id=str(payload.get("object_entity_id", "")).strip() or None,
                    object_data=dict(payload.get("object", {})),
                    status=str(payload.get("status", "active")),
                    confidence=float(payload.get("confidence", 1.0)),
                    valid_from=str(payload.get("valid_from") or event.ts),
                    valid_to=str(payload.get("valid_to")) if payload.get("valid_to") else None,
                    source_event_id=str(payload.get("source_event_id", "")).strip()
                    or event.causal_parent_id,
                    scope_type=str(payload.get("scope_type", "")).strip() or None,
                    scope_id=str(payload.get("scope_id", "")).strip() or None,
                    claim_key=str(payload.get("claim_key", "")).strip() or None,
                    stale_after_seconds=(
                        int(payload.get("stale_after_seconds"))
                        if payload.get("stale_after_seconds") is not None
                        else None
                    ),
                    currentness_status=str(payload.get("currentness_status", "")).strip() or None,
                    review_state=str(payload.get("review_state", "")).strip() or None,
                    retention_class=str(payload.get("retention_class", "")).strip() or None,
                    reason_codes=list(payload.get("reason_codes") or []),
                    last_governance_event_id=event.event_id,
                    governance_updated_at=event.ts,
                    event_context=None,
                )
            return

        if event.event_type == BrainEventType.MEMORY_CLAIM_REVOKED:
            claim_id = str(payload.get("claim_id", "")).strip()
            if claim_id:
                self._claims().revoke_claim(
                    claim_id,
                    reason=str(payload.get("reason", "revoked")),
                    source_event_id=event.event_id,
                    reason_codes=list(payload.get("reason_codes") or []) or None,
                    updated_at=event.ts,
                    event_context=None,
                )
            return

        if event.event_type == BrainEventType.MEMORY_CLAIM_SUPERSEDED:
            prior_claim_id = str(payload.get("prior_claim_id", "")).strip()
            new_claim_id = str(payload.get("new_claim_id", "")).strip()
            if prior_claim_id:
                self._claims()._apply_claim_governance_update(
                    claim_id=prior_claim_id,
                    truth_status="superseded",
                    valid_to=event.ts,
                    governance_transition="superseded",
                    source_event_id=event.event_id,
                    updated_at=event.ts,
                    reason_codes=list(payload.get("reason_codes") or []) or None,
                )
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO claim_supersessions (
                        supersession_id, prior_claim_id, new_claim_id, reason, source_event_id, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"replay:{event.event_id}",
                        prior_claim_id,
                        new_claim_id,
                        str(payload.get("reason", "replay_supersession")),
                        event.event_id,
                        event.ts,
                    ),
                )
                self._conn.commit()
                self._claims()._upsert_claim_fts(prior_claim_id)
            return

        if event.event_type == BrainEventType.MEMORY_CLAIM_REVIEW_REQUESTED:
            claim_id = str(payload.get("claim_id", "")).strip()
            if claim_id:
                self._claims().request_claim_review(
                    claim_id,
                    source_event_id=event.event_id,
                    reason_codes=list(payload.get("reason_codes") or []) or None,
                    summary=str(payload.get("summary", "")).strip() or None,
                    notes=str(payload.get("notes", "")).strip() or None,
                    updated_at=event.ts,
                    event_context=None,
                )
            return

        if event.event_type == BrainEventType.MEMORY_CLAIM_EXPIRED:
            claim_id = str(payload.get("claim_id", "")).strip()
            if claim_id:
                self._claims().expire_claim(
                    claim_id,
                    source_event_id=event.event_id,
                    reason_codes=list(payload.get("reason_codes") or []) or None,
                    review_state=str(payload.get("review_state", "")).strip() or None,
                    summary=str(payload.get("summary", "")).strip() or None,
                    notes=str(payload.get("notes", "")).strip() or None,
                    updated_at=event.ts,
                    event_context=None,
                )
            return

        if event.event_type == BrainEventType.MEMORY_CLAIM_REVALIDATED:
            claim_id = str(payload.get("claim_id", "")).strip()
            if claim_id:
                self._claims().revalidate_claim(
                    claim_id,
                    source_event_id=event.event_id,
                    confidence=(
                        float(payload["confidence"])
                        if payload.get("confidence") is not None
                        else None
                    ),
                    reason_codes=list(payload.get("reason_codes") or [])
                    if "reason_codes" in payload
                    else None,
                    summary=str(payload.get("summary", "")).strip() or None,
                    notes=str(payload.get("notes", "")).strip() or None,
                    updated_at=event.ts,
                    event_context=None,
                )
            return

        if event.event_type == BrainEventType.MEMORY_CLAIM_RETENTION_RECLASSIFIED:
            claim_id = str(payload.get("claim_id", "")).strip()
            retention_class = str(payload.get("retention_class", "")).strip()
            if claim_id and retention_class:
                self._claims().reclassify_claim_retention(
                    claim_id,
                    retention_class=retention_class,
                    source_event_id=event.event_id,
                    reason_codes=list(payload.get("reason_codes") or [])
                    if "reason_codes" in payload
                    else None,
                    summary=str(payload.get("summary", "")).strip() or None,
                    notes=str(payload.get("notes", "")).strip() or None,
                    updated_at=event.ts,
                    event_context=None,
                )
            return

        if event.event_type == BrainEventType.REFLECTION_CYCLE_STARTED:
            cycle_id = str(payload.get("cycle_id", "")).strip()
            if cycle_id and self.get_reflection_cycle(cycle_id=cycle_id) is None:
                self.start_reflection_cycle(
                    cycle_id=cycle_id,
                    user_id=event.user_id,
                    thread_id=event.thread_id,
                    trigger=str(payload.get("trigger", "manual")),
                    input_episode_cursor=int(payload.get("input_episode_cursor", 0)),
                    input_event_cursor=int(payload.get("input_event_cursor", 0)),
                )
                if payload.get("draft_artifact_path"):
                    self._conn.execute(
                        """
                        UPDATE reflection_cycles
                        SET draft_artifact_path = ?
                        WHERE cycle_id = ?
                        """,
                        (str(payload.get("draft_artifact_path")), cycle_id),
                    )
                    self._conn.commit()
            return

        if event.event_type == BrainEventType.REFLECTION_CYCLE_COMPLETED:
            cycle_id = str(payload.get("cycle_id", "")).strip()
            if cycle_id:
                if self.get_reflection_cycle(cycle_id=cycle_id) is None:
                    self.start_reflection_cycle(
                        cycle_id=cycle_id,
                        user_id=event.user_id,
                        thread_id=event.thread_id,
                        trigger=str(payload.get("trigger", "manual")),
                        input_episode_cursor=int(payload.get("input_episode_cursor", 0)),
                        input_event_cursor=int(payload.get("input_event_cursor", 0)),
                    )
                self.complete_reflection_cycle(
                    cycle_id=cycle_id,
                    terminal_episode_cursor=int(
                        payload.get(
                            "terminal_episode_cursor", payload.get("input_episode_cursor", 0)
                        )
                    ),
                    terminal_event_cursor=int(
                        payload.get("terminal_event_cursor", payload.get("input_event_cursor", 0))
                    ),
                    draft_artifact_path=str(payload.get("draft_artifact_path", "")).strip() or None,
                    result_stats=dict(payload.get("result_stats", {})),
                )
            return

        if event.event_type == BrainEventType.REFLECTION_CYCLE_SKIPPED:
            cycle_id = str(payload.get("cycle_id", "")).strip()
            if cycle_id:
                self.skip_reflection_cycle(
                    cycle_id=cycle_id,
                    user_id=event.user_id,
                    thread_id=event.thread_id,
                    trigger=str(payload.get("trigger", "timer")),
                    input_episode_cursor=int(payload.get("input_episode_cursor", 0)),
                    input_event_cursor=int(payload.get("input_event_cursor", 0)),
                    terminal_episode_cursor=int(
                        payload.get(
                            "terminal_episode_cursor", payload.get("input_episode_cursor", 0)
                        )
                    ),
                    terminal_event_cursor=int(
                        payload.get("terminal_event_cursor", payload.get("input_event_cursor", 0))
                    ),
                    skip_reason=str(payload.get("skip_reason", "skipped")),
                    result_stats=dict(payload.get("result_stats", {})),
                )
            return

        if event.event_type == BrainEventType.REFLECTION_CYCLE_FAILED:
            cycle_id = str(payload.get("cycle_id", "")).strip()
            if cycle_id:
                if self.get_reflection_cycle(cycle_id=cycle_id) is None:
                    self.start_reflection_cycle(
                        cycle_id=cycle_id,
                        user_id=event.user_id,
                        thread_id=event.thread_id,
                        trigger=str(payload.get("trigger", "manual")),
                        input_episode_cursor=int(payload.get("input_episode_cursor", 0)),
                        input_event_cursor=int(payload.get("input_event_cursor", 0)),
                    )
                self.fail_reflection_cycle(
                    cycle_id=cycle_id,
                    draft_artifact_path=str(payload.get("draft_artifact_path", "")).strip() or None,
                    terminal_episode_cursor=(
                        int(payload["terminal_episode_cursor"])
                        if payload.get("terminal_episode_cursor") is not None
                        else None
                    ),
                    terminal_event_cursor=(
                        int(payload["terminal_event_cursor"])
                        if payload.get("terminal_event_cursor") is not None
                        else None
                    ),
                    result_stats=dict(payload.get("result_stats", {})),
                    error_payload=dict(payload.get("error", {})),
                )
            return

        if event.event_type == BrainEventType.AUTOBIOGRAPHY_ENTRY_UPSERTED:
            scope_type = str(payload.get("scope_type", "")).strip()
            scope_id = str(payload.get("scope_id", "")).strip()
            entry_kind = str(payload.get("entry_kind", "")).strip()
            rendered_summary = str(payload.get("rendered_summary", "")).strip()
            if scope_type and scope_id and entry_kind and rendered_summary:
                raw_last_governance_event_id = payload.get("last_governance_event_id")
                raw_source_presence_scope_key = payload.get("source_presence_scope_key")
                raw_redacted_at = payload.get("redacted_at")
                self.upsert_autobiographical_entry(
                    scope_type=scope_type,
                    scope_id=scope_id,
                    entry_kind=entry_kind,
                    rendered_summary=rendered_summary,
                    content=dict(payload.get("content", {})),
                    salience=float(payload.get("salience", 0.0)),
                    source_episode_ids=list(payload.get("source_episode_ids", [])),
                    source_claim_ids=list(payload.get("source_claim_ids", [])),
                    source_event_ids=list(payload.get("source_event_ids", [])),
                    source_event_id=event.causal_parent_id,
                    valid_from=str(payload.get("valid_from", "")).strip() or None,
                    created_at=str(payload.get("created_at", "")).strip() or None,
                    updated_at=str(payload.get("updated_at", "")).strip() or None,
                    append_only=entry_kind
                    in {
                        "relationship_milestone",
                        "project_arc",
                    },
                    identity_key=(
                        (
                            str(raw_source_presence_scope_key).strip()
                            if isinstance(raw_source_presence_scope_key, str)
                            else ""
                        )
                        or str(payload.get("content", {}).get("project_key", "")).strip()
                        or (
                            scope_id
                            if entry_kind == BrainAutobiographyEntryKind.SCENE_EPISODE.value
                            else ""
                        )
                        or None
                    ),
                    modality=str(payload.get("modality", "")).strip() or None,
                    review_state=str(payload.get("review_state", "")).strip() or None,
                    retention_class=str(payload.get("retention_class", "")).strip() or None,
                    privacy_class=str(payload.get("privacy_class", "")).strip() or None,
                    governance_reason_codes=list(payload.get("governance_reason_codes", [])),
                    last_governance_event_id=(
                        str(raw_last_governance_event_id).strip()
                        if isinstance(raw_last_governance_event_id, str)
                        and str(raw_last_governance_event_id).strip()
                        else None
                    ),
                    source_presence_scope_key=(
                        str(raw_source_presence_scope_key).strip()
                        if isinstance(raw_source_presence_scope_key, str)
                        and str(raw_source_presence_scope_key).strip()
                        else None
                    ),
                    source_scene_entity_ids=list(payload.get("source_scene_entity_ids", [])),
                    source_scene_affordance_ids=list(
                        payload.get("source_scene_affordance_ids", [])
                    ),
                    redacted_at=(
                        str(raw_redacted_at).strip()
                        if isinstance(raw_redacted_at, str) and str(raw_redacted_at).strip()
                        else None
                    ),
                    event_context=None,
                )
            return

        if event.event_type == BrainEventType.AUTOBIOGRAPHY_ENTRY_REVIEW_REQUESTED:
            entry_id = str(payload.get("entry_id", "")).strip()
            if entry_id:
                self.request_autobiographical_entry_review(
                    entry_id,
                    source_event_id=event.event_id,
                    reason_codes=list(payload.get("reason_codes", []))
                    if "reason_codes" in payload
                    else None,
                    summary=str(payload.get("summary", "")).strip() or None,
                    notes=str(payload.get("notes", "")).strip() or None,
                    updated_at=event.ts,
                    event_context=None,
                )
            return

        if event.event_type == BrainEventType.AUTOBIOGRAPHY_ENTRY_RETENTION_RECLASSIFIED:
            entry_id = str(payload.get("entry_id", "")).strip()
            retention_class = str(payload.get("retention_class", "")).strip()
            if entry_id and retention_class:
                self.reclassify_autobiographical_entry_retention(
                    entry_id,
                    retention_class=retention_class,
                    source_event_id=event.event_id,
                    reason_codes=list(payload.get("reason_codes", []))
                    if "reason_codes" in payload
                    else None,
                    summary=str(payload.get("summary", "")).strip() or None,
                    notes=str(payload.get("notes", "")).strip() or None,
                    updated_at=event.ts,
                    event_context=None,
                )
            return

        if event.event_type == BrainEventType.AUTOBIOGRAPHY_ENTRY_REDACTED:
            entry_id = str(payload.get("entry_id", "")).strip()
            redacted_summary = str(payload.get("redacted_summary", "")).strip()
            if entry_id and redacted_summary:
                self.redact_autobiographical_entry(
                    entry_id,
                    redacted_summary=redacted_summary,
                    source_event_id=event.event_id,
                    reason_codes=list(payload.get("reason_codes", []))
                    if "reason_codes" in payload
                    else None,
                    summary=str(payload.get("summary", "")).strip() or None,
                    notes=str(payload.get("notes", "")).strip() or None,
                    updated_at=event.ts,
                    event_context=None,
                )
            return

        if event.event_type == BrainEventType.MEMORY_HEALTH_REPORTED:
            scope_type = str(payload.get("scope_type", "")).strip()
            scope_id = str(payload.get("scope_id", "")).strip()
            cycle_id = str(payload.get("cycle_id", "")).strip()
            if scope_type and scope_id and cycle_id:
                self.record_memory_health_report(
                    scope_type=scope_type,
                    scope_id=scope_id,
                    cycle_id=cycle_id,
                    score=float(payload.get("score", 1.0)),
                    status=str(payload.get("status", "healthy")),
                    findings=list(payload.get("findings", [])),
                    stats=dict(payload.get("stats", {})),
                    artifact_path=str(payload.get("artifact_path", "")).strip() or None,
                    source_event_id=event.causal_parent_id,
                    event_context=None,
                )
            return

    def apply_reflection_event(self, event: BrainEventRecord):
        """Apply one replayed reflection/autobiography event to the local store."""
        self.apply_memory_event(event)

    def apply_executive_event(self, event: BrainEventRecord):
        """Apply one replayed executive event to durable commitment state."""
        if event.event_type not in {
            BrainEventType.GOAL_CREATED,
            BrainEventType.GOAL_UPDATED,
            BrainEventType.GOAL_DEFERRED,
            BrainEventType.GOAL_RESUMED,
            BrainEventType.GOAL_CANCELLED,
            BrainEventType.GOAL_REPAIRED,
            BrainEventType.GOAL_COMPLETED,
            BrainEventType.GOAL_FAILED,
        }:
            return
        payload = event.payload if isinstance(event.payload, dict) else {}
        commitment_payload = payload.get("commitment")
        goal_payload = payload.get("goal")
        if not isinstance(commitment_payload, dict) or not isinstance(goal_payload, dict):
            return
        commitment_id = str(commitment_payload.get("commitment_id", "")).strip()
        scope_type = str(commitment_payload.get("scope_type", "")).strip()
        scope_id = str(commitment_payload.get("scope_id", "")).strip()
        if not commitment_id or not scope_type or not scope_id:
            return
        goal = BrainGoal.from_dict(goal_payload)
        derived_status = {
            BrainEventType.GOAL_CREATED: None,
            BrainEventType.GOAL_UPDATED: None,
            BrainEventType.GOAL_DEFERRED: BrainCommitmentStatus.DEFERRED.value,
            BrainEventType.GOAL_RESUMED: BrainCommitmentStatus.ACTIVE.value,
            BrainEventType.GOAL_CANCELLED: BrainCommitmentStatus.CANCELLED.value,
            BrainEventType.GOAL_REPAIRED: BrainCommitmentStatus.ACTIVE.value,
            BrainEventType.GOAL_COMPLETED: BrainCommitmentStatus.COMPLETED.value,
            BrainEventType.GOAL_FAILED: BrainCommitmentStatus.BLOCKED.value,
        }.get(event.event_type)
        status = derived_status or str(commitment_payload.get("status", "")).strip()
        if not status:
            status = str(goal.details.get("commitment_status", BrainCommitmentStatus.ACTIVE.value))
        user_id = event.user_id
        thread_id = event.thread_id
        self.upsert_executive_commitment(
            commitment_id=commitment_id,
            scope_type=scope_type,
            scope_id=scope_id,
            user_id=user_id,
            thread_id=thread_id,
            title=goal.title,
            goal_family=goal.goal_family,
            intent=goal.intent,
            status=status,
            details={
                **goal.details,
                "summary": goal.title,
            },
            current_goal_id=goal.goal_id or None,
            blocked_reason=goal.blocked_reason,
            wake_conditions=goal.wake_conditions,
            plan_revision=goal.plan_revision,
            resume_count=goal.resume_count,
            source_event_id=event.event_id,
        )

    def set_presence_snapshot(self, *, scope_key: str, snapshot: dict[str, Any]):
        """Persist the latest runtime presence snapshot."""
        now = _utc_now()
        normalized_snapshot_obj = normalize_presence_snapshot(
            BrainPresenceSnapshot.from_dict(snapshot)
        )
        normalized_snapshot = normalized_snapshot_obj.as_dict()
        self._conn.execute(
            """
            INSERT INTO presence_snapshots (scope_key, snapshot_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(scope_key) DO UPDATE SET
                snapshot_json = excluded.snapshot_json,
                updated_at = excluded.updated_at
            """,
            (scope_key, json.dumps(normalized_snapshot, ensure_ascii=False, sort_keys=True), now),
        )
        self._upsert_projection(
            projection_name=BODY_STATE_PROJECTION,
            scope_key=scope_key,
            projection=normalized_snapshot,
            source_event_id=None,
            updated_at=now,
            commit=False,
        )
        self._upsert_projection(
            projection_name=SCENE_STATE_PROJECTION,
            scope_key=scope_key,
            projection=BrainSceneStateProjection(
                camera_connected=normalized_snapshot_obj.vision_connected,
                person_present="uncertain",
                scene_change_state="unknown",
                source="compatibility",
                updated_at=now,
            ).as_dict(),
            source_event_id=None,
            updated_at=now,
            commit=False,
        )
        self._upsert_projection(
            projection_name=ENGAGEMENT_STATE_PROJECTION,
            scope_key=scope_key,
            projection=BrainEngagementStateProjection(
                engagement_state="unknown",
                attention_to_camera="unknown",
                user_present=bool(
                    normalized_snapshot_obj.vision_connected
                    and not normalized_snapshot_obj.camera_disconnected
                ),
                source="compatibility",
                updated_at=now,
            ).as_dict(),
            source_event_id=None,
            updated_at=now,
            commit=False,
        )
        self._upsert_projection(
            projection_name=RELATIONSHIP_STATE_PROJECTION,
            scope_key=scope_key,
            projection=BrainRelationshipStateProjection(
                user_present=bool(
                    normalized_snapshot_obj.vision_connected
                    and not normalized_snapshot_obj.camera_disconnected
                ),
                source="compatibility",
                updated_at=now,
            ).as_dict(),
            source_event_id=None,
            updated_at=now,
            commit=False,
        )
        self._conn.commit()

    def get_presence_snapshot(self, *, scope_key: str) -> dict[str, Any] | None:
        """Return one presence snapshot."""
        row = self._conn.execute(
            "SELECT snapshot_json FROM presence_snapshots WHERE scope_key = ?",
            (scope_key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(str(row["snapshot_json"]))

    def append_brain_event(
        self,
        *,
        event_type: str,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one typed brain event and update projections."""
        now = ts or _utc_now()
        try:
            normalized_confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            normalized_confidence = 0.0
        cursor = self._conn.execute(
            """
            INSERT INTO brain_events (
                event_id, event_type, ts, agent_id, user_id, session_id, thread_id, source,
                correlation_id, causal_parent_id, confidence, payload_json, tags_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid4().hex,
                event_type,
                now,
                agent_id,
                user_id,
                session_id,
                thread_id,
                source,
                correlation_id,
                causal_parent_id,
                normalized_confidence,
                dumps_bounded_json(
                    payload or {},
                    max_chars=MAX_BRAIN_EVENT_PAYLOAD_JSON_CHARS,
                    overflow_kind=_OVERSIZED_BRAIN_PAYLOAD_MARKER,
                ),
                json.dumps(tags or [], ensure_ascii=False, sort_keys=True),
            ),
        )
        event = self._brain_event_from_row(
            self._conn.execute(
                "SELECT * FROM brain_events WHERE id = ?", (int(cursor.lastrowid),)
            ).fetchone()
        )
        self._apply_event_to_projections(event, commit=False)
        self._maybe_append_scene_episode_autobiography(event)
        self._maybe_refresh_predictive_world_model(event)
        self._maybe_compare_action_outcome_against_rehearsal(event)
        self._conn.commit()
        return event

    def _maybe_append_scene_episode_autobiography(self, event: BrainEventRecord):
        """Refresh bounded scene autobiography only for new scene/perception events."""
        if event.event_type not in {
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.SCENE_CHANGED,
            BrainEventType.ENGAGEMENT_CHANGED,
            BrainEventType.ATTENTION_CHANGED,
        }:
            return
        presence_scope_key = _event_presence_scope_key(event) or "local:presence"
        self.refresh_scene_episode_autobiography(
            user_id=event.user_id,
            thread_id=event.thread_id,
            session_id=event.session_id,
            agent_id=event.agent_id,
            presence_scope_key=presence_scope_key,
            source_event_id=event.event_id,
            updated_at=event.ts,
            event_context={
                "agent_id": event.agent_id,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "thread_id": event.thread_id,
                "source": "memory_v2",
                "correlation_id": event.correlation_id,
            },
        )

    def _maybe_refresh_predictive_world_model(self, event: BrainEventRecord):
        """Append explicit predictive lifecycle events after one relevant source event lands."""
        if is_predictive_event_type(event.event_type):
            return
        if not should_refresh_predictive_world_model(event.event_type):
            return
        predictive_projection = self.get_predictive_world_model_projection(
            scope_key=event.thread_id
        )
        presence_scope_key = (
            _event_presence_scope_key(event)
            or _optional_text(predictive_projection.presence_scope_key)
            or "local:presence"
        )
        scene_world_state = self.build_scene_world_state_projection(
            user_id=event.user_id,
            thread_id=event.thread_id,
            reference_ts=event.ts,
            agent_id=event.agent_id,
            presence_scope_key=presence_scope_key,
        )
        active_situation_model = self.build_active_situation_model_projection(
            user_id=event.user_id,
            thread_id=event.thread_id,
            reference_ts=event.ts,
            agent_id=event.agent_id,
            presence_scope_key=presence_scope_key,
        )
        private_working_memory = self.build_private_working_memory_projection(
            user_id=event.user_id,
            thread_id=event.thread_id,
            reference_ts=event.ts,
            agent_id=event.agent_id,
            presence_scope_key=presence_scope_key,
        )
        procedural_skills = self.build_procedural_skill_projection(
            scope_type="thread",
            scope_id=event.thread_id,
        )
        scene_episodes = tuple(
            self.autobiographical_entries(
                scope_type="presence",
                scope_id=presence_scope_key,
                entry_kinds=(BrainAutobiographyEntryKind.SCENE_EPISODE.value,),
                statuses=("current", "superseded"),
                modalities=("scene_world",),
                limit=8,
            )
        )
        commitment_projection = self.get_session_commitment_projection(
            agent_id=event.agent_id,
            user_id=event.user_id,
            thread_id=event.thread_id,
        )
        meaningful_scene_context = bool(scene_episodes) or any(
            record.source_event_ids
            for record in (*scene_world_state.entities, *scene_world_state.affordances)
        )
        actionable_prediction_context = any(
            record.record_kind
            in {"goal_state", "commitment_state", "plan_state", "procedural_state"}
            for record in active_situation_model.records
        )
        if not predictive_projection.active_predictions and not (
            meaningful_scene_context and actionable_prediction_context
        ):
            return
        active_predictions = list(predictive_projection.active_predictions)
        confirmed_subject_keys: set[tuple[str, str]] = set()
        for prediction in active_predictions:
            valid_to = _optional_text(prediction.valid_to)
            if valid_to and (
                _parse_ts(valid_to) is not None
                and (_parse_ts(event.ts) or datetime.now(UTC))
                > (_parse_ts(valid_to) or datetime.now(UTC))
            ):
                self._append_prediction_event(
                    event_type=BrainEventType.PREDICTION_EXPIRED,
                    prediction=build_prediction_resolution(
                        prediction=prediction,
                        trigger_event=event,
                        resolution_kind=BrainPredictionResolutionKind.EXPIRED.value,
                        resolution_summary="Prediction window expired before confirmation.",
                    ),
                    trigger_event=event,
                    presence_scope_key=presence_scope_key,
                )
                continue
            resolution = resolve_prediction_against_state(
                prediction=prediction,
                trigger_event=event,
                scene_world_state=scene_world_state,
                active_situation_model=active_situation_model,
            )
            if resolution is None:
                continue
            if resolution:
                confirmed_subject_keys.add((prediction.prediction_kind, prediction.subject_id))
                self._append_prediction_event(
                    event_type=BrainEventType.PREDICTION_CONFIRMED,
                    prediction=build_prediction_resolution(
                        prediction=prediction,
                        trigger_event=event,
                        resolution_kind=BrainPredictionResolutionKind.CONFIRMED.value,
                        resolution_summary="Observed state matched the short-horizon prediction.",
                    ),
                    trigger_event=event,
                    presence_scope_key=presence_scope_key,
                )
            else:
                self._append_prediction_event(
                    event_type=BrainEventType.PREDICTION_INVALIDATED,
                    prediction=build_prediction_resolution(
                        prediction=prediction,
                        trigger_event=event,
                        resolution_kind=BrainPredictionResolutionKind.INVALIDATED.value,
                        resolution_summary="Observed state contradicted the short-horizon prediction.",
                    ),
                    trigger_event=event,
                    presence_scope_key=presence_scope_key,
                )

        refreshed_projection = self.get_predictive_world_model_projection(scope_key=event.thread_id)
        active_by_subject = {
            (record.prediction_kind, record.subject_id): record
            for record in refreshed_projection.active_predictions
        }
        adapter_response = self._world_model_adapter.propose_predictions(
            WorldModelAdapterRequest(
                scope_key=event.thread_id,
                presence_scope_key=presence_scope_key,
                reference_ts=event.ts,
                scene_world_state=scene_world_state,
                active_situation_model=active_situation_model,
                private_working_memory=private_working_memory,
                procedural_skills=procedural_skills,
                scene_episodes=scene_episodes,
                commitment_projection=commitment_projection,
            )
        )
        desired_predictions = build_prediction_records_from_proposals(
            scope_key=event.thread_id,
            presence_scope_key=presence_scope_key,
            reference_ts=event.ts,
            proposals=adapter_response.proposals,
        )
        for prediction in desired_predictions:
            subject_key = (prediction.prediction_kind, prediction.subject_id)
            if subject_key in confirmed_subject_keys:
                continue
            existing = active_by_subject.get(subject_key)
            if existing is not None:
                if existing.prediction_id == prediction.prediction_id:
                    continue
                self._append_prediction_event(
                    event_type=BrainEventType.PREDICTION_INVALIDATED,
                    prediction=build_prediction_resolution(
                        prediction=existing,
                        trigger_event=event,
                        resolution_kind=BrainPredictionResolutionKind.INVALIDATED.value,
                        resolution_summary="Prediction was superseded by a newer short-horizon estimate.",
                    ),
                    trigger_event=event,
                    presence_scope_key=presence_scope_key,
                )
            self._append_prediction_event(
                event_type=prediction_generation_event_type(prediction.prediction_kind),
                prediction=prediction,
                trigger_event=event,
                presence_scope_key=presence_scope_key,
            )

    def _maybe_compare_action_outcome_against_rehearsal(self, event: BrainEventRecord):
        """Append one derived rehearsal calibration event after a robot action outcome lands."""
        if event.event_type != BrainEventType.ROBOT_ACTION_OUTCOME:
            return
        from blink.brain.counterfactuals import build_outcome_comparison

        payload = _event_payload_dict(event)
        projection = self.get_counterfactual_rehearsal_projection(scope_key=event.thread_id)
        if not projection.recent_rehearsals:
            return
        if any(
            record.observed_event_id == event.event_id for record in projection.recent_comparisons
        ):
            return
        rehearsal_id = _optional_text(payload.get("rehearsal_id"))
        goal_id = _optional_text(payload.get("goal_id"))
        step_index = payload.get("step_index")
        resolved_step_index = int(step_index) if step_index is not None else None
        action_id = _optional_text(payload.get("action_id"))
        matched_result = next(
            (
                record
                for record in projection.recent_rehearsals
                if not record.skipped
                and rehearsal_id is not None
                and record.rehearsal_id == rehearsal_id
            ),
            None,
        )
        if matched_result is None:
            matched_result = next(
                (
                    record
                    for record in projection.recent_rehearsals
                    if not record.skipped
                    and action_id is not None
                    and record.candidate_action_id == action_id
                    and (goal_id is None or record.goal_id == goal_id)
                    and (
                        resolved_step_index is None
                        or int(record.step_index) == int(resolved_step_index)
                    )
                ),
                None,
            )
        if matched_result is None:
            return
        if bool(payload.get("preview_only")):
            observed_outcome_kind = BrainObservedActionOutcomeKind.PREVIEW_ONLY.value
        elif bool(payload.get("accepted")):
            observed_outcome_kind = BrainObservedActionOutcomeKind.SUCCESS.value
        else:
            observed_outcome_kind = BrainObservedActionOutcomeKind.FAILURE.value
        comparison = build_outcome_comparison(
            rehearsal_result=matched_result,
            trigger_event=event,
            observed_outcome_kind=observed_outcome_kind,
        )
        presence_scope_key = (
            _event_presence_scope_key(event)
            or _optional_text(projection.presence_scope_key)
            or "local:presence"
        )
        self.append_brain_event(
            event_type=BrainEventType.ACTION_OUTCOME_COMPARED,
            agent_id=event.agent_id,
            user_id=event.user_id,
            session_id=event.session_id,
            thread_id=event.thread_id,
            source="counterfactuals",
            payload={
                "comparison": comparison.as_dict(),
                "comparison_id": comparison.comparison_id,
                "rehearsal_id": comparison.rehearsal_id,
                "goal_id": comparison.goal_id,
                "commitment_id": comparison.commitment_id,
                "plan_proposal_id": comparison.plan_proposal_id,
                "step_index": comparison.step_index,
                "candidate_action_id": comparison.candidate_action_id,
                "observed_outcome_kind": comparison.observed_outcome_kind,
                "calibration_bucket": comparison.calibration_bucket,
                "presence_scope_key": presence_scope_key,
            },
            correlation_id=comparison.goal_id or comparison.rehearsal_id,
            causal_parent_id=event.event_id,
            confidence=matched_result.predicted_success_probability,
            ts=event.ts,
        )

    def _append_prediction_event(
        self,
        *,
        event_type: str,
        prediction: BrainPredictionRecord,
        trigger_event: BrainEventRecord,
        presence_scope_key: str,
    ) -> BrainEventRecord:
        """Append one explicit predictive lifecycle event."""
        return self.append_brain_event(
            event_type=event_type,
            agent_id=trigger_event.agent_id,
            user_id=trigger_event.user_id,
            session_id=trigger_event.session_id,
            thread_id=trigger_event.thread_id,
            source="world_model",
            payload={
                "prediction": prediction.as_dict(),
                "prediction_id": prediction.prediction_id,
                "prediction_kind": prediction.prediction_kind,
                "subject_kind": prediction.subject_kind,
                "subject_id": prediction.subject_id,
                "presence_scope_key": presence_scope_key,
            },
            correlation_id=trigger_event.correlation_id,
            causal_parent_id=trigger_event.event_id,
            confidence=max(prediction.confidence, 0.0),
            ts=trigger_event.ts,
        )

    def recent_brain_events(
        self,
        *,
        user_id: str,
        thread_id: str,
        limit: int = 12,
        event_types: tuple[str, ...] | None = None,
    ) -> list[BrainEventRecord]:
        """Return recent typed brain events for one user thread."""
        clauses = ["user_id = ?", "thread_id = ?"]
        params: list[Any] = [user_id, thread_id]
        if event_types:
            clauses.append(f"event_type IN ({','.join('?' for _ in event_types)})")
            params.extend(event_types)
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT * FROM brain_events
            WHERE {" AND ".join(clauses)}
            ORDER BY id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [self._brain_event_from_row(row) for row in rows]

    def append_memory_use_trace(
        self,
        *,
        trace: BrainMemoryUseTrace,
        session_id: str,
        source: str = "memory_use_trace",
        causal_parent_id: str | None = None,
        ts: str | None = None,
    ) -> BrainMemoryUseTrace:
        """Append one compact user-facing memory-use trace event."""
        created_at = ts or _utc_now()
        stamped = stamp_memory_use_trace(trace, created_at=created_at)
        self.append_brain_event(
            event_type=BrainEventType.MEMORY_USE_TRACED,
            agent_id=stamped.agent_id,
            user_id=stamped.user_id,
            session_id=session_id,
            thread_id=stamped.thread_id,
            source=source,
            causal_parent_id=causal_parent_id,
            payload={"trace": stamped.as_dict()},
            ts=created_at,
        )
        return stamped

    def recent_memory_use_traces(
        self,
        *,
        user_id: str,
        thread_id: str,
        limit: int = 8,
    ) -> tuple[BrainMemoryUseTrace, ...]:
        """Return recent memory-use traces for one user thread, newest first."""
        traces: list[BrainMemoryUseTrace] = []
        for event in self.recent_brain_events(
            user_id=user_id,
            thread_id=thread_id,
            limit=limit,
            event_types=(BrainEventType.MEMORY_USE_TRACED,),
        ):
            payload = event.payload if isinstance(event.payload, dict) else {}
            trace_payload = (
                payload.get("trace") if isinstance(payload.get("trace"), dict) else payload
            )
            if not isinstance(trace_payload, dict):
                continue
            try:
                trace = BrainMemoryUseTrace.from_dict(trace_payload)
            except (TypeError, ValueError):
                continue
            if not trace.created_at:
                trace = stamp_memory_use_trace(trace, created_at=event.ts)
            traces.append(trace)
        return tuple(traces)

    def latest_memory_use_trace(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> BrainMemoryUseTrace | None:
        """Return the latest memory-use trace for one user thread."""
        traces = self.recent_memory_use_traces(user_id=user_id, thread_id=thread_id, limit=1)
        return traces[0] if traces else None

    def append_memory_continuity_trace(
        self,
        *,
        trace: MemoryContinuityTrace,
        session_id: str,
        source: str = "memory_continuity_trace",
        causal_parent_id: str | None = None,
        ts: str | None = None,
    ) -> MemoryContinuityTrace:
        """Append one public-safe memory-continuity trace event."""
        created_at = ts or _utc_now()
        stamped = stamp_memory_continuity_trace(
            trace,
            created_at=created_at,
            session_id=session_id,
        )
        self.append_brain_event(
            event_type=BrainEventType.MEMORY_CONTINUITY_TRACED,
            agent_id=stamped.agent_id,
            user_id=stamped.user_id,
            session_id=session_id,
            thread_id=stamped.thread_id,
            source=source,
            causal_parent_id=causal_parent_id,
            payload={"trace": stamped.as_dict()},
            ts=created_at,
        )
        return stamped

    def recent_memory_continuity_traces(
        self,
        *,
        user_id: str,
        thread_id: str,
        limit: int = 8,
    ) -> tuple[MemoryContinuityTrace, ...]:
        """Return recent memory-continuity traces for one user thread, newest first."""
        traces: list[MemoryContinuityTrace] = []
        for event in self.recent_brain_events(
            user_id=user_id,
            thread_id=thread_id,
            limit=limit,
            event_types=(BrainEventType.MEMORY_CONTINUITY_TRACED,),
        ):
            payload = event.payload if isinstance(event.payload, dict) else {}
            trace_payload = (
                payload.get("trace") if isinstance(payload.get("trace"), dict) else payload
            )
            if not isinstance(trace_payload, dict):
                continue
            try:
                trace = MemoryContinuityTrace.from_dict(trace_payload)
            except (TypeError, ValueError):
                continue
            if not trace.created_at:
                trace = stamp_memory_continuity_trace(
                    trace,
                    created_at=event.ts,
                    session_id=event.session_id,
                )
            traces.append(trace)
        return tuple(traces)

    def latest_memory_continuity_trace(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> MemoryContinuityTrace | None:
        """Return the latest memory-continuity trace for one user thread."""
        traces = self.recent_memory_continuity_traces(
            user_id=user_id,
            thread_id=thread_id,
            limit=1,
        )
        return traces[0] if traces else None

    def append_discourse_episode(
        self,
        *,
        episode: DiscourseEpisode,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str = "discourse_episode_v3",
        causal_parent_id: str | None = None,
        ts: str | None = None,
    ) -> DiscourseEpisode:
        """Append one public-safe discourse episode event."""
        created_at = ts or _utc_now()
        self.append_brain_event(
            event_type=BrainEventType.MEMORY_DISCOURSE_EPISODE_DERIVED,
            agent_id=str(agent_id),
            user_id=str(user_id),
            session_id=str(session_id),
            thread_id=str(thread_id),
            source=source,
            causal_parent_id=causal_parent_id,
            payload={"discourse_episode": episode.as_dict()},
            ts=created_at,
        )
        return episode

    def recent_discourse_episodes(
        self,
        *,
        user_id: str,
        thread_id: str,
        limit: int = 8,
    ) -> tuple[DiscourseEpisode, ...]:
        """Return recent discourse episodes for one user thread, newest first."""
        episodes: list[DiscourseEpisode] = []
        for event in self.recent_brain_events(
            user_id=user_id,
            thread_id=thread_id,
            limit=limit,
            event_types=(BrainEventType.MEMORY_DISCOURSE_EPISODE_DERIVED,),
        ):
            payload = event.payload if isinstance(event.payload, dict) else {}
            episode_payload = (
                payload.get("discourse_episode")
                if isinstance(payload.get("discourse_episode"), dict)
                else payload
            )
            if not isinstance(episode_payload, dict):
                continue
            try:
                episodes.append(DiscourseEpisode.from_dict(episode_payload))
            except (TypeError, ValueError):
                continue
        return tuple(episodes)

    def latest_discourse_episode(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> DiscourseEpisode | None:
        """Return the latest discourse episode for one user thread."""
        episodes = self.recent_discourse_episodes(user_id=user_id, thread_id=thread_id, limit=1)
        return episodes[0] if episodes else None

    def latest_brain_event(
        self,
        *,
        user_id: str,
        thread_id: str,
        event_types: tuple[str, ...] | None = None,
    ) -> BrainEventRecord | None:
        """Return the latest typed brain event for one thread and optional type filter."""
        clauses = ["user_id = ?", "thread_id = ?"]
        params: list[Any] = [user_id, thread_id]
        if event_types:
            clauses.append(f"event_type IN ({','.join('?' for _ in event_types)})")
            params.extend(event_types)
        row = self._conn.execute(
            f"""
            SELECT * FROM brain_events
            WHERE {" AND ".join(clauses)}
            ORDER BY id DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()
        return self._brain_event_from_row(row) if row is not None else None

    def brain_events_since(
        self,
        *,
        user_id: str,
        thread_id: str,
        after_id: int = 0,
        limit: int = 64,
    ) -> list[BrainEventRecord]:
        """Return typed brain events newer than `after_id` in ascending order."""
        rows = self._conn.execute(
            """
            SELECT * FROM brain_events
            WHERE user_id = ? AND thread_id = ? AND id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (user_id, thread_id, after_id, limit),
        ).fetchall()
        return [self._brain_event_from_row(row) for row in rows]

    def import_brain_event(self, event: BrainEventRecord):
        """Import one existing brain event into this store for deterministic replay."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO brain_events (
                event_id, event_type, ts, agent_id, user_id, session_id, thread_id, source,
                correlation_id, causal_parent_id, confidence, payload_json, tags_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.event_type,
                event.ts,
                event.agent_id,
                event.user_id,
                event.session_id,
                event.thread_id,
                event.source,
                event.correlation_id,
                event.causal_parent_id,
                event.confidence,
                event.payload_json,
                event.tags_json,
            ),
        )
        self._apply_event_to_projections(event, commit=False)
        self._conn.commit()

    def rebuild_brain_projections(self):
        """Rebuild projection tables by replaying the append-only event log."""
        self._conn.execute("DELETE FROM brain_projections")
        rows = self._conn.execute("SELECT * FROM brain_events ORDER BY id ASC").fetchall()
        for row in rows:
            self._apply_event_to_projections(self._brain_event_from_row(row), commit=False)
        claim_scopes = self._conn.execute(
            """
            SELECT DISTINCT scope_type, scope_id
            FROM claims
            WHERE scope_type IS NOT NULL AND scope_id IS NOT NULL
            """
        ).fetchall()
        for row in claim_scopes:
            self.build_claim_governance_projection(
                scope_type=str(row["scope_type"]),
                scope_id=str(row["scope_id"]),
                commit=False,
            )
        self._conn.commit()

    def get_body_state_projection(self, *, scope_key: str) -> BrainPresenceSnapshot:
        """Return the current body-state projection for one runtime scope."""
        projection = self._get_projection_dict(
            projection_name=BODY_STATE_PROJECTION,
            scope_key=scope_key,
        )
        if projection is not None:
            return normalize_presence_snapshot(BrainPresenceSnapshot.from_dict(projection))
        return normalize_presence_snapshot(
            BrainPresenceSnapshot.from_dict(self.get_presence_snapshot(scope_key=scope_key))
        )

    def get_scene_state_projection(self, *, scope_key: str) -> BrainSceneStateProjection:
        """Return the current symbolic scene projection for one runtime scope."""
        return BrainSceneStateProjection.from_dict(
            self._get_projection_dict(
                projection_name=SCENE_STATE_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_scene_world_state_projection(
        self,
        *,
        scope_key: str,
    ) -> BrainSceneWorldProjection:
        """Return the current symbolic scene world-state projection for one runtime scope."""
        return BrainSceneWorldProjection.from_dict(
            self._get_projection_dict(
                projection_name=SCENE_WORLD_STATE_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_engagement_state_projection(self, *, scope_key: str) -> BrainEngagementStateProjection:
        """Return the current symbolic engagement projection for one runtime scope."""
        return BrainEngagementStateProjection.from_dict(
            self._get_projection_dict(
                projection_name=ENGAGEMENT_STATE_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_relationship_state_projection(
        self,
        *,
        scope_key: str,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> BrainRelationshipStateProjection:
        """Return the current projected relationship continuity surface."""
        projection_payload = self._get_projection_dict(
            projection_name=RELATIONSHIP_STATE_PROJECTION,
            scope_key=scope_key,
        )
        projection = BrainRelationshipStateProjection.from_dict(projection_payload)
        if user_id is None:
            return projection

        relationship_scope_id = self._relationship_scope_id(
            agent_id=_DEFAULT_RELATIONSHIP_AGENT_ID,
            user_id=user_id,
        )
        relationship_arc = self.latest_autobiographical_entry(
            scope_type="relationship",
            scope_id=relationship_scope_id,
            entry_kind="relationship_arc",
        )
        if relationship_arc is not None:
            projection.continuity_summary = relationship_arc.rendered_summary

        active_commitments_block = self.get_current_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.ACTIVE_COMMITMENTS.value,
            scope_type="relationship",
            scope_id=relationship_scope_id,
        )
        if active_commitments_block is not None:
            commitments = list(active_commitments_block.content.get("commitments", []))
            projection.open_commitments = [
                str(item.get("title") or item.get("summary") or "").strip()
                for item in commitments
                if str(item.get("title") or item.get("summary") or "").strip()
            ][:4]

        commitments = self.narrative_memories(
            user_id=user_id,
            thread_id=thread_id,
            kinds=("commitment",),
            statuses=("open", "active"),
            limit=4,
            include_stale=False,
        )
        continuity = self.narrative_memories(
            user_id=user_id,
            thread_id=thread_id,
            kinds=("session_summary", "daily_summary"),
            statuses=("active",),
            limit=2,
            include_stale=False,
        )
        style_hint_records = self.semantic_memories(
            user_id=user_id,
            namespaces=("interaction.style", "interaction.preference"),
            limit=4,
            include_stale=False,
        )
        if not projection.open_commitments:
            projection.open_commitments = [record.summary for record in commitments]
        if projection.continuity_summary is None:
            projection.continuity_summary = (
                " / ".join(record.summary for record in continuity)
                if continuity
                else projection.continuity_summary
            )
        updated_at_candidates = [
            _parse_ts(relationship_arc.updated_at) if relationship_arc is not None else None,
            _parse_ts(active_commitments_block.updated_at)
            if active_commitments_block is not None
            else None,
            *(_parse_ts(record.updated_at) for record in commitments),
            *(_parse_ts(record.updated_at) for record in continuity),
            *(_parse_ts(record.updated_at) for record in style_hint_records),
        ]
        relationship_style_state = None
        relationship_style_block = self.get_current_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
            scope_type="relationship",
            scope_id=relationship_scope_id,
        )
        if relationship_style_block is not None:
            updated_at_candidates.append(_parse_ts(relationship_style_block.updated_at))
        if relationship_style_block is not None:
            try:
                relationship_style_state = RelationshipStyleStateSpec.model_validate(
                    relationship_style_block.content
                )
            except ValidationError:
                relationship_style_state = None
        if relationship_style_state is None:
            try:
                relationship_style_state = compile_relationship_style_state(
                    self.get_agent_blocks(),
                    store=self,
                    user_id=user_id,
                    thread_id=thread_id or f"relationship:{user_id}",
                    agent_id=_DEFAULT_RELATIONSHIP_AGENT_ID,
                )
            except (ValidationError, ValueError):
                relationship_style_state = None

        teaching_profile_state = None
        teaching_profile_block = self.get_current_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
            scope_type="relationship",
            scope_id=relationship_scope_id,
        )
        if teaching_profile_block is not None:
            updated_at_candidates.append(_parse_ts(teaching_profile_block.updated_at))
        if teaching_profile_block is not None:
            try:
                teaching_profile_state = TeachingProfileStateSpec.model_validate(
                    teaching_profile_block.content
                )
            except ValidationError:
                teaching_profile_state = None
        if teaching_profile_state is None:
            try:
                teaching_profile_state = compile_teaching_profile_state(
                    self.get_agent_blocks(),
                    store=self,
                    user_id=user_id,
                    thread_id=thread_id or f"relationship:{user_id}",
                    agent_id=_DEFAULT_RELATIONSHIP_AGENT_ID,
                )
            except (ValidationError, ValueError):
                teaching_profile_state = None

        projection.interaction_style_hints = [
            record.rendered_text for record in style_hint_records if record.rendered_text
        ]
        if relationship_style_state is not None:
            if relationship_style_state.interaction_style_hints:
                projection.interaction_style_hints = list(
                    relationship_style_state.interaction_style_hints
                )
            projection.collaboration_style = relationship_style_state.collaboration_style
            projection.boundaries = list(relationship_style_state.boundaries)
            projection.known_misfires = list(relationship_style_state.known_misfires)
        if teaching_profile_state is not None:
            projection.preferred_teaching_modes = list(teaching_profile_state.preferred_modes[:4])
            projection.analogy_domains = list(teaching_profile_state.analogy_domains[:4])
        latest_updated_at = max(
            (candidate for candidate in updated_at_candidates if candidate is not None),
            default=_parse_ts(projection.updated_at) if projection_payload is not None else None,
        )
        if latest_updated_at is not None:
            projection.updated_at = latest_updated_at.isoformat()
        return projection

    def get_working_context_projection(self, *, scope_key: str) -> BrainWorkingContextProjection:
        """Return the current working-context projection for one thread."""
        return BrainWorkingContextProjection.from_dict(
            self._get_projection_dict(
                projection_name=WORKING_CONTEXT_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_private_working_memory_projection(
        self,
        *,
        scope_key: str,
    ) -> BrainPrivateWorkingMemoryProjection:
        """Return the current private working-memory projection for one thread."""
        return BrainPrivateWorkingMemoryProjection.from_dict(
            self._get_projection_dict(
                projection_name=PRIVATE_WORKING_MEMORY_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_active_situation_model_projection(
        self,
        *,
        scope_key: str,
    ) -> BrainActiveSituationProjection:
        """Return the current active situation-model projection for one thread."""
        return BrainActiveSituationProjection.from_dict(
            self._get_projection_dict(
                projection_name=ACTIVE_SITUATION_MODEL_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_predictive_world_model_projection(
        self,
        *,
        scope_key: str,
    ) -> BrainPredictiveWorldModelProjection:
        """Return the current predictive world-model projection for one thread."""
        projection = self._get_projection_dict(
            projection_name=PREDICTIVE_WORLD_MODEL_PROJECTION,
            scope_key=scope_key,
        )
        if projection is None:
            return BrainPredictiveWorldModelProjection(
                scope_key=scope_key,
                presence_scope_key="local:presence",
                calibration_summary=BrainPredictionCalibrationSummary(),
            )
        return BrainPredictiveWorldModelProjection.from_dict(projection)

    def get_counterfactual_rehearsal_projection(
        self,
        *,
        scope_key: str,
    ) -> BrainCounterfactualRehearsalProjection:
        """Return the current counterfactual rehearsal projection for one thread."""
        projection = self._get_projection_dict(
            projection_name=COUNTERFACTUAL_REHEARSAL_PROJECTION,
            scope_key=scope_key,
        )
        if projection is None:
            return BrainCounterfactualRehearsalProjection(
                scope_key=scope_key,
                presence_scope_key="local:presence",
                calibration_summary=BrainCounterfactualCalibrationSummary(),
                updated_at="",
            )
        return BrainCounterfactualRehearsalProjection.from_dict(projection)

    def get_embodied_executive_projection(
        self,
        *,
        scope_key: str,
    ) -> BrainEmbodiedExecutiveProjection:
        """Return the current embodied executive projection for one thread."""
        projection = self._get_projection_dict(
            projection_name=EMBODIED_EXECUTIVE_PROJECTION,
            scope_key=scope_key,
        )
        if projection is None:
            return BrainEmbodiedExecutiveProjection(
                scope_key=scope_key,
                presence_scope_key="local:presence",
            )
        return BrainEmbodiedExecutiveProjection.from_dict(projection)

    def get_practice_director_projection(
        self,
        *,
        scope_key: str,
    ) -> BrainPracticeDirectorProjection:
        """Return the current practice-director projection for one thread."""
        projection = self._get_projection_dict(
            projection_name=PRACTICE_DIRECTOR_PROJECTION,
            scope_key=scope_key,
        )
        if projection is None:
            return BrainPracticeDirectorProjection(
                scope_key=scope_key,
                presence_scope_key="local:presence",
            )
        return BrainPracticeDirectorProjection.from_dict(projection)

    def get_skill_evidence_projection(
        self,
        *,
        scope_key: str,
    ) -> BrainSkillEvidenceLedger:
        """Return the current skill-evidence projection for one thread."""
        projection = self._get_projection_dict(
            projection_name=SKILL_EVIDENCE_PROJECTION,
            scope_key=scope_key,
        )
        if projection is None:
            return BrainSkillEvidenceLedger(
                scope_type="thread",
                scope_id=scope_key,
            )
        return BrainSkillEvidenceLedger.from_dict(projection)

    def get_skill_governance_projection(
        self,
        *,
        scope_key: str,
    ) -> BrainSkillGovernanceProjection:
        """Return the current skill-governance projection for one thread."""
        projection = self._get_projection_dict(
            projection_name=SKILL_GOVERNANCE_PROJECTION,
            scope_key=scope_key,
        )
        if projection is None:
            return BrainSkillGovernanceProjection(
                scope_type="thread",
                scope_id=scope_key,
            )
        return BrainSkillGovernanceProjection.from_dict(projection)

    def get_adapter_governance_projection(
        self,
        *,
        scope_key: str,
    ) -> BrainAdapterGovernanceProjection:
        """Return the current adapter-governance projection for one thread."""
        projection = self._get_projection_dict(
            projection_name=ADAPTER_GOVERNANCE_PROJECTION,
            scope_key=scope_key,
        )
        if projection is None:
            return BrainAdapterGovernanceProjection(scope_key=scope_key)
        return BrainAdapterGovernanceProjection.from_dict(projection)

    def build_private_working_memory_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        reference_ts: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
    ) -> BrainPrivateWorkingMemoryProjection:
        """Build the derived bounded private working-memory projection for one thread."""
        recent_events = self.recent_brain_events(
            user_id=user_id,
            thread_id=thread_id,
            limit=recent_event_limit,
        )
        latest_event = recent_events[0] if recent_events else None
        resolved_agent_id = (
            agent_id
            or str(getattr(latest_event, "agent_id", "")).strip()
            or _DEFAULT_RELATIONSHIP_AGENT_ID
        )
        resolved_presence_scope_key = presence_scope_key or next(
            (
                scope_key
                for event in recent_events
                if (scope_key := _event_presence_scope_key(event))
            ),
            "local:presence",
        )
        working_context = self.get_working_context_projection(scope_key=thread_id)
        agenda = self.get_agenda_projection(scope_key=thread_id, user_id=user_id)
        commitment_projection = self.get_session_commitment_projection(
            agent_id=resolved_agent_id,
            user_id=user_id,
            thread_id=thread_id,
        )
        core_blocks = self._list_continuity_core_blocks(
            user_id=user_id,
            agent_id=resolved_agent_id,
        )
        scene_world_state = self.build_scene_world_state_projection(
            user_id=user_id,
            thread_id=thread_id,
            reference_ts=reference_ts,
            recent_event_limit=recent_event_limit,
            agent_id=resolved_agent_id,
            presence_scope_key=resolved_presence_scope_key,
        )
        procedural_skills = self.build_procedural_skill_projection(
            scope_type="thread",
            scope_id=thread_id,
        )
        scene = self.get_scene_state_projection(scope_key=resolved_presence_scope_key)
        engagement = self.get_engagement_state_projection(scope_key=resolved_presence_scope_key)
        body = self.get_body_state_projection(scope_key=resolved_presence_scope_key)
        continuity_graph = self.build_continuity_graph(
            agent_id=resolved_agent_id,
            user_id=user_id,
            thread_id=thread_id,
            scope_type="user",
            scope_id=user_id,
            reference_ts=reference_ts,
            core_blocks=core_blocks,
            commitment_projection=commitment_projection,
            agenda=agenda,
            procedural_skills=procedural_skills,
            scene_world_state=scene_world_state,
            presence_scope_key=resolved_presence_scope_key,
            recent_event_limit=recent_event_limit,
            current_claim_limit=32,
            historical_claim_limit=16,
            autobiography_limit=24,
        )
        continuity_dossiers = self.build_continuity_dossiers(
            agent_id=resolved_agent_id,
            user_id=user_id,
            thread_id=thread_id,
            scope_type="user",
            scope_id=user_id,
            continuity_graph=continuity_graph,
            reference_ts=reference_ts,
            core_blocks=core_blocks,
            commitment_projection=commitment_projection,
            agenda=agenda,
            procedural_skills=procedural_skills,
            scene_world_state=scene_world_state,
            presence_scope_key=resolved_presence_scope_key,
            recent_event_limit=recent_event_limit,
            current_claim_limit=32,
            historical_claim_limit=16,
            autobiography_limit=24,
        )
        planning_digest = build_planning_digest(
            agenda=agenda,
            commitment_projection=commitment_projection,
            recent_events=recent_events,
        )
        self_core_blocks = {
            record.block_kind: record
            for record in self.list_current_core_memory_blocks(
                scope_id=resolved_agent_id,
                scope_type="agent",
                block_kinds=(
                    BrainCoreMemoryBlockKind.SELF_CORE.value,
                    BrainCoreMemoryBlockKind.SELF_CURRENT_ARC.value,
                ),
                limit=4,
            )
        }
        return build_private_working_memory_projection(
            scope_type="thread",
            scope_id=thread_id,
            working_context=working_context,
            agenda=agenda,
            commitment_projection=commitment_projection,
            scene=scene,
            scene_world_state=scene_world_state,
            engagement=engagement,
            body=body,
            continuity_graph=continuity_graph,
            continuity_dossiers=continuity_dossiers,
            procedural_skills=procedural_skills,
            recent_events=recent_events,
            planning_digest=planning_digest,
            self_core_blocks=self_core_blocks,
            reference_ts=reference_ts,
        )

    def build_scene_world_state_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        reference_ts: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
    ) -> BrainSceneWorldProjection:
        """Build the symbolic scene world-state projection for one presence scope."""
        recent_events = self.recent_brain_events(
            user_id=user_id,
            thread_id=thread_id,
            limit=recent_event_limit,
        )
        latest_event = recent_events[0] if recent_events else None
        resolved_presence_scope_key = presence_scope_key or next(
            (
                scope_key
                for event in recent_events
                if (scope_key := _event_presence_scope_key(event))
            ),
            "local:presence",
        )
        del agent_id, latest_event
        scene = self.get_scene_state_projection(scope_key=resolved_presence_scope_key)
        engagement = self.get_engagement_state_projection(scope_key=resolved_presence_scope_key)
        body = self.get_body_state_projection(scope_key=resolved_presence_scope_key)
        return derive_scene_world_state_projection(
            scope_type="presence",
            scope_id=resolved_presence_scope_key,
            scene=scene,
            engagement=engagement,
            body=body,
            recent_events=recent_events,
            reference_ts=reference_ts,
        )

    def build_predictive_world_model_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        reference_ts: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
    ) -> BrainPredictiveWorldModelProjection:
        """Return the current predictive world-model projection for one thread."""
        del user_id, recent_event_limit, agent_id
        projection = self.get_predictive_world_model_projection(scope_key=thread_id)
        if presence_scope_key is not None and projection.presence_scope_key != presence_scope_key:
            projection.presence_scope_key = presence_scope_key
        if reference_ts is not None and str(reference_ts).strip():
            projection.updated_at = str(reference_ts)
            projection.sync_lists()
        return projection

    def build_counterfactual_rehearsal_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        reference_ts: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
    ) -> BrainCounterfactualRehearsalProjection:
        """Return the current counterfactual rehearsal projection for one thread."""
        del user_id, recent_event_limit, agent_id
        projection = self.get_counterfactual_rehearsal_projection(scope_key=thread_id)
        if presence_scope_key is not None and projection.presence_scope_key != presence_scope_key:
            projection.presence_scope_key = presence_scope_key
        if reference_ts is not None and str(reference_ts).strip():
            projection.updated_at = str(reference_ts)
            projection.sync_lists()
        return projection

    def build_embodied_executive_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        reference_ts: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
    ) -> BrainEmbodiedExecutiveProjection:
        """Return the current embodied executive projection for one thread."""
        del user_id, recent_event_limit, agent_id
        projection = self.get_embodied_executive_projection(scope_key=thread_id)
        if presence_scope_key is not None and projection.presence_scope_key != presence_scope_key:
            projection.presence_scope_key = presence_scope_key
        if reference_ts is not None and str(reference_ts).strip():
            projection.updated_at = str(reference_ts)
            projection.sync_lists()
        return projection

    def build_practice_director_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        reference_ts: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
    ) -> BrainPracticeDirectorProjection:
        """Return the current practice-director projection for one thread."""
        del user_id, recent_event_limit, agent_id
        projection = self.get_practice_director_projection(scope_key=thread_id)
        if presence_scope_key is not None and projection.presence_scope_key != presence_scope_key:
            projection.presence_scope_key = presence_scope_key
        if reference_ts is not None and str(reference_ts).strip():
            projection.updated_at = str(reference_ts)
            projection.sync_lists()
        return projection

    def build_skill_evidence_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        reference_ts: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
    ) -> BrainSkillEvidenceLedger:
        """Return the current skill-evidence projection for one thread."""
        del user_id, recent_event_limit, agent_id, presence_scope_key
        projection = self.get_skill_evidence_projection(scope_key=thread_id)
        if reference_ts is not None and str(reference_ts).strip():
            projection.updated_at = str(reference_ts)
            projection.sync_lists()
        return projection

    def build_skill_governance_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        reference_ts: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
    ) -> BrainSkillGovernanceProjection:
        """Return the current skill-governance projection for one thread."""
        del user_id, recent_event_limit, agent_id, presence_scope_key
        projection = self.get_skill_governance_projection(scope_key=thread_id)
        if reference_ts is not None and str(reference_ts).strip():
            projection.updated_at = str(reference_ts)
            projection.sync_lists()
        return projection

    def build_adapter_governance_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        reference_ts: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
    ) -> BrainAdapterGovernanceProjection:
        """Return the current adapter-governance projection for one thread."""
        del user_id, recent_event_limit, agent_id, presence_scope_key
        projection = self.get_adapter_governance_projection(scope_key=thread_id)
        if reference_ts is not None and str(reference_ts).strip():
            projection.updated_at = str(reference_ts)
            projection.sync_lists()
        return projection

    def build_active_situation_model_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        reference_ts: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
    ) -> BrainActiveSituationProjection:
        """Build the derived active situation-model projection for one thread."""
        recent_events = self.recent_brain_events(
            user_id=user_id,
            thread_id=thread_id,
            limit=recent_event_limit,
        )
        latest_event = recent_events[0] if recent_events else None
        resolved_agent_id = (
            agent_id
            or str(getattr(latest_event, "agent_id", "")).strip()
            or _DEFAULT_RELATIONSHIP_AGENT_ID
        )
        resolved_presence_scope_key = presence_scope_key or next(
            (
                scope_key
                for event in recent_events
                if (scope_key := _event_presence_scope_key(event))
            ),
            "local:presence",
        )
        private_working_memory = self.build_private_working_memory_projection(
            user_id=user_id,
            thread_id=thread_id,
            reference_ts=reference_ts,
            recent_event_limit=recent_event_limit,
            agent_id=resolved_agent_id,
            presence_scope_key=resolved_presence_scope_key,
        )
        agenda = self.get_agenda_projection(scope_key=thread_id, user_id=user_id)
        commitment_projection = self.get_session_commitment_projection(
            agent_id=resolved_agent_id,
            user_id=user_id,
            thread_id=thread_id,
        )
        core_blocks = self._list_continuity_core_blocks(
            user_id=user_id,
            agent_id=resolved_agent_id,
        )
        scene = self.get_scene_state_projection(scope_key=resolved_presence_scope_key)
        engagement = self.get_engagement_state_projection(scope_key=resolved_presence_scope_key)
        body = self.get_body_state_projection(scope_key=resolved_presence_scope_key)
        scene_world_state = self.build_scene_world_state_projection(
            user_id=user_id,
            thread_id=thread_id,
            reference_ts=reference_ts,
            recent_event_limit=recent_event_limit,
            agent_id=resolved_agent_id,
            presence_scope_key=resolved_presence_scope_key,
        )
        procedural_skills = self.build_procedural_skill_projection(
            scope_type="thread",
            scope_id=thread_id,
        )
        predictive_world_model = self.build_predictive_world_model_projection(
            user_id=user_id,
            thread_id=thread_id,
            reference_ts=reference_ts,
            recent_event_limit=recent_event_limit,
            agent_id=resolved_agent_id,
            presence_scope_key=resolved_presence_scope_key,
        )
        continuity_graph = self.build_continuity_graph(
            agent_id=resolved_agent_id,
            user_id=user_id,
            thread_id=thread_id,
            scope_type="user",
            scope_id=user_id,
            reference_ts=reference_ts,
            core_blocks=core_blocks,
            commitment_projection=commitment_projection,
            agenda=agenda,
            procedural_skills=procedural_skills,
            scene_world_state=scene_world_state,
            presence_scope_key=resolved_presence_scope_key,
            recent_event_limit=recent_event_limit,
            current_claim_limit=32,
            historical_claim_limit=16,
            autobiography_limit=24,
        )
        continuity_dossiers = self.build_continuity_dossiers(
            agent_id=resolved_agent_id,
            user_id=user_id,
            thread_id=thread_id,
            scope_type="user",
            scope_id=user_id,
            continuity_graph=continuity_graph,
            reference_ts=reference_ts,
            core_blocks=core_blocks,
            commitment_projection=commitment_projection,
            agenda=agenda,
            procedural_skills=procedural_skills,
            scene_world_state=scene_world_state,
            presence_scope_key=resolved_presence_scope_key,
            recent_event_limit=recent_event_limit,
            current_claim_limit=32,
            historical_claim_limit=16,
            autobiography_limit=24,
        )
        planning_digest = build_planning_digest(
            agenda=agenda,
            commitment_projection=commitment_projection,
            recent_events=recent_events,
        )
        return derive_active_situation_model_projection(
            scope_type="thread",
            scope_id=thread_id,
            private_working_memory=private_working_memory,
            agenda=agenda,
            commitment_projection=commitment_projection,
            scene=scene,
            scene_world_state=scene_world_state,
            engagement=engagement,
            body=body,
            continuity_graph=continuity_graph,
            continuity_dossiers=continuity_dossiers,
            procedural_skills=procedural_skills,
            predictive_world_model=predictive_world_model,
            recent_events=recent_events,
            planning_digest=planning_digest,
            reference_ts=reference_ts,
        )

    def refresh_private_working_memory_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_event_id: str | None,
        updated_at: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
        commit: bool = False,
    ) -> BrainPrivateWorkingMemoryProjection:
        """Rebuild and persist the current private working-memory projection."""
        return self._refresh_private_working_memory_projection(
            user_id=user_id,
            thread_id=thread_id,
            source_event_id=source_event_id,
            updated_at=updated_at,
            recent_event_limit=recent_event_limit,
            agent_id=agent_id,
            presence_scope_key=presence_scope_key,
            commit=commit,
        )

    def refresh_scene_world_state_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_event_id: str | None,
        updated_at: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
        commit: bool = False,
    ) -> BrainSceneWorldProjection:
        """Rebuild and persist the symbolic scene world-state projection."""
        return self._refresh_scene_world_state_projection(
            user_id=user_id,
            thread_id=thread_id,
            source_event_id=source_event_id,
            updated_at=updated_at,
            recent_event_limit=recent_event_limit,
            agent_id=agent_id,
            presence_scope_key=presence_scope_key,
            commit=commit,
        )

    def refresh_active_situation_model_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_event_id: str | None,
        updated_at: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
        commit: bool = False,
    ) -> BrainActiveSituationProjection:
        """Rebuild and persist the current active situation-model projection."""
        return self._refresh_active_situation_model_projection(
            user_id=user_id,
            thread_id=thread_id,
            source_event_id=source_event_id,
            updated_at=updated_at,
            recent_event_limit=recent_event_limit,
            agent_id=agent_id,
            presence_scope_key=presence_scope_key,
            commit=commit,
        )

    def refresh_predictive_world_model_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_event_id: str | None,
        updated_at: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
        commit: bool = False,
    ) -> BrainPredictiveWorldModelProjection:
        """Persist the current predictive world-model projection without recomputing heuristics."""
        return self._refresh_predictive_world_model_projection(
            user_id=user_id,
            thread_id=thread_id,
            source_event_id=source_event_id,
            updated_at=updated_at,
            recent_event_limit=recent_event_limit,
            agent_id=agent_id,
            presence_scope_key=presence_scope_key,
            commit=commit,
        )

    def get_agenda_projection(
        self,
        *,
        scope_key: str,
        user_id: str | None = None,
    ) -> BrainAgendaProjection:
        """Return the current agenda projection with task compatibility fallback."""
        projection = BrainAgendaProjection.from_dict(
            self._get_projection_dict(
                projection_name=AGENDA_PROJECTION,
                scope_key=scope_key,
            )
        )
        if user_id is not None and not projection.goals and not projection.open_goals:
            for task in self.active_tasks(user_id=user_id, limit=8):
                title = str(task["title"])
                if title not in projection.open_goals and title not in projection.completed_goals:
                    projection.open_goals.append(title)
        return projection

    def get_heartbeat_projection(self, *, scope_key: str) -> BrainHeartbeatProjection:
        """Return the current heartbeat projection for one thread."""
        return BrainHeartbeatProjection.from_dict(
            self._get_projection_dict(
                projection_name=HEARTBEAT_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_autonomy_ledger_projection(self, *, scope_key: str) -> BrainAutonomyLedgerProjection:
        """Return the current autonomy-ledger projection for one thread."""
        return BrainAutonomyLedgerProjection.from_dict(
            self._get_projection_dict(
                projection_name=AUTONOMY_LEDGER_PROJECTION,
                scope_key=scope_key,
            )
        )

    def add_action_event(
        self,
        *,
        agent_id: str,
        user_id: str,
        thread_id: str,
        action_id: str,
        source: str,
        accepted: bool,
        preview_only: bool,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Append an embodied action event."""
        cursor = self._conn.execute(
            """
            INSERT INTO action_events (
                agent_id, user_id, thread_id, action_id, source,
                accepted, preview_only, summary, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                user_id,
                thread_id,
                action_id,
                source,
                int(bool(accepted)),
                int(bool(preview_only)),
                summary,
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                _utc_now(),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def recent_action_events(
        self,
        *,
        user_id: str,
        thread_id: str,
        limit: int = 6,
    ) -> list[BrainActionEventRecord]:
        """Return recent embodied action events for one user thread."""
        rows = self._conn.execute(
            """
            SELECT * FROM action_events
            WHERE user_id = ? AND thread_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, thread_id, limit),
        ).fetchall()
        return [
            BrainActionEventRecord(
                id=int(row["id"]),
                agent_id=str(row["agent_id"]),
                user_id=str(row["user_id"]),
                thread_id=str(row["thread_id"]),
                action_id=str(row["action_id"]),
                source=str(row["source"]),
                accepted=bool(row["accepted"]),
                preview_only=bool(row["preview_only"]),
                summary=str(row["summary"]),
                metadata_json=str(row["metadata_json"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    @property
    def fts_enabled(self) -> bool:
        """Return whether SQLite FTS-backed retrieval is available."""
        return self._fts_enabled

    def remember_semantic_memory(
        self,
        *,
        user_id: str,
        namespace: str,
        subject: str,
        value: dict[str, Any],
        rendered_text: str,
        confidence: float,
        singleton: bool,
        source_event_id: str | None = None,
        source_episode_id: int | None = None,
        provenance: dict[str, Any] | None = None,
        stale_after_seconds: int | None = None,
    ) -> BrainSemanticMemoryRecord:
        """Insert or update one canonical semantic memory row."""
        now = _utc_now()
        normalized_subject = " ".join((subject or "").split()).strip() or "user"
        normalized_confidence = max(0.0, min(1.0, float(confidence)))
        contradiction_key = semantic_contradiction_key(namespace, normalized_subject, singleton)
        resolved_staleness = (
            semantic_default_staleness(namespace)
            if stale_after_seconds is None
            else stale_after_seconds
        )
        value_json = json.dumps(value, ensure_ascii=False, sort_keys=True)
        provenance_json = json.dumps(provenance or {}, ensure_ascii=False, sort_keys=True)

        if source_event_id is not None:
            row = self._conn.execute(
                """
                SELECT * FROM memory_semantic
                WHERE user_id = ? AND source_event_id = ? AND namespace = ? AND subject = ?
                ORDER BY id DESC LIMIT 1
                """,
                (user_id, source_event_id, namespace, normalized_subject),
            ).fetchone()
            if row is not None:
                return self._semantic_memory_from_row(row)

        existing_same = self._conn.execute(
            """
            SELECT * FROM memory_semantic
            WHERE user_id = ? AND namespace = ? AND subject = ? AND rendered_text = ? AND status = 'active'
            ORDER BY id DESC LIMIT 1
            """,
            (user_id, namespace, normalized_subject, rendered_text),
        ).fetchone()
        if existing_same is not None:
            merged_confidence = min(
                1.0, max(normalized_confidence, float(existing_same["confidence"]) + 0.05)
            )
            self._conn.execute(
                """
                UPDATE memory_semantic
                SET confidence = ?, value_json = ?, provenance_json = ?, updated_at = ?, stale_after_seconds = ?
                WHERE id = ?
                """,
                (
                    merged_confidence,
                    value_json,
                    provenance_json,
                    now,
                    resolved_staleness,
                    int(existing_same["id"]),
                ),
            )
            self._upsert_semantic_fts(
                memory_id=int(existing_same["id"]),
                user_id=user_id,
                rendered_text=rendered_text,
                namespace=namespace,
                subject=normalized_subject,
            )
            self._conn.commit()
            return self._semantic_memory_from_row(
                self._conn.execute(
                    "SELECT * FROM memory_semantic WHERE id = ?",
                    (int(existing_same["id"]),),
                ).fetchone()
            )

        supersedes_memory_id: int | None = None
        if contradiction_key:
            conflicting_rows = self._conn.execute(
                """
                SELECT * FROM memory_semantic
                WHERE user_id = ? AND contradiction_key = ? AND status = 'active'
                ORDER BY id DESC
                """,
                (user_id, contradiction_key),
            ).fetchall()
            for row in conflicting_rows:
                row_id = int(row["id"])
                if row["rendered_text"] == rendered_text:
                    supersedes_memory_id = row_id
                    continue
                supersedes_memory_id = supersedes_memory_id or row_id
                self._conn.execute(
                    """
                    UPDATE memory_semantic
                    SET status = 'superseded', updated_at = ?
                    WHERE id = ?
                    """,
                    (now, row_id),
                )

        cursor = self._conn.execute(
            """
            INSERT INTO memory_semantic (
                user_id, namespace, subject, value_json, rendered_text, confidence, status,
                source_event_id, source_episode_id, provenance_json, contradiction_key,
                supersedes_memory_id, observed_at, updated_at, stale_after_seconds
            )
            VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                namespace,
                normalized_subject,
                value_json,
                rendered_text,
                normalized_confidence,
                source_event_id,
                source_episode_id,
                provenance_json,
                contradiction_key,
                supersedes_memory_id,
                now,
                now,
                resolved_staleness,
            ),
        )
        memory_id = int(cursor.lastrowid)
        self._upsert_semantic_fts(
            memory_id=memory_id,
            user_id=user_id,
            rendered_text=rendered_text,
            namespace=namespace,
            subject=normalized_subject,
        )
        self._conn.commit()
        return self._semantic_memory_from_row(
            self._conn.execute(
                "SELECT * FROM memory_semantic WHERE id = ?", (memory_id,)
            ).fetchone()
        )

    def semantic_memories(
        self,
        *,
        user_id: str,
        namespaces: tuple[str, ...] | None = None,
        statuses: tuple[str, ...] = ("active",),
        limit: int = 24,
        include_inactive: bool = False,
        include_stale: bool = True,
    ) -> list[BrainSemanticMemoryRecord]:
        """Return canonical semantic memories for one user."""
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if namespaces:
            clauses.append(f"namespace IN ({','.join('?' for _ in namespaces)})")
            params.extend(namespaces)
        if not include_inactive:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(statuses)
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_semantic
            WHERE {" AND ".join(clauses)}
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        records = [self._semantic_memory_from_row(row) for row in rows]
        if not include_stale:
            records = [record for record in records if not record.is_stale]
        return records

    def upsert_narrative_memory(
        self,
        *,
        user_id: str,
        thread_id: str,
        kind: str,
        title: str,
        summary: str,
        details: dict[str, Any] | None = None,
        status: str,
        confidence: float,
        source_event_id: str | None = None,
        provenance: dict[str, Any] | None = None,
        contradiction_key: str | None = None,
        stale_after_seconds: int | None = None,
    ) -> BrainNarrativeMemoryRecord:
        """Insert or update one canonical narrative memory row."""
        now = _utc_now()
        normalized_title = " ".join((title or "").split()).strip()
        if not normalized_title:
            raise ValueError("Narrative title must not be empty")
        normalized_summary = (
            " ".join((summary or normalized_title).split()).strip() or normalized_title
        )
        normalized_confidence = max(0.0, min(1.0, float(confidence)))
        resolved_contradiction_key = contradiction_key or (
            f"{kind}:{normalized_title.lower()}"
            if kind in {"commitment", "session_summary", "daily_summary"}
            else None
        )
        resolved_staleness = (
            narrative_default_staleness(kind)
            if stale_after_seconds is None
            else stale_after_seconds
        )
        details_json = json.dumps(details or {}, ensure_ascii=False, sort_keys=True)
        provenance_json = json.dumps(provenance or {}, ensure_ascii=False, sort_keys=True)

        if source_event_id is not None:
            row = self._conn.execute(
                """
                SELECT * FROM memory_narrative
                WHERE user_id = ? AND thread_id = ? AND source_event_id = ? AND kind = ?
                ORDER BY id DESC LIMIT 1
                """,
                (user_id, thread_id, source_event_id, kind),
            ).fetchone()
            if row is not None:
                return self._narrative_memory_from_row(row)

        existing = self._conn.execute(
            """
            SELECT * FROM memory_narrative
            WHERE user_id = ? AND thread_id = ? AND kind = ? AND title = ?
              AND status NOT IN ('done', 'cancelled', 'superseded')
            ORDER BY id DESC LIMIT 1
            """,
            (user_id, thread_id, kind, normalized_title),
        ).fetchone()
        if existing is not None:
            row_id = int(existing["id"])
            self._conn.execute(
                """
                UPDATE memory_narrative
                SET summary = ?, details_json = ?, status = ?, confidence = ?, provenance_json = ?,
                    updated_at = ?, stale_after_seconds = ?
                WHERE id = ?
                """,
                (
                    normalized_summary,
                    details_json,
                    status,
                    max(normalized_confidence, float(existing["confidence"])),
                    provenance_json,
                    now,
                    resolved_staleness,
                    row_id,
                ),
            )
            self._upsert_narrative_fts(
                memory_id=row_id,
                user_id=user_id,
                thread_id=thread_id,
                title=normalized_title,
                summary=normalized_summary,
                kind=kind,
            )
            self._conn.commit()
            return self._narrative_memory_from_row(
                self._conn.execute(
                    "SELECT * FROM memory_narrative WHERE id = ?", (row_id,)
                ).fetchone()
            )

        supersedes_memory_id: int | None = None
        if resolved_contradiction_key:
            conflicting_rows = self._conn.execute(
                """
                SELECT * FROM memory_narrative
                WHERE user_id = ? AND thread_id = ? AND contradiction_key = ?
                  AND status NOT IN ('done', 'cancelled', 'superseded')
                ORDER BY id DESC
                """,
                (user_id, thread_id, resolved_contradiction_key),
            ).fetchall()
            for row in conflicting_rows:
                row_id = int(row["id"])
                if row["title"] == normalized_title and row["summary"] == normalized_summary:
                    supersedes_memory_id = row_id
                    continue
                supersedes_memory_id = supersedes_memory_id or row_id
                self._conn.execute(
                    """
                    UPDATE memory_narrative
                    SET status = 'superseded', updated_at = ?
                    WHERE id = ?
                    """,
                    (now, row_id),
                )

        cursor = self._conn.execute(
            """
            INSERT INTO memory_narrative (
                user_id, thread_id, kind, title, summary, details_json, status, confidence,
                source_event_id, provenance_json, contradiction_key, supersedes_memory_id,
                observed_at, updated_at, stale_after_seconds
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                thread_id,
                kind,
                normalized_title,
                normalized_summary,
                details_json,
                status,
                normalized_confidence,
                source_event_id,
                provenance_json,
                resolved_contradiction_key,
                supersedes_memory_id,
                now,
                now,
                resolved_staleness,
            ),
        )
        memory_id = int(cursor.lastrowid)
        self._upsert_narrative_fts(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            title=normalized_title,
            summary=normalized_summary,
            kind=kind,
        )
        self._conn.commit()
        return self._narrative_memory_from_row(
            self._conn.execute(
                "SELECT * FROM memory_narrative WHERE id = ?", (memory_id,)
            ).fetchone()
        )

    def narrative_memories(
        self,
        *,
        user_id: str,
        thread_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        statuses: tuple[str, ...] = ("open", "active"),
        limit: int = 12,
        include_stale: bool = True,
    ) -> list[BrainNarrativeMemoryRecord]:
        """Return canonical narrative memories for one user/thread."""
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if kinds:
            clauses.append(f"kind IN ({','.join('?' for _ in kinds)})")
            params.extend(kinds)
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(statuses)
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_narrative
            WHERE {" AND ".join(clauses)}
            ORDER BY updated_at DESC, confidence DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        records = [self._narrative_memory_from_row(row) for row in rows]
        if not include_stale:
            records = [record for record in records if not record.is_stale]
        return records

    def add_episodic_memory(
        self,
        *,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        kind: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        confidence: float,
        source_event_id: str | None = None,
        provenance: dict[str, Any] | None = None,
        stale_after_seconds: int | None = None,
        status: str = "active",
        observed_at: str | None = None,
    ) -> BrainEpisodicMemoryRecord:
        """Insert one canonical episodic memory row."""
        now = _utc_now()
        normalized_summary = " ".join((summary or "").split()).strip()
        if not normalized_summary:
            raise ValueError("Episodic summary must not be empty")
        if source_event_id is not None:
            row = self._conn.execute(
                """
                SELECT * FROM memory_episodic
                WHERE user_id = ? AND thread_id = ? AND source_event_id = ? AND kind = ?
                ORDER BY id DESC LIMIT 1
                """,
                (user_id, thread_id, source_event_id, kind),
            ).fetchone()
            if row is not None:
                return self._episodic_memory_from_row(row)

        cursor = self._conn.execute(
            """
            INSERT INTO memory_episodic (
                agent_id, user_id, session_id, thread_id, kind, summary, payload_json, confidence,
                source_event_id, provenance_json, observed_at, updated_at, stale_after_seconds, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                user_id,
                session_id,
                thread_id,
                kind,
                normalized_summary,
                json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
                max(0.0, min(1.0, float(confidence))),
                source_event_id,
                json.dumps(provenance or {}, ensure_ascii=False, sort_keys=True),
                observed_at or now,
                now,
                stale_after_seconds,
                status,
            ),
        )
        memory_id = int(cursor.lastrowid)
        self._upsert_episodic_fts(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            summary=normalized_summary,
            kind=kind,
        )
        self._conn.commit()
        return self._episodic_memory_from_row(
            self._conn.execute(
                "SELECT * FROM memory_episodic WHERE id = ?", (memory_id,)
            ).fetchone()
        )

    def episodic_memories(
        self,
        *,
        user_id: str,
        thread_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        statuses: tuple[str, ...] = ("active",),
        limit: int = 12,
        include_stale: bool = True,
    ) -> list[BrainEpisodicMemoryRecord]:
        """Return canonical episodic memories for one user/thread."""
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if kinds:
            clauses.append(f"kind IN ({','.join('?' for _ in kinds)})")
            params.extend(kinds)
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(statuses)
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_episodic
            WHERE {" AND ".join(clauses)}
            ORDER BY observed_at DESC, confidence DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        records = [self._episodic_memory_from_row(row) for row in rows]
        if not include_stale:
            records = [record for record in records if not record.is_stale]
        return records

    def search_semantic_memories(
        self,
        *,
        user_id: str,
        text: str,
        namespaces: tuple[str, ...] | None = None,
        limit: int = 8,
        include_stale: bool = False,
    ) -> list[BrainMemorySearchResult]:
        """Return semantic-memory retrieval hits with FTS fallback."""
        records = self.semantic_memories(
            user_id=user_id,
            namespaces=namespaces,
            limit=max(limit * 4, limit),
            include_inactive=False,
            include_stale=True,
        )
        return self._search_memory_records(
            records=records,
            text=text,
            layer="semantic",
            limit=limit,
            summary_getter=lambda record: record.rendered_text,
            include_stale=include_stale,
            fts_table="memory_semantic_fts",
            fts_join_sql="""
                SELECT ms.*, bm25(memory_semantic_fts) AS lexical_rank
                FROM memory_semantic_fts
                JOIN memory_semantic AS ms ON ms.id = memory_semantic_fts.memory_id
                WHERE memory_semantic_fts MATCH ? AND ms.user_id = ? AND ms.status = 'active'
            """,
            fts_params_builder=lambda: [text, user_id],
            record_builder=self._semantic_memory_from_row,
            namespace_filter=namespaces,
        )

    def search_narrative_memories(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        text: str,
        kinds: tuple[str, ...] | None = None,
        limit: int = 8,
        include_stale: bool = False,
    ) -> list[BrainMemorySearchResult]:
        """Return narrative-memory retrieval hits with FTS fallback."""
        records = self.narrative_memories(
            user_id=user_id,
            thread_id=thread_id,
            kinds=kinds,
            statuses=("open", "active"),
            limit=max(limit * 4, limit),
            include_stale=True,
        )
        return self._search_memory_records(
            records=records,
            text=text,
            layer="narrative",
            limit=limit,
            summary_getter=lambda record: record.summary,
            include_stale=include_stale,
            fts_table="memory_narrative_fts",
            fts_join_sql="""
                SELECT mn.*, bm25(memory_narrative_fts) AS lexical_rank
                FROM memory_narrative_fts
                JOIN memory_narrative AS mn ON mn.id = memory_narrative_fts.memory_id
                WHERE memory_narrative_fts MATCH ? AND mn.user_id = ?
                  AND mn.status IN ('open', 'active')
            """,
            fts_params_builder=lambda: [text, user_id],
            record_builder=self._narrative_memory_from_row,
            thread_filter=thread_id,
            kind_filter=kinds,
        )

    def search_episodic_memories(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        text: str,
        kinds: tuple[str, ...] | None = None,
        limit: int = 8,
        include_stale: bool = False,
    ) -> list[BrainMemorySearchResult]:
        """Return episodic-memory retrieval hits with FTS fallback."""
        records = self.episodic_memories(
            user_id=user_id,
            thread_id=thread_id,
            kinds=kinds,
            limit=max(limit * 4, limit),
            include_stale=True,
        )
        return self._search_memory_records(
            records=records,
            text=text,
            layer="episodic",
            limit=limit,
            summary_getter=lambda record: record.summary,
            include_stale=include_stale,
            fts_table="memory_episodic_fts",
            fts_join_sql="""
                SELECT me.*, bm25(memory_episodic_fts) AS lexical_rank
                FROM memory_episodic_fts
                JOIN memory_episodic AS me ON me.id = memory_episodic_fts.memory_id
                WHERE memory_episodic_fts MATCH ? AND me.user_id = ? AND me.status = 'active'
            """,
            fts_params_builder=lambda: [text, user_id],
            record_builder=self._episodic_memory_from_row,
            thread_filter=thread_id,
            kind_filter=kinds,
        )

    def record_memory_export(
        self,
        *,
        user_id: str,
        thread_id: str,
        export_kind: str,
        path: Path,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ):
        """Persist one inspectable memory export artifact."""
        self._conn.execute(
            """
            INSERT INTO memory_exports (
                user_id, thread_id, export_kind, path, payload_json, metadata_json, generated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                thread_id,
                export_kind,
                str(path),
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                _utc_now(),
            ),
        )
        self._conn.commit()

    def _memory_export_from_row(self, row: sqlite3.Row | None) -> BrainMemoryExportRecord | None:
        """Build one typed memory-export record from a SQLite row."""
        if row is None:
            return None
        return BrainMemoryExportRecord(
            id=int(row["id"]),
            user_id=str(row["user_id"]),
            thread_id=str(row["thread_id"]),
            export_kind=str(row["export_kind"]),
            path=str(row["path"]),
            payload_json=str(row["payload_json"]),
            metadata_json=str(row["metadata_json"] or "{}"),
            generated_at=str(row["generated_at"]),
        )

    def list_memory_exports(
        self,
        *,
        user_id: str,
        thread_id: str,
        export_kind: str | None = None,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Return recent memory-export artifacts for one thread."""
        clauses = ["user_id = ?", "thread_id = ?"]
        params: list[Any] = [user_id, thread_id]
        if export_kind is not None:
            clauses.append("export_kind = ?")
            params.append(export_kind)
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_exports
            WHERE {" AND ".join(clauses)}
            ORDER BY id DESC, generated_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        records = [self._memory_export_from_row(row) for row in rows]
        return [record.as_dict() for record in records if record is not None]

    def upsert_memory_embedding(
        self,
        *,
        layer: str,
        memory_id: int,
        provider_name: str,
        vector: list[float],
    ):
        """Store an optional embedding vector for one memory row."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memory_embeddings (
                layer, memory_id, provider_name, vector_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                layer,
                memory_id,
                provider_name,
                json.dumps(vector, ensure_ascii=False),
                _utc_now(),
            ),
        )
        self._conn.commit()

    def memory_embeddings_available(
        self,
        *,
        provider_name: str,
        layer: str | None = None,
    ) -> bool:
        """Return whether optional embeddings exist for the requested provider/layer."""
        if layer is None:
            row = self._conn.execute(
                """
                SELECT 1 FROM memory_embeddings
                WHERE provider_name = ?
                LIMIT 1
                """,
                (provider_name,),
            ).fetchone()
        else:
            row = self._conn.execute(
                """
                SELECT 1 FROM memory_embeddings
                WHERE provider_name = ? AND layer = ?
                LIMIT 1
                """,
                (provider_name, layer),
            ).fetchone()
        return row is not None

    def _semantic_memory_from_row(self, row: sqlite3.Row | None) -> BrainSemanticMemoryRecord:
        """Convert one SQLite row into a canonical semantic memory record."""
        if row is None:
            raise KeyError("Missing semantic memory row")
        return BrainSemanticMemoryRecord(
            id=int(row["id"]),
            user_id=str(row["user_id"]),
            namespace=str(row["namespace"]),
            subject=str(row["subject"]),
            value_json=str(row["value_json"]),
            rendered_text=str(row["rendered_text"]),
            confidence=float(row["confidence"]),
            status=str(row["status"]),
            source_event_id=str(row["source_event_id"])
            if row["source_event_id"] is not None
            else None,
            source_episode_id=(
                int(row["source_episode_id"]) if row["source_episode_id"] is not None else None
            ),
            provenance_json=str(row["provenance_json"]),
            contradiction_key=(
                str(row["contradiction_key"]) if row["contradiction_key"] is not None else None
            ),
            supersedes_memory_id=(
                int(row["supersedes_memory_id"])
                if row["supersedes_memory_id"] is not None
                else None
            ),
            observed_at=str(row["observed_at"]),
            updated_at=str(row["updated_at"]),
            stale_after_seconds=(
                int(row["stale_after_seconds"]) if row["stale_after_seconds"] is not None else None
            ),
        )

    def _narrative_memory_from_row(self, row: sqlite3.Row | None) -> BrainNarrativeMemoryRecord:
        """Convert one SQLite row into a canonical narrative memory record."""
        if row is None:
            raise KeyError("Missing narrative memory row")
        return BrainNarrativeMemoryRecord(
            id=int(row["id"]),
            user_id=str(row["user_id"]),
            thread_id=str(row["thread_id"]),
            kind=str(row["kind"]),
            title=str(row["title"]),
            summary=str(row["summary"]),
            details_json=str(row["details_json"]),
            status=str(row["status"]),
            confidence=float(row["confidence"]),
            source_event_id=str(row["source_event_id"])
            if row["source_event_id"] is not None
            else None,
            provenance_json=str(row["provenance_json"]),
            contradiction_key=(
                str(row["contradiction_key"]) if row["contradiction_key"] is not None else None
            ),
            supersedes_memory_id=(
                int(row["supersedes_memory_id"])
                if row["supersedes_memory_id"] is not None
                else None
            ),
            observed_at=str(row["observed_at"]),
            updated_at=str(row["updated_at"]),
            stale_after_seconds=(
                int(row["stale_after_seconds"]) if row["stale_after_seconds"] is not None else None
            ),
        )

    def _episodic_memory_from_row(self, row: sqlite3.Row | None) -> BrainEpisodicMemoryRecord:
        """Convert one SQLite row into a canonical episodic memory record."""
        if row is None:
            raise KeyError("Missing episodic memory row")
        return BrainEpisodicMemoryRecord(
            id=int(row["id"]),
            agent_id=str(row["agent_id"]),
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            thread_id=str(row["thread_id"]),
            kind=str(row["kind"]),
            summary=str(row["summary"]),
            payload_json=str(row["payload_json"]),
            confidence=float(row["confidence"]),
            source_event_id=str(row["source_event_id"])
            if row["source_event_id"] is not None
            else None,
            provenance_json=str(row["provenance_json"]),
            observed_at=str(row["observed_at"]),
            updated_at=str(row["updated_at"]),
            stale_after_seconds=(
                int(row["stale_after_seconds"]) if row["stale_after_seconds"] is not None else None
            ),
            status=str(row["status"]),
        )

    def _procedural_skill_from_row(
        self,
        row: sqlite3.Row | None,
    ) -> BrainProceduralSkillRecord | None:
        """Convert one SQLite row into a canonical procedural skill record."""
        if row is None:
            return None
        payload = {
            "skill_id": str(row["skill_id"]),
            "skill_family_key": str(row["skill_family_key"]),
            "template_fingerprint": str(row["template_fingerprint"]),
            "scope_type": str(row["scope_type"]),
            "scope_id": str(row["scope_id"]),
            "title": str(row["title"]),
            "purpose": str(row["purpose"]),
            "goal_family": str(row["goal_family"]),
            "status": str(row["status"]),
            "confidence": float(row["confidence"]),
            "activation_conditions": json.loads(str(row["activation_conditions_json"] or "[]")),
            "invariants": json.loads(str(row["invariants_json"] or "[]")),
            "step_template": json.loads(str(row["step_template_json"] or "[]")),
            "required_capability_ids": json.loads(str(row["required_capability_ids_json"] or "[]")),
            "effects": json.loads(str(row["effects_json"] or "[]")),
            "termination_conditions": json.loads(str(row["termination_conditions_json"] or "[]")),
            "failure_signatures": json.loads(str(row["failure_signatures_json"] or "[]")),
            "stats": json.loads(str(row["stats_json"] or "{}")),
            "supporting_trace_ids": json.loads(str(row["supporting_trace_ids_json"] or "[]")),
            "supporting_outcome_ids": json.loads(str(row["supporting_outcome_ids_json"] or "[]")),
            "supporting_plan_proposal_ids": json.loads(
                str(row["supporting_plan_proposal_ids_json"] or "[]")
            ),
            "supporting_commitment_ids": json.loads(
                str(row["supporting_commitment_ids_json"] or "[]")
            ),
            "supersedes_skill_id": _optional_text(row["supersedes_skill_id"]),
            "superseded_by_skill_id": _optional_text(row["superseded_by_skill_id"]),
            "retirement_reason": _optional_text(row["retirement_reason"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "retired_at": _optional_text(row["retired_at"]),
            "details": json.loads(str(row["details_json"] or "{}")),
        }
        return BrainProceduralSkillRecord.from_dict(payload)

    def _upsert_semantic_fts(
        self,
        *,
        memory_id: int,
        user_id: str,
        rendered_text: str,
        namespace: str,
        subject: str,
    ):
        """Refresh one semantic-memory FTS row."""
        if not self._fts_enabled:
            return
        self._conn.execute("DELETE FROM memory_semantic_fts WHERE memory_id = ?", (memory_id,))
        self._conn.execute(
            """
            INSERT INTO memory_semantic_fts (memory_id, user_id, rendered_text, namespace, subject)
            VALUES (?, ?, ?, ?, ?)
            """,
            (memory_id, user_id, rendered_text, namespace, subject),
        )

    def _upsert_narrative_fts(
        self,
        *,
        memory_id: int,
        user_id: str,
        thread_id: str,
        title: str,
        summary: str,
        kind: str,
    ):
        """Refresh one narrative-memory FTS row."""
        if not self._fts_enabled:
            return
        self._conn.execute("DELETE FROM memory_narrative_fts WHERE memory_id = ?", (memory_id,))
        self._conn.execute(
            """
            INSERT INTO memory_narrative_fts (memory_id, user_id, thread_id, title, summary, kind)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (memory_id, user_id, thread_id, title, summary, kind),
        )

    def _upsert_episodic_fts(
        self,
        *,
        memory_id: int,
        user_id: str,
        thread_id: str,
        summary: str,
        kind: str,
    ):
        """Refresh one episodic-memory FTS row."""
        if not self._fts_enabled:
            return
        self._conn.execute("DELETE FROM memory_episodic_fts WHERE memory_id = ?", (memory_id,))
        self._conn.execute(
            """
            INSERT INTO memory_episodic_fts (memory_id, user_id, thread_id, summary, kind)
            VALUES (?, ?, ?, ?, ?)
            """,
            (memory_id, user_id, thread_id, summary, kind),
        )

    def _search_memory_records(
        self,
        *,
        records: list[Any],
        text: str,
        layer: str,
        limit: int,
        summary_getter,
        include_stale: bool,
        fts_table: str,
        fts_join_sql: str,
        fts_params_builder,
        record_builder,
        namespace_filter: tuple[str, ...] | None = None,
        thread_filter: str | None = None,
        kind_filter: tuple[str, ...] | None = None,
    ) -> list[BrainMemorySearchResult]:
        """Search one layer with FTS when available and lexical fallback otherwise."""
        normalized_query = " ".join((text or "").split()).strip()
        filtered_records = list(records)
        if not include_stale:
            filtered_records = [
                record for record in filtered_records if not getattr(record, "is_stale", False)
            ]
        if not normalized_query:
            return [
                BrainMemorySearchResult(
                    layer=layer,
                    record_id=record.id,
                    summary=summary_getter(record),
                    score=max(1e-6, float(getattr(record, "confidence", 0.0))),
                    confidence=float(getattr(record, "confidence", 0.0)),
                    stale=bool(getattr(record, "is_stale", False)),
                    metadata={},
                )
                for record in filtered_records[:limit]
            ]

        if self._fts_enabled:
            try:
                rows = self._conn.execute(fts_join_sql, tuple(fts_params_builder())).fetchall()
                hits: list[BrainMemorySearchResult] = []
                for row in rows:
                    if (
                        namespace_filter
                        and "namespace" in row.keys()
                        and str(row["namespace"]) not in namespace_filter
                    ):
                        continue
                    if (
                        thread_filter
                        and "thread_id" in row.keys()
                        and str(row["thread_id"]) != thread_filter
                    ):
                        continue
                    if kind_filter and "kind" in row.keys() and str(row["kind"]) not in kind_filter:
                        continue
                    record = record_builder(row)
                    if not include_stale and getattr(record, "is_stale", False):
                        continue
                    lexical_rank = (
                        float(row["lexical_rank"]) if row["lexical_rank"] is not None else 0.0
                    )
                    score = max(1e-6, 1.0 / (1.0 + max(lexical_rank, 0.0))) + float(
                        getattr(record, "confidence", 0.0)
                    )
                    hits.append(
                        BrainMemorySearchResult(
                            layer=layer,
                            record_id=record.id,
                            summary=summary_getter(record),
                            score=score,
                            confidence=float(getattr(record, "confidence", 0.0)),
                            stale=bool(getattr(record, "is_stale", False)),
                            metadata={},
                        )
                    )
                if hits:
                    return sorted(
                        hits, key=lambda item: (item.score, item.confidence), reverse=True
                    )[:limit]
            except sqlite3.DatabaseError:
                pass

        query_tokens = [token for token in normalized_query.lower().split() if token]
        results: list[BrainMemorySearchResult] = []
        for record in filtered_records:
            lowered = summary_getter(record).lower()
            token_hits = sum(1 for token in query_tokens if token in lowered)
            if token_hits <= 0 and normalized_query.lower() not in lowered:
                continue
            score = float(token_hits) + float(getattr(record, "confidence", 0.0))
            results.append(
                BrainMemorySearchResult(
                    layer=layer,
                    record_id=record.id,
                    summary=summary_getter(record),
                    score=score,
                    confidence=float(getattr(record, "confidence", 0.0)),
                    stale=bool(getattr(record, "is_stale", False)),
                    metadata={},
                )
            )
        return sorted(results, key=lambda item: (item.score, item.confidence), reverse=True)[:limit]

    def _brain_event_from_row(self, row: sqlite3.Row | None) -> BrainEventRecord:
        """Convert one SQLite row into a typed brain event record."""
        if row is None:
            raise KeyError("Missing brain event row")
        return BrainEventRecord(
            id=int(row["id"]),
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            ts=str(row["ts"]),
            agent_id=str(row["agent_id"]),
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            thread_id=str(row["thread_id"]),
            source=str(row["source"]),
            correlation_id=str(row["correlation_id"])
            if row["correlation_id"] is not None
            else None,
            causal_parent_id=str(row["causal_parent_id"])
            if row["causal_parent_id"] is not None
            else None,
            confidence=float(row["confidence"]),
            payload_json=str(row["payload_json"]),
            tags_json=str(row["tags_json"]),
        )

    def _reflection_cycle_from_row(self, row: sqlite3.Row | None) -> BrainReflectionCycleRecord:
        """Convert one SQLite row into a typed reflection-cycle record."""
        if row is None:
            raise KeyError("Missing reflection cycle row")
        return BrainReflectionCycleRecord(
            cycle_id=str(row["cycle_id"]),
            user_id=str(row["user_id"]),
            thread_id=str(row["thread_id"]),
            trigger=str(row["trigger"]),
            status=str(row["status"]),
            input_episode_cursor=int(row["input_episode_cursor"]),
            input_event_cursor=int(row["input_event_cursor"]),
            terminal_episode_cursor=int(row["terminal_episode_cursor"]),
            terminal_event_cursor=int(row["terminal_event_cursor"]),
            draft_artifact_path=(
                str(row["draft_artifact_path"]) if row["draft_artifact_path"] is not None else None
            ),
            result_stats_json=str(row["result_stats_json"] or "{}"),
            skip_reason=str(row["skip_reason"]) if row["skip_reason"] is not None else None,
            error_json=str(row["error_json"]) if row["error_json"] is not None else None,
            started_at=str(row["started_at"]),
            completed_at=str(row["completed_at"]) if row["completed_at"] is not None else None,
        )

    def _get_projection_dict(
        self, *, projection_name: str, scope_key: str
    ) -> dict[str, Any] | None:
        """Return one raw projection JSON blob."""
        row = self._conn.execute(
            """
            SELECT projection_json FROM brain_projections
            WHERE projection_name = ? AND scope_key = ?
            """,
            (projection_name, scope_key),
        ).fetchone()
        if row is None:
            return None
        return json.loads(str(row["projection_json"]))

    def _upsert_projection(
        self,
        *,
        projection_name: str,
        scope_key: str,
        projection: dict[str, Any],
        source_event_id: str | None,
        updated_at: str,
        commit: bool,
    ):
        """Persist one raw projection JSON blob."""
        self._conn.execute(
            """
            INSERT INTO brain_projections (
                projection_name, scope_key, projection_json, source_event_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(projection_name, scope_key) DO UPDATE SET
                projection_json = excluded.projection_json,
                source_event_id = excluded.source_event_id,
                updated_at = excluded.updated_at
            """,
            (
                projection_name,
                scope_key,
                json.dumps(projection, ensure_ascii=False, sort_keys=True),
                source_event_id,
                updated_at,
            ),
        )
        if commit:
            self._conn.commit()

    def _refresh_private_working_memory_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_event_id: str | None,
        updated_at: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
        commit: bool = False,
    ) -> BrainPrivateWorkingMemoryProjection:
        """Rebuild and persist the current private working-memory projection."""
        projection = self.build_private_working_memory_projection(
            user_id=user_id,
            thread_id=thread_id,
            reference_ts=updated_at,
            recent_event_limit=recent_event_limit,
            agent_id=agent_id,
            presence_scope_key=presence_scope_key,
        )
        self._upsert_projection(
            projection_name=PRIVATE_WORKING_MEMORY_PROJECTION,
            scope_key=thread_id,
            projection=projection.as_dict(),
            source_event_id=source_event_id,
            updated_at=projection.updated_at,
            commit=commit,
        )
        return projection

    def _refresh_scene_world_state_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_event_id: str | None,
        updated_at: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
        commit: bool = False,
    ) -> BrainSceneWorldProjection:
        """Rebuild and persist the current symbolic scene world-state projection."""
        projection = self.build_scene_world_state_projection(
            user_id=user_id,
            thread_id=thread_id,
            reference_ts=updated_at,
            recent_event_limit=recent_event_limit,
            agent_id=agent_id,
            presence_scope_key=presence_scope_key,
        )
        self._upsert_projection(
            projection_name=SCENE_WORLD_STATE_PROJECTION,
            scope_key=projection.scope_id,
            projection=projection.as_dict(),
            source_event_id=source_event_id,
            updated_at=projection.updated_at,
            commit=commit,
        )
        return projection

    def _refresh_predictive_world_model_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_event_id: str | None,
        updated_at: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
        commit: bool = False,
    ) -> BrainPredictiveWorldModelProjection:
        """Persist the current predictive world-model projection."""
        projection = self.build_predictive_world_model_projection(
            user_id=user_id,
            thread_id=thread_id,
            reference_ts=updated_at,
            recent_event_limit=recent_event_limit,
            agent_id=agent_id,
            presence_scope_key=presence_scope_key,
        )
        self._upsert_projection(
            projection_name=PREDICTIVE_WORLD_MODEL_PROJECTION,
            scope_key=thread_id,
            projection=projection.as_dict(),
            source_event_id=source_event_id,
            updated_at=projection.updated_at,
            commit=commit,
        )
        return projection

    def _refresh_active_situation_model_projection(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_event_id: str | None,
        updated_at: str | None = None,
        recent_event_limit: int = 96,
        agent_id: str | None = None,
        presence_scope_key: str | None = None,
        commit: bool = False,
    ) -> BrainActiveSituationProjection:
        """Rebuild and persist the current active situation-model projection."""
        self._refresh_scene_world_state_projection(
            user_id=user_id,
            thread_id=thread_id,
            source_event_id=source_event_id,
            updated_at=updated_at,
            recent_event_limit=recent_event_limit,
            agent_id=agent_id,
            presence_scope_key=presence_scope_key,
            commit=False,
        )
        self._refresh_private_working_memory_projection(
            user_id=user_id,
            thread_id=thread_id,
            source_event_id=source_event_id,
            updated_at=updated_at,
            recent_event_limit=recent_event_limit,
            agent_id=agent_id,
            presence_scope_key=presence_scope_key,
            commit=False,
        )
        self._refresh_predictive_world_model_projection(
            user_id=user_id,
            thread_id=thread_id,
            source_event_id=source_event_id,
            updated_at=updated_at,
            recent_event_limit=recent_event_limit,
            agent_id=agent_id,
            presence_scope_key=presence_scope_key,
            commit=False,
        )
        projection = self.build_active_situation_model_projection(
            user_id=user_id,
            thread_id=thread_id,
            reference_ts=updated_at,
            recent_event_limit=recent_event_limit,
            agent_id=agent_id,
            presence_scope_key=presence_scope_key,
        )
        self._upsert_projection(
            projection_name=ACTIVE_SITUATION_MODEL_PROJECTION,
            scope_key=thread_id,
            projection=projection.as_dict(),
            source_event_id=source_event_id,
            updated_at=projection.updated_at,
            commit=commit,
        )
        return projection

    def _persist_body_state(
        self,
        *,
        scope_key: str,
        snapshot: BrainPresenceSnapshot,
        source_event_id: str | None,
        updated_at: str,
        commit: bool,
    ):
        """Persist the current body-state projection and compatibility snapshot."""
        snapshot = normalize_presence_snapshot(snapshot)
        self._conn.execute(
            """
            INSERT INTO presence_snapshots (scope_key, snapshot_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(scope_key) DO UPDATE SET
                snapshot_json = excluded.snapshot_json,
                updated_at = excluded.updated_at
            """,
            (
                scope_key,
                json.dumps(snapshot.as_dict(), ensure_ascii=False, sort_keys=True),
                updated_at,
            ),
        )
        self._upsert_projection(
            projection_name=BODY_STATE_PROJECTION,
            scope_key=scope_key,
            projection=snapshot.as_dict(),
            source_event_id=source_event_id,
            updated_at=updated_at,
            commit=False,
        )
        if commit:
            self._conn.commit()

    def _apply_event_to_projections(self, event: BrainEventRecord, *, commit: bool):
        """Update projection tables from one append-only event."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        updated_at = event.ts
        self._apply_body_state_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_scene_state_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_engagement_state_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_relationship_state_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_working_context_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_agenda_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_autonomy_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_heartbeat_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_scene_world_state_event(
            event=event,
            payload=payload,
            updated_at=updated_at,
        )
        self._apply_predictive_world_model_event(
            event=event,
            payload=payload,
            updated_at=updated_at,
        )
        self._apply_counterfactual_rehearsal_event(
            event=event,
            payload=payload,
            updated_at=updated_at,
        )
        self._apply_embodied_executive_event(
            event=event,
            payload=payload,
            updated_at=updated_at,
        )
        self._apply_practice_director_event(
            event=event,
            payload=payload,
            updated_at=updated_at,
        )
        self._apply_skill_evidence_event(
            event=event,
            payload=payload,
            updated_at=updated_at,
        )
        self._apply_skill_governance_event(
            event=event,
            payload=payload,
            updated_at=updated_at,
        )
        self._apply_adapter_governance_event(
            event=event,
            payload=payload,
            updated_at=updated_at,
        )
        self._apply_private_working_memory_event(
            event=event,
            payload=payload,
            updated_at=updated_at,
        )
        self._apply_active_situation_model_event(
            event=event,
            payload=payload,
            updated_at=updated_at,
        )
        if commit:
            self._conn.commit()

    def _apply_body_state_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        if event.event_type == BrainEventType.BODY_STATE_UPDATED:
            scope_key = str(payload.get("scope_key", "local:presence"))
            self._persist_body_state(
                scope_key=scope_key,
                snapshot=BrainPresenceSnapshot.from_dict(payload.get("snapshot")),
                source_event_id=event.event_id,
                updated_at=updated_at,
                commit=False,
            )
            return

        if event.event_type not in {
            BrainEventType.ROBOT_ACTION_OUTCOME,
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.ENGAGEMENT_CHANGED,
            BrainEventType.ATTENTION_CHANGED,
            BrainEventType.SCENE_CHANGED,
        }:
            return

        scope_key = str(
            payload.get("presence_scope_key") or payload.get("scope_key") or "local:presence"
        )
        snapshot = self.get_body_state_projection(scope_key=scope_key)
        if event.event_type == BrainEventType.ROBOT_ACTION_OUTCOME:
            status = payload.get("status", {})
            snapshot.robot_head_mode = str(status.get("mode", snapshot.robot_head_mode))
            snapshot.robot_head_armed = bool(status.get("armed", snapshot.robot_head_armed))
            snapshot.robot_head_available = bool(
                status.get("available", snapshot.robot_head_available)
            )
            snapshot.robot_head_last_action = payload.get("action_id")
            if payload.get("accepted", False):
                snapshot.robot_head_last_accepted_action = payload.get("action_id")
            else:
                snapshot.robot_head_last_rejected_action = payload.get("action_id")
            if payload.get("action_id") in {"cmd_return_neutral", "auto_safe_idle"}:
                snapshot.robot_head_last_safe_state = "neutral"
            snapshot.warnings = list(status.get("warnings", snapshot.warnings))
            snapshot.details = dict(status.get("details", snapshot.details))
        else:
            snapshot.vision_enabled = True
            snapshot.vision_connected = bool(
                payload.get("camera_connected", snapshot.vision_connected)
            )
            snapshot.camera_track_state = str(
                payload.get("camera_track_state", snapshot.camera_track_state)
            )
            snapshot.camera_disconnected = bool(
                snapshot.vision_enabled
                and not payload.get("camera_connected", snapshot.vision_connected)
            )
            snapshot.perception_disabled = False
            snapshot.perception_unreliable = (
                not bool(payload.get("camera_fresh", True))
            ) or snapshot.camera_track_state in {"stalled", "recovering"}
            snapshot.vision_unavailable = str(payload.get("sensor_health_reason", "")) in {
                "presence_detector_unavailable",
                "presence_detector_invalid_frame",
            }
            snapshot.last_fresh_frame_at = payload.get("last_fresh_frame_at")
            snapshot.frame_age_ms = (
                int(payload["frame_age_ms"])
                if payload.get("frame_age_ms") is not None
                else snapshot.frame_age_ms
            )
            snapshot.detection_backend = (
                payload.get("detection_backend") or snapshot.detection_backend
            )
            snapshot.detection_confidence = (
                float(payload["detection_confidence"])
                if payload.get("detection_confidence") is not None
                else snapshot.detection_confidence
            )
            snapshot.sensor_health_reason = payload.get("sensor_health_reason")
            snapshot.recovery_in_progress = bool(
                payload.get("recovery_in_progress", snapshot.recovery_in_progress)
            )
            snapshot.recovery_attempts = int(
                payload.get("recovery_attempts", snapshot.recovery_attempts)
            )
            if payload.get("person_present") == "present":
                snapshot.attention_target = "user"
            elif payload.get("person_present") == "absent":
                snapshot.attention_target = None
            if payload.get("engagement_state") in {"engaged", "listening", "speaking"}:
                snapshot.engagement_pose = "attentive"
            elif payload.get("engagement_state") in {"away", "idle"}:
                snapshot.engagement_pose = "neutral"
            details = dict(snapshot.details)
            if payload.get("summary"):
                details["last_visual_summary"] = payload.get("summary")
            details["last_perception_event_type"] = event.event_type
            if payload.get("enrichment_available") is not None:
                details["vision_enrichment_available"] = payload.get("enrichment_available")
            snapshot.details = details
        snapshot.updated_at = updated_at
        self._persist_body_state(
            scope_key=scope_key,
            snapshot=snapshot,
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_scene_state_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        scope_key = str(
            payload.get("presence_scope_key") or payload.get("scope_key") or "local:presence"
        )
        projection = self.get_scene_state_projection(scope_key=scope_key)
        if event.event_type == BrainEventType.BODY_STATE_UPDATED:
            snapshot = BrainPresenceSnapshot.from_dict(payload.get("snapshot"))
            projection.camera_connected = snapshot.vision_connected
            projection.camera_track_state = snapshot.camera_track_state
            projection.last_fresh_frame_at = snapshot.last_fresh_frame_at
            projection.frame_age_ms = snapshot.frame_age_ms
            projection.detection_backend = snapshot.detection_backend
            projection.detection_confidence = snapshot.detection_confidence
            projection.sensor_health_reason = snapshot.sensor_health_reason
            projection.recovery_in_progress = snapshot.recovery_in_progress
            projection.recovery_attempts = snapshot.recovery_attempts
            projection.enrichment_available = snapshot.details.get("vision_enrichment_available")
            if snapshot.camera_track_state != "healthy":
                projection.person_present = "uncertain"
            projection.source = event.source
            projection.updated_at = updated_at
        elif event.event_type in {
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.SCENE_CHANGED,
            BrainEventType.ENGAGEMENT_CHANGED,
        }:
            projection.camera_connected = bool(
                payload.get("camera_connected", projection.camera_connected)
            )
            projection.camera_track_state = str(
                payload.get("camera_track_state", projection.camera_track_state)
            )
            projection.person_present = str(
                payload.get("person_present", projection.person_present)
            )
            projection.scene_change_state = str(
                payload.get("scene_change", projection.scene_change_state)
            )
            projection.last_visual_summary = (
                payload.get("summary") or projection.last_visual_summary
            )
            projection.last_observed_at = payload.get("observed_at") or projection.last_observed_at
            projection.last_fresh_frame_at = (
                payload.get("last_fresh_frame_at") or projection.last_fresh_frame_at
            )
            projection.frame_age_ms = (
                int(payload["frame_age_ms"])
                if payload.get("frame_age_ms") is not None
                else projection.frame_age_ms
            )
            projection.detection_backend = (
                payload.get("detection_backend") or projection.detection_backend
            )
            projection.detection_confidence = (
                float(payload["detection_confidence"])
                if payload.get("detection_confidence") is not None
                else projection.detection_confidence
            )
            projection.sensor_health_reason = (
                payload.get("sensor_health_reason") or projection.sensor_health_reason
            )
            projection.recovery_in_progress = bool(
                payload.get("recovery_in_progress", projection.recovery_in_progress)
            )
            projection.recovery_attempts = int(
                payload.get("recovery_attempts", projection.recovery_attempts)
            )
            if payload.get("enrichment_available") is not None:
                projection.enrichment_available = payload.get("enrichment_available")
            confidence = payload.get("confidence")
            projection.confidence = (
                float(confidence) if confidence is not None else projection.confidence
            )
            projection.source = event.source
            projection.updated_at = updated_at
        else:
            return
        self._upsert_projection(
            projection_name=SCENE_STATE_PROJECTION,
            scope_key=scope_key,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_engagement_state_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        scope_key = str(
            payload.get("presence_scope_key") or payload.get("scope_key") or "local:presence"
        )
        projection = self.get_engagement_state_projection(scope_key=scope_key)
        if event.event_type == BrainEventType.BODY_STATE_UPDATED:
            snapshot = BrainPresenceSnapshot.from_dict(payload.get("snapshot"))
            projection.user_present = bool(
                snapshot.vision_connected
                and not snapshot.camera_disconnected
                and snapshot.camera_track_state == "healthy"
            )
            projection.source = event.source
            projection.updated_at = updated_at
        elif event.event_type in {
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.ENGAGEMENT_CHANGED,
            BrainEventType.ATTENTION_CHANGED,
            BrainEventType.SCENE_CHANGED,
        }:
            projection.engagement_state = str(
                payload.get("engagement_state", projection.engagement_state)
            )
            projection.attention_to_camera = str(
                payload.get("attention_to_camera", projection.attention_to_camera)
            )
            projection.user_present = payload.get("person_present") == "present"
            if projection.user_present and projection.engagement_state in {
                "engaged",
                "listening",
                "speaking",
            }:
                projection.last_engaged_at = payload.get("observed_at") or updated_at
            projection.source = event.source
            projection.updated_at = updated_at
        else:
            return
        self._upsert_projection(
            projection_name=ENGAGEMENT_STATE_PROJECTION,
            scope_key=scope_key,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_relationship_state_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        scope_key = str(
            payload.get("presence_scope_key") or payload.get("scope_key") or "local:presence"
        )
        projection = self.get_relationship_state_projection(scope_key=scope_key)
        if event.event_type in {
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.ENGAGEMENT_CHANGED,
            BrainEventType.ATTENTION_CHANGED,
            BrainEventType.SCENE_CHANGED,
        }:
            user_present = payload.get("person_present") == "present"
            projection.user_present = user_present
            if user_present:
                projection.last_seen_at = payload.get("observed_at") or updated_at
            projection.engagement_state = str(
                payload.get("engagement_state", projection.engagement_state)
            )
            projection.attention_to_camera = str(
                payload.get("attention_to_camera", projection.attention_to_camera)
            )
            projection.source = event.source
            projection.updated_at = updated_at
        elif event.event_type == BrainEventType.BODY_STATE_UPDATED:
            snapshot = BrainPresenceSnapshot.from_dict(payload.get("snapshot"))
            projection.user_present = bool(
                snapshot.vision_connected and not snapshot.camera_disconnected
            )
            projection.source = event.source
            projection.updated_at = updated_at
        else:
            return
        self._upsert_projection(
            projection_name=RELATIONSHIP_STATE_PROJECTION,
            scope_key=scope_key,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_working_context_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        projection = self.get_working_context_projection(scope_key=event.thread_id)
        if event.event_type == BrainEventType.USER_TURN_STARTED:
            projection.user_turn_open = True
        elif event.event_type == BrainEventType.USER_TURN_ENDED:
            projection.user_turn_open = False
        elif event.event_type == BrainEventType.USER_TURN_TRANSCRIBED:
            projection.last_user_text = str(payload.get("text", "")).strip() or None
        elif event.event_type == BrainEventType.ASSISTANT_TURN_STARTED:
            projection.assistant_turn_open = True
        elif event.event_type == BrainEventType.ASSISTANT_TURN_ENDED:
            projection.assistant_turn_open = False
            projection.last_assistant_text = str(payload.get("text", "")).strip() or None
        elif event.event_type == BrainEventType.TOOL_CALLED:
            projection.last_tool_name = payload.get("function_name")
        elif event.event_type == BrainEventType.TOOL_COMPLETED:
            projection.last_tool_name = payload.get("function_name") or projection.last_tool_name
            projection.last_tool_result = payload.get("result")
        else:
            return
        projection.updated_at = updated_at
        self._upsert_projection(
            projection_name=WORKING_CONTEXT_PROJECTION,
            scope_key=event.thread_id,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_agenda_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        projection = self.get_agenda_projection(scope_key=event.thread_id)
        if event.event_type == BrainEventType.USER_TURN_TRANSCRIBED:
            projection.agenda_seed = str(payload.get("text", "")).strip() or projection.agenda_seed
        elif event.event_type == BrainEventType.GOAL_CREATED:
            if isinstance(payload.get("goal"), dict):
                goal = BrainGoal.from_dict(payload.get("goal"))
                if goal.goal_id and projection.goal(goal.goal_id) is None:
                    projection.goals.append(goal)
            else:
                title = str(payload.get("title", "")).strip()
                if title and title not in projection.open_goals:
                    projection.open_goals.append(title)
                if title in projection.completed_goals:
                    projection.completed_goals = [
                        goal_title
                        for goal_title in projection.completed_goals
                        if goal_title != title
                    ]
        elif event.event_type == BrainEventType.PLANNING_REQUESTED:
            goal_id = str(payload.get("goal_id", "")).strip()
            goal = projection.goal(goal_id)
            if goal is None:
                return
            goal.status = BrainGoalStatus.PLANNING.value
            goal.planning_requested = True
            goal.updated_at = updated_at
        elif event.event_type in {
            BrainEventType.GOAL_UPDATED,
            BrainEventType.GOAL_DEFERRED,
            BrainEventType.GOAL_RESUMED,
            BrainEventType.GOAL_CANCELLED,
            BrainEventType.GOAL_REPAIRED,
            BrainEventType.GOAL_FAILED,
            BrainEventType.GOAL_COMPLETED,
        } and isinstance(payload.get("goal"), dict):
            goal = BrainGoal.from_dict(payload.get("goal"))
            existing = projection.goal(goal.goal_id)
            if existing is None:
                projection.goals.append(goal)
            else:
                projection.goals = [
                    goal if item.goal_id == goal.goal_id else item for item in projection.goals
                ]
        elif event.event_type == BrainEventType.GOAL_COMPLETED:
            title = str(payload.get("title", "")).strip()
            projection.open_goals = [goal for goal in projection.open_goals if goal != title]
            if title and title not in projection.completed_goals:
                projection.completed_goals.append(title)
        elif event.event_type == BrainEventType.GOAL_CANCELLED:
            title = str(payload.get("title", "")).strip()
            projection.open_goals = [goal for goal in projection.open_goals if goal != title]
            if title and title not in projection.cancelled_goals:
                projection.cancelled_goals.append(title)
        else:
            return
        if projection.goals:
            projection.sync_lists()
        projection.updated_at = updated_at
        self._upsert_projection(
            projection_name=AGENDA_PROJECTION,
            scope_key=event.thread_id,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_autonomy_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        decision_kind = autonomy_decision_kind_for_event_type(event.event_type)
        if decision_kind is None:
            return

        projection = self.get_autonomy_ledger_projection(scope_key=event.thread_id)
        candidate_goal_id = _optional_text(payload.get("candidate_goal_id"))
        reason = _optional_text(payload.get("reason"))
        reason_details = dict(payload.get("reason_details") or {})
        reason_codes = sorted(
            {str(item).strip() for item in payload.get("reason_codes", []) if str(item).strip()}
        )
        executive_policy = (
            dict(payload.get("executive_policy", {}))
            if isinstance(payload.get("executive_policy"), dict)
            else None
        )
        expected_reevaluation_condition = _optional_text(
            payload.get("expected_reevaluation_condition")
        )
        expected_reevaluation_condition_kind = _optional_text(
            payload.get("expected_reevaluation_condition_kind")
        )
        expected_reevaluation_condition_details = dict(
            payload.get("expected_reevaluation_condition_details") or {}
        )
        existing_candidate = projection.candidate(candidate_goal_id)
        summary: str | None = existing_candidate.summary if existing_candidate is not None else None

        if event.event_type == BrainEventType.GOAL_CANDIDATE_CREATED:
            if not isinstance(payload.get("candidate_goal"), dict):
                return
            candidate = BrainCandidateGoal.from_dict(payload.get("candidate_goal"))
            if not candidate.candidate_goal_id:
                return
            projection.current_candidates = [
                item
                for item in projection.current_candidates
                if item.candidate_goal_id != candidate.candidate_goal_id
            ]
            projection.current_candidates.append(candidate)
            candidate_goal_id = candidate.candidate_goal_id
            summary = candidate.summary
        elif event.event_type in {
            BrainEventType.GOAL_CANDIDATE_SUPPRESSED,
            BrainEventType.GOAL_CANDIDATE_MERGED,
            BrainEventType.GOAL_CANDIDATE_ACCEPTED,
            BrainEventType.GOAL_CANDIDATE_EXPIRED,
        }:
            if candidate_goal_id:
                projection.current_candidates = [
                    item
                    for item in projection.current_candidates
                    if item.candidate_goal_id != candidate_goal_id
                ]
        elif event.event_type == BrainEventType.DIRECTOR_NON_ACTION_RECORDED:
            if summary is None and reason is not None:
                summary = reason
            if existing_candidate is not None:
                existing_candidate.expected_reevaluation_condition = expected_reevaluation_condition
                existing_candidate.expected_reevaluation_condition_kind = (
                    expected_reevaluation_condition_kind
                )
                existing_candidate.expected_reevaluation_condition_details = dict(
                    expected_reevaluation_condition_details
                )

        entry = BrainAutonomyLedgerEntry(
            event_id=event.event_id,
            event_type=event.event_type,
            decision_kind=decision_kind,
            candidate_goal_id=candidate_goal_id,
            summary=summary,
            reason=reason,
            reason_details=reason_details,
            reason_codes=reason_codes,
            executive_policy=executive_policy,
            merged_into_candidate_goal_id=_optional_text(
                payload.get("merged_into_candidate_goal_id")
            ),
            accepted_goal_id=_optional_text(payload.get("goal_id")),
            accepted_commitment_id=_optional_text(payload.get("commitment_id")),
            expected_reevaluation_condition=expected_reevaluation_condition,
            expected_reevaluation_condition_kind=expected_reevaluation_condition_kind,
            expected_reevaluation_condition_details=expected_reevaluation_condition_details,
            ts=event.ts,
        )
        projection.recent_entries.append(entry)
        projection.trim_recent_entries()
        projection.updated_at = updated_at
        self._upsert_projection(
            projection_name=AUTONOMY_LEDGER_PROJECTION,
            scope_key=event.thread_id,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_heartbeat_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        projection = self.get_heartbeat_projection(scope_key=event.thread_id)
        projection.last_event_type = event.event_type
        projection.last_event_at = event.ts
        if event.event_type in {BrainEventType.TOOL_CALLED, BrainEventType.TOOL_COMPLETED}:
            projection.last_tool_name = payload.get("function_name")
        if event.event_type in {
            BrainEventType.CAPABILITY_REQUESTED,
            BrainEventType.CAPABILITY_COMPLETED,
            BrainEventType.CAPABILITY_FAILED,
        }:
            projection.last_tool_name = payload.get("capability_id") or projection.last_tool_name
        if event.event_type in {
            BrainEventType.REFLECTION_CYCLE_STARTED,
            BrainEventType.REFLECTION_CYCLE_COMPLETED,
            BrainEventType.REFLECTION_CYCLE_SKIPPED,
            BrainEventType.REFLECTION_CYCLE_FAILED,
        }:
            projection.last_tool_name = payload.get("cycle_id") or projection.last_tool_name
        if event.event_type == BrainEventType.ROBOT_ACTION_OUTCOME:
            projection.last_robot_action = payload.get("action_id")
            projection.warnings = list(payload.get("status", {}).get("warnings", []))
        if event.event_type == BrainEventType.BODY_STATE_UPDATED:
            snapshot = BrainPresenceSnapshot.from_dict(payload.get("snapshot"))
            projection.warnings = list(snapshot.warnings)
        if event.event_type == BrainEventType.MEMORY_HEALTH_REPORTED:
            projection.warnings = [
                finding.get("code", "")
                for finding in payload.get("findings", [])
                if isinstance(finding, dict) and finding.get("severity") in {"warning", "critical"}
            ]
        if event.event_type in {
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.ENGAGEMENT_CHANGED,
            BrainEventType.ATTENTION_CHANGED,
            BrainEventType.SCENE_CHANGED,
        }:
            body_state = self.get_body_state_projection(
                scope_key=str(
                    payload.get("presence_scope_key")
                    or payload.get("scope_key")
                    or "local:presence"
                )
            )
            projection.warnings = list(body_state.warnings)
        projection.updated_at = updated_at
        self._upsert_projection(
            projection_name=HEARTBEAT_PROJECTION,
            scope_key=event.thread_id,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_private_working_memory_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        """Rebuild private working memory from the latest landed thread state."""
        presence_scope_key = str(
            payload.get("presence_scope_key") or payload.get("scope_key") or "local:presence"
        )
        self._refresh_private_working_memory_projection(
            user_id=event.user_id,
            thread_id=event.thread_id,
            source_event_id=event.event_id,
            updated_at=updated_at,
            agent_id=event.agent_id,
            presence_scope_key=presence_scope_key,
            commit=False,
        )

    def _apply_scene_world_state_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        """Rebuild symbolic scene world state from the latest landed presence state."""
        presence_scope_key = _optional_text(
            payload.get("presence_scope_key") or payload.get("scope_key")
        )
        self._refresh_scene_world_state_projection(
            user_id=event.user_id,
            thread_id=event.thread_id,
            source_event_id=event.event_id,
            updated_at=updated_at,
            agent_id=event.agent_id,
            presence_scope_key=presence_scope_key,
            commit=False,
        )

    def _apply_predictive_world_model_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        """Apply explicit predictive lifecycle events to the predictive projection."""
        if event.event_type not in prediction_event_types():
            return
        prediction = BrainPredictionRecord.from_dict(payload.get("prediction"))
        if prediction is None:
            return
        presence_scope_key = str(
            payload.get("presence_scope_key")
            or prediction.presence_scope_key
            or payload.get("scope_key")
            or "local:presence"
        )
        projection = self.get_predictive_world_model_projection(scope_key=event.thread_id)
        if not projection.scope_key:
            projection.scope_key = event.thread_id
        projection.presence_scope_key = presence_scope_key
        if event.event_type in {
            BrainEventType.PREDICTION_CONFIRMED,
            BrainEventType.PREDICTION_INVALIDATED,
            BrainEventType.PREDICTION_EXPIRED,
        }:
            append_prediction_resolution(projection, prediction)
        else:
            append_prediction_generation(projection, prediction)
        projection.updated_at = updated_at
        self._upsert_projection(
            projection_name=PREDICTIVE_WORLD_MODEL_PROJECTION,
            scope_key=event.thread_id,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=projection.updated_at,
            commit=False,
        )

    def _apply_counterfactual_rehearsal_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        """Apply explicit rehearsal lifecycle events to the rehearsal projection."""
        from blink.brain.counterfactuals import (
            append_outcome_comparison,
            append_rehearsal_request,
            append_rehearsal_result,
            counterfactual_event_types,
        )

        if event.event_type not in counterfactual_event_types():
            return
        projection = self.get_counterfactual_rehearsal_projection(scope_key=event.thread_id)
        if not projection.scope_key:
            projection.scope_key = event.thread_id
        presence_scope_key = str(
            payload.get("presence_scope_key")
            or projection.presence_scope_key
            or payload.get("scope_key")
            or "local:presence"
        )
        projection.presence_scope_key = presence_scope_key
        if event.event_type == BrainEventType.ACTION_REHEARSAL_REQUESTED:
            request = BrainActionRehearsalRequest.from_dict(payload.get("rehearsal_request"))
            if request is None:
                return
            append_rehearsal_request(projection, request)
        elif event.event_type in {
            BrainEventType.ACTION_REHEARSAL_COMPLETED,
            BrainEventType.ACTION_REHEARSAL_SKIPPED,
        }:
            result = BrainActionRehearsalResult.from_dict(payload.get("rehearsal_result"))
            if result is None:
                return
            append_rehearsal_result(projection, result)
        elif event.event_type == BrainEventType.ACTION_OUTCOME_COMPARED:
            comparison = BrainActionOutcomeComparisonRecord.from_dict(payload.get("comparison"))
            if comparison is None:
                return
            append_outcome_comparison(projection, comparison)
        else:
            return
        projection.updated_at = updated_at
        self._upsert_projection(
            projection_name=COUNTERFACTUAL_REHEARSAL_PROJECTION,
            scope_key=event.thread_id,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=projection.updated_at,
            commit=False,
        )

    def _apply_embodied_executive_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        """Apply explicit embodied lifecycle events to the embodied projection."""
        if event.event_type not in embodied_event_types():
            return
        projection = self.get_embodied_executive_projection(scope_key=event.thread_id)
        if not projection.scope_key:
            projection.scope_key = event.thread_id
        presence_scope_key = str(
            payload.get("presence_scope_key")
            or projection.presence_scope_key
            or payload.get("scope_key")
            or "local:presence"
        )
        projection.presence_scope_key = presence_scope_key
        intent = BrainEmbodiedIntent.from_dict(payload.get("intent"))
        envelope = BrainEmbodiedActionEnvelope.from_dict(payload.get("envelope"))
        trace = BrainEmbodiedExecutionTrace.from_dict(payload.get("execution_trace"))
        recovery = BrainEmbodiedRecoveryRecord.from_dict(payload.get("recovery"))
        if intent is not None:
            append_embodied_intent(projection, intent)
        if envelope is not None:
            append_embodied_action_envelope(projection, envelope)
        if trace is not None:
            append_embodied_execution_trace(projection, trace)
        if recovery is not None:
            append_embodied_recovery(projection, recovery)
        projection.updated_at = updated_at
        self._upsert_projection(
            projection_name=EMBODIED_EXECUTIVE_PROJECTION,
            scope_key=event.thread_id,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=projection.updated_at,
            commit=False,
        )

    def _apply_practice_director_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        """Apply explicit practice-plan lifecycle events to the practice projection."""
        if event.event_type not in practice_event_types():
            return
        plan = BrainPracticePlan.from_dict(payload.get("practice_plan"))
        if plan is None:
            return
        scope_key = _optional_text(payload.get("scope_key")) or event.thread_id
        projection = self.get_practice_director_projection(scope_key=scope_key)
        if not projection.scope_key:
            projection.scope_key = scope_key
        projection.presence_scope_key = str(
            payload.get("presence_scope_key")
            or plan.presence_scope_key
            or projection.presence_scope_key
            or "local:presence"
        )
        append_practice_plan(projection, plan)
        projection.updated_at = max(projection.updated_at, plan.updated_at, updated_at)
        self._upsert_projection(
            projection_name=PRACTICE_DIRECTOR_PROJECTION,
            scope_key=scope_key,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=projection.updated_at,
            commit=False,
        )

    def _apply_skill_evidence_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        """Apply explicit skill-evidence update events to the evidence projection."""
        if event.event_type != BrainEventType.SKILL_EVIDENCE_UPDATED:
            return
        ledger = BrainSkillEvidenceLedger.from_dict(
            payload.get("skill_evidence_ledger") or payload.get("skill_evidence")
        )
        scope_key = _optional_text(payload.get("scope_key")) or event.thread_id
        if not ledger.scope_id:
            ledger.scope_id = scope_key
        if not ledger.scope_type:
            ledger.scope_type = "thread"
        projection = self.get_skill_evidence_projection(scope_key=scope_key)
        apply_skill_evidence_update(projection, ledger)
        projection.updated_at = max(projection.updated_at, ledger.updated_at, updated_at)
        self._upsert_projection(
            projection_name=SKILL_EVIDENCE_PROJECTION,
            scope_key=scope_key,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=projection.updated_at,
            commit=False,
        )

    def _apply_skill_governance_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        """Apply explicit skill-governance proposal events to the governance projection."""
        if event.event_type not in {
            BrainEventType.SKILL_PROMOTION_PROPOSED,
            BrainEventType.SKILL_PROMOTION_BLOCKED,
            BrainEventType.SKILL_DEMOTION_PROPOSED,
        }:
            return
        scope_key = _optional_text(payload.get("scope_key")) or event.thread_id
        projection = self.get_skill_governance_projection(scope_key=scope_key)
        if not projection.scope_id:
            projection.scope_id = scope_key
        if not projection.scope_type:
            projection.scope_type = "thread"
        if event.event_type in {
            BrainEventType.SKILL_PROMOTION_PROPOSED,
            BrainEventType.SKILL_PROMOTION_BLOCKED,
        }:
            promotion_payload = dict(
                payload.get("promotion_proposal") or payload.get("proposal") or {}
            )
            if event.event_type == BrainEventType.SKILL_PROMOTION_BLOCKED:
                promotion_payload["status"] = BrainSkillGovernanceStatus.BLOCKED.value
                promotion_payload.setdefault("updated_at", updated_at)
            proposal = BrainSkillPromotionProposal.from_dict(promotion_payload)
            if proposal is None:
                return
            append_skill_promotion_proposal(projection, proposal)
            projection.updated_at = max(projection.updated_at, proposal.updated_at, updated_at)
        else:
            demotion_payload = dict(
                payload.get("demotion_proposal") or payload.get("proposal") or {}
            )
            proposal = BrainSkillDemotionProposal.from_dict(demotion_payload)
            if proposal is None:
                return
            append_skill_demotion_proposal(projection, proposal)
            projection.updated_at = max(projection.updated_at, proposal.updated_at, updated_at)
        self._upsert_projection(
            projection_name=SKILL_GOVERNANCE_PROJECTION,
            scope_key=scope_key,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=projection.updated_at,
            commit=False,
        )

    def _apply_adapter_governance_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        """Apply explicit adapter-governance events to the governance projection."""
        if event.event_type not in {
            BrainEventType.ADAPTER_CARD_UPSERTED,
            BrainEventType.ADAPTER_BENCHMARK_REPORTED,
            BrainEventType.ADAPTER_PROMOTION_DECIDED,
            BrainEventType.ADAPTER_ROLLBACK_RECORDED,
        }:
            return
        scope_key = _optional_text(payload.get("scope_key")) or event.thread_id
        projection = self.get_adapter_governance_projection(scope_key=scope_key)
        if not projection.scope_key:
            projection.scope_key = scope_key
        card = BrainAdapterCard.from_dict(payload.get("adapter_card"))
        report = BrainAdapterBenchmarkComparisonReport.from_dict(payload.get("benchmark_report"))
        decision = BrainAdapterPromotionDecision.from_dict(payload.get("promotion_decision"))
        if card is not None:
            append_adapter_card(projection, card)
        if report is not None:
            append_adapter_benchmark_report(projection, report)
            if card is None:
                existing_card = next(
                    (
                        record
                        for record in projection.adapter_cards
                        if record.adapter_family == report.adapter_family
                        and record.backend_id == report.candidate_backend_id
                        and record.backend_version == report.candidate_backend_version
                    ),
                    None,
                )
                if existing_card is not None:
                    append_adapter_card(
                        projection,
                        with_card_benchmark_summary(existing_card, report),
                    )
        if decision is not None:
            append_adapter_promotion_decision(projection, decision)
            if card is None:
                existing_card = next(
                    (
                        record
                        for record in projection.adapter_cards
                        if record.adapter_family == decision.adapter_family
                        and record.backend_id == decision.backend_id
                        and record.backend_version == decision.backend_version
                    ),
                    None,
                )
                if existing_card is not None:
                    append_adapter_card(
                        projection,
                        apply_promotion_decision_to_card(existing_card, decision),
                    )
        if card is None and report is None and decision is None:
            return
        projection.updated_at = max(projection.updated_at, updated_at)
        self._upsert_projection(
            projection_name=ADAPTER_GOVERNANCE_PROJECTION,
            scope_key=scope_key,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=projection.updated_at,
            commit=False,
        )

    def _apply_active_situation_model_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        """Rebuild the active situation model from the latest landed thread state."""
        presence_scope_key = _optional_text(
            payload.get("presence_scope_key") or payload.get("scope_key")
        )
        self._refresh_active_situation_model_projection(
            user_id=event.user_id,
            thread_id=event.thread_id,
            source_event_id=event.event_id,
            updated_at=updated_at,
            agent_id=event.agent_id,
            presence_scope_key=presence_scope_key,
            commit=False,
        )

    def active_tasks(self, *, user_id: str, limit: int = 8) -> list[dict[str, Any]]:
        """Return unresolved tasks backed by durable executive commitments when available."""
        commitment_rows = self.list_executive_commitments(
            user_id=user_id,
            statuses=(
                BrainCommitmentStatus.ACTIVE.value,
                BrainCommitmentStatus.DEFERRED.value,
                BrainCommitmentStatus.BLOCKED.value,
            ),
            limit=limit,
        )
        visible_commitments = [
            record
            for record in commitment_rows
            if record.goal_family == BrainGoalFamily.CONVERSATION.value
            or bool(record.details.get("task_visible"))
        ]
        if visible_commitments:
            return [
                {
                    "commitment_id": record.commitment_id,
                    "title": record.title,
                    "details": record.details,
                    "status": record.status,
                    "goal_family": record.goal_family,
                    "current_goal_id": record.current_goal_id,
                    "blocked_reason": (
                        record.blocked_reason.as_dict() if record.blocked_reason else None
                    ),
                    "wake_conditions": [item.as_dict() for item in record.wake_conditions],
                    "plan_revision": record.plan_revision,
                    "resume_count": record.resume_count,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                }
                for record in visible_commitments[:limit]
            ]

        scope_id = self._relationship_scope_id(
            agent_id=_DEFAULT_RELATIONSHIP_AGENT_ID,
            user_id=user_id,
        )
        commitment_block = self.get_current_core_memory_block(
            block_kind=BrainCoreMemoryBlockKind.ACTIVE_COMMITMENTS.value,
            scope_type="relationship",
            scope_id=scope_id,
        )
        if commitment_block is not None:
            commitments = list(commitment_block.content.get("commitments", []))
            if commitments:
                return commitments[:limit]

        commitments = self.narrative_memories(
            user_id=user_id,
            kinds=("commitment",),
            statuses=("open",),
            limit=limit,
        )
        if commitments:
            return [
                {
                    "title": record.title,
                    "details": record.details,
                    "status": record.status,
                    "created_at": record.observed_at,
                    "updated_at": record.updated_at,
                }
                for record in commitments
            ]

        rows = self._conn.execute(
            """
            SELECT title, details_json, status, created_at, updated_at
            FROM tasks
            WHERE user_id = ? AND status NOT IN ('done', 'cancelled')
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [
            {
                "title": str(row["title"]),
                "details": json.loads(str(row["details_json"])),
                "status": str(row["status"]),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    def upsert_task(
        self,
        *,
        user_id: str,
        title: str,
        details: dict[str, Any] | None = None,
        status: str = "open",
        thread_id: str | None = None,
        source_event_id: str | None = None,
        provenance: dict[str, Any] | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Insert or refresh one unresolved task by title.

        Args:
            user_id: Stable user id that owns the task.
            title: Human-readable task title.
            details: Optional structured task metadata.
            status: Task status, usually `open`.
            thread_id: Optional originating thread id for canonical narrative storage.
            source_event_id: Optional source brain-event id for deduplication and provenance.
            provenance: Optional provenance metadata for the canonical narrative layer.
            agent_id: Optional agent id for continuity block updates.
            session_id: Optional session id for continuity mutation events.

        Returns:
            The persisted task row id.
        """
        normalized_title = " ".join((title or "").split()).strip()
        if not normalized_title:
            raise ValueError("Task title must not be empty")

        relationship_scope_id = self._relationship_scope_id(agent_id=agent_id, user_id=user_id)
        existing_commitment = self.find_executive_commitment(
            scope_type="relationship",
            scope_id=relationship_scope_id,
            goal_family=BrainGoalFamily.CONVERSATION.value,
            intent="narrative.commitment",
            title=normalized_title,
        )
        goal_id = existing_commitment.current_goal_id if existing_commitment is not None else None
        if (
            goal_id is None
            and thread_id is not None
            and agent_id is not None
            and session_id is not None
        ):
            goal = BrainGoal(
                goal_id=uuid4().hex,
                title=normalized_title,
                intent="narrative.commitment",
                source="memory",
                goal_family=BrainGoalFamily.CONVERSATION.value,
                status=BrainGoalStatus.WAITING.value,
                details={
                    **dict(details or {}),
                    "commitment_status": BrainCommitmentStatus.ACTIVE.value,
                },
            )
            created_event = self.append_brain_event(
                event_type=BrainEventType.GOAL_CREATED,
                agent_id=agent_id,
                user_id=user_id,
                session_id=session_id,
                thread_id=thread_id,
                source="memory",
                correlation_id=source_event_id or goal.goal_id,
                causal_parent_id=source_event_id,
                payload={
                    "goal": goal.as_dict(),
                    "commitment": {
                        "commitment_id": _stable_commitment_id(
                            scope_type="relationship",
                            scope_id=relationship_scope_id,
                            goal_family=BrainGoalFamily.CONVERSATION.value,
                            intent="narrative.commitment",
                            title=normalized_title,
                        ),
                        "scope_type": "relationship",
                        "scope_id": relationship_scope_id,
                        "status": BrainCommitmentStatus.ACTIVE.value,
                    },
                },
            )
            goal_id = goal.goal_id
            source_event_id = created_event.event_id

        narrative_record = self.upsert_narrative_memory(
            user_id=user_id,
            thread_id=thread_id or f"tasks:{user_id}",
            kind="commitment",
            title=normalized_title,
            summary=normalized_title,
            details=details or {},
            status=status,
            confidence=0.86,
            source_event_id=source_event_id,
            provenance=provenance,
        )
        self.upsert_executive_commitment(
            scope_type="relationship",
            scope_id=relationship_scope_id,
            user_id=user_id,
            thread_id=thread_id or f"tasks:{user_id}",
            title=normalized_title,
            goal_family=BrainGoalFamily.CONVERSATION.value,
            intent="narrative.commitment",
            status=BrainCommitmentStatus.ACTIVE.value,
            details={
                **dict(details or {}),
                "summary": normalized_title,
            },
            current_goal_id=goal_id,
            plan_revision=existing_commitment.plan_revision
            if existing_commitment is not None
            else 1,
            resume_count=existing_commitment.resume_count if existing_commitment is not None else 0,
            source_event_id=source_event_id,
        )

        now = _utc_now()
        existing = self._conn.execute(
            """
            SELECT id FROM tasks
            WHERE user_id = ? AND title = ? AND status NOT IN ('done', 'cancelled')
            ORDER BY id DESC LIMIT 1
            """,
            (user_id, normalized_title),
        ).fetchone()
        if existing is not None:
            task_id = int(existing["id"])
            self._conn.execute(
                """
                UPDATE tasks
                SET details_json = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                    status,
                    now,
                    task_id,
                ),
            )
            self._conn.commit()
            self._refresh_active_commitments_block(
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                thread_id=thread_id or f"tasks:{user_id}",
                source_event_id=source_event_id,
            )
            if thread_id is not None:
                self._refresh_relationship_core_block(
                    user_id=user_id,
                    thread_id=thread_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    source_event_id=source_event_id,
                    summary=self.get_session_summary(user_id=user_id, thread_id=thread_id),
                )
            return narrative_record.id

        cursor = self._conn.execute(
            """
            INSERT INTO tasks (user_id, title, details_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                normalized_title,
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                status,
                now,
                now,
            ),
        )
        self._conn.commit()
        self._refresh_active_commitments_block(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            thread_id=thread_id or f"tasks:{user_id}",
            source_event_id=source_event_id,
        )
        if thread_id is not None:
            self._refresh_relationship_core_block(
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id,
                session_id=session_id,
                source_event_id=source_event_id,
                summary=self.get_session_summary(user_id=user_id, thread_id=thread_id),
            )
        return narrative_record.id

    def forget_facts(
        self,
        *,
        user_id: str,
        namespace: str,
        subject: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        thread_id: str | None = None,
    ) -> int:
        """Mark active facts as forgotten.

        Args:
            user_id: Stable user id that owns the facts.
            namespace: Fact namespace to forget.
            subject: Optional exact subject filter.
            agent_id: Optional agent id for continuity mutation events.
            session_id: Optional session id for continuity mutation events.
            thread_id: Optional thread id for continuity mutation events.

        Returns:
            Number of facts marked as forgotten.
        """
        semantic_clauses = ["user_id = ?", "namespace = ?", "status = 'active'"]
        semantic_params: list[Any] = [user_id, namespace]
        if subject is not None:
            semantic_clauses.append("subject = ?")
            semantic_params.append(subject)
        semantic_cursor = self._conn.execute(
            f"""
            UPDATE memory_semantic
            SET status = 'forgotten', updated_at = ?
            WHERE {" AND ".join(semantic_clauses)}
            """,
            (_utc_now(), *semantic_params),
        )
        now = _utc_now()
        if subject is None:
            cursor = self._conn.execute(
                """
                UPDATE facts
                SET status = 'forgotten', updated_at = ?
                WHERE user_id = ? AND namespace = ? AND status = 'active'
                """,
                (now, user_id, namespace),
            )
        else:
            cursor = self._conn.execute(
                """
                UPDATE facts
                SET status = 'forgotten', updated_at = ?
                WHERE user_id = ? AND namespace = ? AND subject = ? AND status = 'active'
                """,
                (now, user_id, namespace, subject),
            )
        self._conn.commit()
        if semantic_cursor.rowcount:
            user_entity = self.ensure_entity(
                entity_type="user",
                canonical_name=user_id,
                aliases=[user_id],
                attributes={"user_id": user_id},
            )
            for claim in self.query_claims(
                temporal_mode="current",
                subject_entity_id=user_entity.entity_id,
                predicate=namespace,
                scope_type="user",
                scope_id=user_id,
                limit=24,
            ):
                if (
                    subject is not None
                    and not namespace.startswith("profile.")
                    and str(claim.object.get("value", "")).strip().lower()
                    != str(subject).strip().lower()
                ):
                    continue
                self._claims().revoke_claim(
                    claim.claim_id,
                    reason="forgotten",
                    source_event_id=None,
                    event_context=self._memory_event_context(
                        user_id=user_id,
                        thread_id=thread_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        source="memory_tool",
                    ),
                )
            if namespace in {"profile.name", "profile.role", "profile.origin"}:
                self._refresh_user_core_block(
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    thread_id=thread_id,
                    source_event_id=None,
                )
        if agent_id is not None and thread_id is not None:
            if namespace in _PERSONA_RELATIONSHIP_MEMORY_NAMESPACE_SET:
                self._refresh_relationship_persona_blocks(
                    user_id=user_id,
                    thread_id=thread_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    source_event_id=None,
                )
        return int(max(cursor.rowcount, semantic_cursor.rowcount))

    def update_task_status(
        self,
        *,
        user_id: str,
        title: str,
        status: str,
        thread_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        source_event_id: str | None = None,
    ) -> int:
        """Update task status by exact title for one user.

        Args:
            user_id: Stable user id that owns the task.
            title: Exact human-readable task title.
            status: New task status such as `done` or `cancelled`.
            thread_id: Optional originating thread id for continuity block updates.
            agent_id: Optional agent id for continuity mutation events.
            session_id: Optional session id for continuity mutation events.
            source_event_id: Optional source brain-event id for audit correlation.

        Returns:
            Number of tasks updated.
        """
        normalized_title = " ".join((title or "").split()).strip()
        if not normalized_title:
            raise ValueError("Task title must not be empty")
        relationship_scope_id = self._relationship_scope_id(agent_id=agent_id, user_id=user_id)
        commitment = self.find_executive_commitment(
            scope_type="relationship",
            scope_id=relationship_scope_id,
            goal_family=BrainGoalFamily.CONVERSATION.value,
            intent="narrative.commitment",
            title=normalized_title,
        )
        if commitment is not None:
            commitment_status = {
                "open": BrainCommitmentStatus.ACTIVE.value,
                "active": BrainCommitmentStatus.ACTIVE.value,
                "done": BrainCommitmentStatus.COMPLETED.value,
                "cancelled": BrainCommitmentStatus.CANCELLED.value,
            }.get(status, BrainCommitmentStatus.ACTIVE.value)
            current_goal = None
            if commitment.current_goal_id and thread_id is not None:
                current_goal = self.get_agenda_projection(
                    scope_key=thread_id, user_id=user_id
                ).goal(commitment.current_goal_id)
            updated_goal = (
                BrainGoal.from_dict(current_goal.as_dict()) if current_goal is not None else None
            )
            if updated_goal is not None:
                if commitment_status == BrainCommitmentStatus.CANCELLED.value:
                    updated_goal.status = BrainGoalStatus.CANCELLED.value
                    updated_goal.details["commitment_status"] = (
                        BrainCommitmentStatus.CANCELLED.value
                    )
                elif commitment_status == BrainCommitmentStatus.COMPLETED.value:
                    updated_goal.status = BrainGoalStatus.COMPLETED.value
                    updated_goal.details["commitment_status"] = (
                        BrainCommitmentStatus.COMPLETED.value
                    )
                else:
                    updated_goal.status = BrainGoalStatus.WAITING.value
                    updated_goal.details["commitment_status"] = commitment_status
                updated_goal.updated_at = _utc_now()
                if thread_id is not None and agent_id is not None and session_id is not None:
                    self.append_brain_event(
                        event_type=(
                            BrainEventType.GOAL_CANCELLED
                            if updated_goal.status == BrainGoalStatus.CANCELLED.value
                            else (
                                BrainEventType.GOAL_COMPLETED
                                if updated_goal.status == BrainGoalStatus.COMPLETED.value
                                else BrainEventType.GOAL_UPDATED
                            )
                        ),
                        agent_id=agent_id,
                        user_id=user_id,
                        session_id=session_id,
                        thread_id=thread_id,
                        source="memory",
                        correlation_id=commitment.commitment_id,
                        causal_parent_id=source_event_id,
                        payload={
                            "goal": updated_goal.as_dict(),
                            "commitment": {
                                "commitment_id": commitment.commitment_id,
                                "scope_type": commitment.scope_type,
                                "scope_id": commitment.scope_id,
                                "status": commitment_status,
                            },
                        },
                    )
            self.upsert_executive_commitment(
                commitment_id=commitment.commitment_id,
                scope_type=commitment.scope_type,
                scope_id=commitment.scope_id,
                user_id=user_id,
                thread_id=thread_id or f"tasks:{user_id}",
                title=commitment.title,
                goal_family=commitment.goal_family,
                intent=commitment.intent,
                status=commitment_status,
                details={
                    **commitment.details,
                    "summary": commitment.details.get("summary") or commitment.title,
                },
                current_goal_id=commitment.current_goal_id,
                blocked_reason=None
                if commitment_status != BrainCommitmentStatus.BLOCKED.value
                else commitment.blocked_reason,
                wake_conditions=[]
                if commitment_status
                in {BrainCommitmentStatus.COMPLETED.value, BrainCommitmentStatus.CANCELLED.value}
                else commitment.wake_conditions,
                plan_revision=commitment.plan_revision,
                resume_count=commitment.resume_count,
                source_event_id=source_event_id,
            )

        narrative_cursor = self._conn.execute(
            """
            UPDATE memory_narrative
            SET status = ?, updated_at = ?
            WHERE user_id = ? AND kind = 'commitment' AND title = ?
              AND status NOT IN ('done', 'cancelled', 'superseded')
            """,
            (status, _utc_now(), user_id, normalized_title),
        )
        cursor = self._conn.execute(
            """
            UPDATE tasks
            SET status = ?, updated_at = ?
            WHERE user_id = ? AND title = ? AND status NOT IN ('done', 'cancelled')
            """,
            (status, _utc_now(), user_id, normalized_title),
        )
        self._conn.commit()
        self._refresh_active_commitments_block(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            thread_id=thread_id or f"tasks:{user_id}",
            source_event_id=source_event_id,
        )
        if thread_id is not None:
            self._refresh_relationship_core_block(
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id,
                session_id=session_id,
                source_event_id=source_event_id,
                summary=self.get_session_summary(user_id=user_id, thread_id=thread_id),
            )
        return int(max(cursor.rowcount, narrative_cursor.rowcount))

    def migrate_legacy_local_facts(
        self,
        *,
        user_id: str,
        facts: Iterable[tuple[str, str]],
    ):
        """Import legacy local_brain facts into the typed facts table once."""
        metadata_key = f"legacy_local_brain_migrated:{user_id}"
        migrated = self._conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (metadata_key,),
        ).fetchone()
        if migrated is not None:
            return

        for namespace, rendered_text in facts:
            self.remember_fact(
                user_id=user_id,
                namespace=namespace,
                subject="user",
                value={"text": rendered_text},
                rendered_text=rendered_text,
                confidence=0.65,
                singleton=namespace in {"profile.name", "profile.role", "profile.origin"},
                source_episode_id=None,
            )

        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, '1')",
            (metadata_key,),
        )
        self._conn.commit()
