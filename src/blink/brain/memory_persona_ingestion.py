"""Deterministic curated memory and personality ingestion for Blink."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from blink.brain.identity import load_default_agent_blocks
from blink.brain.memory_layers.semantic import render_preference_fact, render_profile_fact
from blink.brain.persona.behavior_controls import (
    apply_behavior_control_update,
    validate_behavior_control_update_payload,
)
from blink.brain.session import BrainSessionIds
from blink.brain.store import BrainStore

INGESTION_SCHEMA_VERSION = 1
INGESTION_SOURCE = "curated_memory_persona_seed"
_MAX_TEXT_LENGTH = 320
_MAX_CANDIDATES = 200
_PROFILE_FIELDS = {
    "name": "profile.name",
    "role": "profile.role",
    "origin": "profile.origin",
}
_PREFERENCE_FIELDS = {
    "likes": "preference.like",
    "dislikes": "preference.dislike",
}
_RELATIONSHIP_NAMESPACES = {
    "interaction.style": "interaction.style",
    "interaction.preference": "interaction.preference",
    "interaction.misfire": "interaction.misfire",
    "style": "interaction.style",
    "styles": "interaction.style",
    "preference": "interaction.preference",
    "preferences": "interaction.preference",
    "misfire": "interaction.misfire",
    "misfires": "interaction.misfire",
}
_TEACHING_NAMESPACES = {
    "teaching.preference.mode": "teaching.preference.mode",
    "teaching.preference.analogy_domain": "teaching.preference.analogy_domain",
    "teaching.history.helpful_pattern": "teaching.history.helpful_pattern",
    "mode": "teaching.preference.mode",
    "modes": "teaching.preference.mode",
    "analogy_domain": "teaching.preference.analogy_domain",
    "analogy_domains": "teaching.preference.analogy_domain",
    "helpful_pattern": "teaching.history.helpful_pattern",
    "helpful_patterns": "teaching.history.helpful_pattern",
}
_ALLOWED_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "language",
        "user_profile",
        "preferences",
        "relationship_style",
        "teaching_profile",
        "behavior_controls",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "base_url",
        "canonical_name",
        "childhood",
        "consciousness",
        "developer_message",
        "developer_prompt",
        "exclusive",
        "exclusivity",
        "fake_human_identity",
        "family",
        "hardware",
        "hardware_control",
        "human_identity",
        "human_identity_claims_allowed",
        "identity",
        "llm_provider",
        "model",
        "model_id",
        "motor",
        "ontological_status",
        "openai_api_key",
        "persona",
        "persona_prompt",
        "persona_truth",
        "provider",
        "prompt",
        "robot_head",
        "romance",
        "romantic",
        "secret",
        "sentience",
        "servo",
        "sexual",
        "system_message",
        "system_prompt",
    }
)
_UNSAFE_RELATIONSHIP_VALUE_FRAGMENTS = (
    "boyfriend",
    "exclusive",
    "girlfriend",
    "husband",
    "lover",
    "romantic",
    "sexual",
    "wife",
    "专属",
    "恋人",
    "浪漫伴侣",
    "男朋友",
    "女朋友",
)


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalized_key(value: Any) -> str:
    return _normalized_text(value).lower().replace("-", "_").replace(" ", "_")


def _canonical_seed_payload(seed: dict[str, Any]) -> str:
    return json.dumps(seed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def seed_sha256(seed: dict[str, Any]) -> str:
    """Return the stable SHA-256 hash for one parsed seed payload."""
    return hashlib.sha256(_canonical_seed_payload(seed).encode("utf-8")).hexdigest()


def import_id_for_seed(*, seed_hash: str, session_ids: BrainSessionIds) -> str:
    """Return a deterministic import id for one seed/session pair."""
    digest = hashlib.sha256(
        f"{seed_hash}|{session_ids.agent_id}|{session_ids.user_id}".encode("utf-8")
    ).hexdigest()
    return f"memory_persona_import_{digest[:16]}"


def _entry_id(
    *,
    import_id: str,
    namespace: str,
    subject: str,
    summary: str,
) -> str:
    digest = hashlib.sha256(
        f"{import_id}|{namespace}|{subject}|{summary}".encode("utf-8")
    ).hexdigest()
    return f"memory_persona_entry_{digest[:16]}"


def _dedupe(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalized_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _reason_codes(*values: str) -> tuple[str, ...]:
    return _dedupe(values)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _json_public(value: Any) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _json_public(value.as_dict())
    if isinstance(value, dict):
        return {str(key): _json_public(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_public(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _contains_forbidden_key(payload: Any, *, prefix: str = "") -> list["IngestionRejectedEntry"]:
    rejected: list[IngestionRejectedEntry] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = _normalized_key(key)
            path = f"{prefix}.{normalized}" if prefix else normalized
            if normalized in _FORBIDDEN_KEYS:
                rejected.append(
                    IngestionRejectedEntry(
                        path=f"{prefix}.forbidden_seed_key" if prefix else "forbidden_seed_key",
                        fatal=True,
                        reason_codes=_reason_codes(
                            "memory_persona_seed_rejected",
                            "forbidden_seed_key",
                        ),
                    )
                )
                continue
            rejected.extend(_contains_forbidden_key(value, prefix=path))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            rejected.extend(_contains_forbidden_key(item, prefix=f"{prefix}[{index}]"))
    return rejected


def _text_is_safe_for_personality(value: str) -> bool:
    lower = value.lower()
    return not any(fragment in lower for fragment in _UNSAFE_RELATIONSHIP_VALUE_FRAGMENTS)


@dataclass(frozen=True)
class IngestionCandidate:
    """One validated memory/personality seed entry."""

    entry_id: str
    kind: str
    namespace: str
    subject: str
    value: dict[str, str]
    rendered_text: str
    summary: str
    confidence: float
    singleton: bool
    reason_codes: tuple[str, ...]

    def as_public_dict(self) -> dict[str, Any]:
        """Serialize public candidate fields for preview reports."""
        return {
            "entry_id": self.entry_id,
            "kind": self.kind,
            "namespace": self.namespace,
            "subject": self.subject,
            "summary": self.summary,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class IngestionRejectedEntry:
    """One rejected seed entry with public-safe reason codes."""

    path: str
    reason_codes: tuple[str, ...]
    fatal: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialize public-safe rejection details."""
        return {
            "path": self.path,
            "fatal": self.fatal,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class IngestionAppliedEntry:
    """One applied or no-op candidate result."""

    entry_id: str
    kind: str
    namespace: str
    status: str
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize public-safe apply details."""
        return {
            "entry_id": self.entry_id,
            "kind": self.kind,
            "namespace": self.namespace,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BrainMemoryPersonaIngestionReport:
    """Public-safe preview/apply report for curated memory/personality ingestion."""

    schema_version: int
    accepted: bool
    applied: bool
    import_id: str
    seed_sha256: str
    counts: dict[str, int]
    candidates: tuple[IngestionCandidate, ...]
    rejected_entries: tuple[IngestionRejectedEntry, ...]
    applied_entries: tuple[IngestionAppliedEntry, ...]
    behavior_control_result: dict[str, Any] | None
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the report with stable public-safe keys."""
        return {
            "schema_version": self.schema_version,
            "accepted": self.accepted,
            "applied": self.applied,
            "import_id": self.import_id,
            "seed_sha256": self.seed_sha256,
            "counts": dict(self.counts),
            "candidates": [candidate.as_public_dict() for candidate in self.candidates],
            "rejected_entries": [entry.as_dict() for entry in self.rejected_entries],
            "applied_entries": [entry.as_dict() for entry in self.applied_entries],
            "behavior_control_result": _json_public(self.behavior_control_result),
            "reason_codes": list(self.reason_codes),
        }


def _empty_report(
    *,
    accepted: bool,
    applied: bool,
    import_id: str = "",
    seed_hash: str = "",
    rejected_entries: Iterable[IngestionRejectedEntry] = (),
    reason_codes: Iterable[str],
) -> BrainMemoryPersonaIngestionReport:
    rejected = tuple(rejected_entries)
    return BrainMemoryPersonaIngestionReport(
        schema_version=INGESTION_SCHEMA_VERSION,
        accepted=accepted,
        applied=applied,
        import_id=import_id,
        seed_sha256=seed_hash,
        counts={
            "accepted_candidates": 0,
            "rejected_entries": len(rejected),
            "applied_entries": 0,
            "memory_written": 0,
            "memory_noop": 0,
            "behavior_controls_applied": 0,
            "behavior_controls_noop": 0,
        },
        candidates=(),
        rejected_entries=rejected,
        applied_entries=(),
        behavior_control_result=None,
        reason_codes=_reason_codes(*reason_codes),
    )


def rejected_seed_load_report(*reason_codes: str) -> BrainMemoryPersonaIngestionReport:
    """Return a public-safe report for a seed file load failure."""
    return _empty_report(
        accepted=False,
        applied=False,
        rejected_entries=(
            IngestionRejectedEntry(
                path="seed",
                fatal=True,
                reason_codes=_reason_codes("memory_persona_seed_rejected", *reason_codes),
            ),
        ),
        reason_codes=("memory_persona_seed_rejected", *reason_codes),
    )


def _reject(
    rejected: list[IngestionRejectedEntry],
    *,
    path: str,
    fatal: bool = False,
    reason_codes: Iterable[str],
) -> None:
    rejected.append(
        IngestionRejectedEntry(
            path=path,
            fatal=fatal,
            reason_codes=_reason_codes("memory_persona_seed_rejected", *reason_codes),
        )
    )


def _candidate(
    *,
    import_id: str,
    kind: str,
    namespace: str,
    subject: str,
    value: str,
    rendered_text: str,
    confidence: float,
    singleton: bool,
    reason_codes: Iterable[str],
) -> IngestionCandidate:
    summary = rendered_text
    return IngestionCandidate(
        entry_id=_entry_id(
            import_id=import_id,
            namespace=namespace,
            subject=subject,
            summary=summary,
        ),
        kind=kind,
        namespace=namespace,
        subject=subject,
        value={"value": value},
        rendered_text=rendered_text,
        summary=summary,
        confidence=confidence,
        singleton=singleton,
        reason_codes=_reason_codes(*reason_codes),
    )


def _validate_entry_text(
    value: Any,
    *,
    path: str,
    rejected: list[IngestionRejectedEntry],
    personality_value: bool = False,
) -> str | None:
    if isinstance(value, (dict, list, tuple)):
        _reject(rejected, path=path, reason_codes=("entry_must_be_scalar",))
        return None
    text = _normalized_text(value)
    if not text:
        _reject(rejected, path=path, reason_codes=("empty_entry",))
        return None
    if len(text) > _MAX_TEXT_LENGTH:
        _reject(rejected, path=path, reason_codes=("entry_too_long",))
        return None
    if personality_value and not _text_is_safe_for_personality(text):
        _reject(rejected, path=path, fatal=True, reason_codes=("unsafe_personality_value",))
        return None
    return text


def _namespace_entries(
    payload: dict[str, Any],
    *,
    section: str,
    namespaces: dict[str, str],
    rejected: list[IngestionRejectedEntry],
    key_path: tuple[str, ...] = (),
) -> list[tuple[str, str, Any]]:
    entries: list[tuple[str, str, Any]] = []
    for key, raw_values in sorted(payload.items()):
        normalized_key = _normalized_key(key)
        current_path = (*key_path, normalized_key)
        dotted_path = ".".join(current_path)
        namespace = namespaces.get(dotted_path)
        if namespace is None and len(current_path) == 1:
            namespace = namespaces.get(normalized_key)
        if namespace is not None:
            entries.append((dotted_path, namespace, raw_values))
            continue
        if isinstance(raw_values, dict):
            entries.extend(
                _namespace_entries(
                    raw_values,
                    section=section,
                    namespaces=namespaces,
                    rejected=rejected,
                    key_path=current_path,
                )
            )
            continue
        _reject(
            rejected,
            path=f"{section}.{dotted_path}",
            reason_codes=("unsupported_namespace",),
        )
    return entries


def _collect_profile_candidates(
    seed: dict[str, Any],
    *,
    import_id: str,
    rejected: list[IngestionRejectedEntry],
) -> list[IngestionCandidate]:
    payload = seed.get("user_profile")
    if payload in (None, ""):
        return []
    if not isinstance(payload, dict):
        _reject(rejected, path="user_profile", reason_codes=("section_must_be_object",))
        return []

    candidates: list[IngestionCandidate] = []
    for key, raw_value in sorted(payload.items()):
        normalized_key = _normalized_key(key)
        namespace = _PROFILE_FIELDS.get(normalized_key)
        path = f"user_profile.{normalized_key}"
        if namespace is None:
            _reject(rejected, path=path, reason_codes=("unsupported_profile_field",))
            continue
        value = _validate_entry_text(raw_value, path=path, rejected=rejected)
        if value is None:
            continue
        candidates.append(
            _candidate(
                import_id=import_id,
                kind="user_profile",
                namespace=namespace,
                subject="user",
                value=value,
                rendered_text=render_profile_fact(namespace, value),
                confidence=0.92,
                singleton=True,
                reason_codes=("memory_persona_candidate:profile",),
            )
        )
    return candidates


def _collect_preference_candidates(
    seed: dict[str, Any],
    *,
    import_id: str,
    rejected: list[IngestionRejectedEntry],
) -> list[IngestionCandidate]:
    payload = seed.get("preferences")
    if payload in (None, ""):
        return []
    if not isinstance(payload, dict):
        _reject(rejected, path="preferences", reason_codes=("section_must_be_object",))
        return []

    candidates: list[IngestionCandidate] = []
    for key, raw_values in sorted(payload.items()):
        normalized_key = _normalized_key(key)
        namespace = _PREFERENCE_FIELDS.get(normalized_key)
        path = f"preferences.{normalized_key}"
        if namespace is None:
            _reject(rejected, path=path, reason_codes=("unsupported_preference_field",))
            continue
        for index, raw_value in enumerate(_as_list(raw_values)):
            entry_path = f"{path}[{index}]"
            value = _validate_entry_text(raw_value, path=entry_path, rejected=rejected)
            if value is None:
                continue
            candidates.append(
                _candidate(
                    import_id=import_id,
                    kind="preference",
                    namespace=namespace,
                    subject=value.lower(),
                    value=value,
                    rendered_text=render_preference_fact(namespace, value),
                    confidence=0.86,
                    singleton=False,
                    reason_codes=("memory_persona_candidate:preference",),
                )
            )
    return candidates


def _collect_namespace_candidates(
    seed: dict[str, Any],
    *,
    section: str,
    namespaces: dict[str, str],
    import_id: str,
    kind: str,
    confidence: float,
    rejected: list[IngestionRejectedEntry],
) -> list[IngestionCandidate]:
    payload = seed.get(section)
    if payload in (None, ""):
        return []
    if not isinstance(payload, dict):
        _reject(rejected, path=section, reason_codes=("section_must_be_object",))
        return []

    candidates: list[IngestionCandidate] = []
    for dotted_path, namespace, raw_values in _namespace_entries(
        payload,
        section=section,
        namespaces=namespaces,
        rejected=rejected,
    ):
        path = f"{section}.{dotted_path}"
        for index, raw_value in enumerate(_as_list(raw_values)):
            entry_path = f"{path}[{index}]"
            value = _validate_entry_text(
                raw_value,
                path=entry_path,
                rejected=rejected,
                personality_value=True,
            )
            if value is None:
                continue
            candidates.append(
                _candidate(
                    import_id=import_id,
                    kind=kind,
                    namespace=namespace,
                    subject=value.lower(),
                    value=value,
                    rendered_text=value,
                    confidence=confidence,
                    singleton=False,
                    reason_codes=(f"memory_persona_candidate:{kind}",),
                )
            )
    return candidates


def _candidate_counts(
    *,
    candidates: tuple[IngestionCandidate, ...],
    rejected_entries: tuple[IngestionRejectedEntry, ...],
    applied_entries: tuple[IngestionAppliedEntry, ...] = (),
    behavior_control_result: dict[str, Any] | None = None,
) -> dict[str, int]:
    memory_written = sum(1 for entry in applied_entries if entry.status == "written")
    memory_noop = sum(1 for entry in applied_entries if entry.status == "noop")
    behavior_applied = 0
    behavior_noop = 0
    if behavior_control_result:
        if behavior_control_result.get("applied") is True:
            behavior_applied = 1
        elif behavior_control_result.get("accepted") is True:
            behavior_noop = 1
    return {
        "accepted_candidates": len(candidates),
        "rejected_entries": len(rejected_entries),
        "applied_entries": len(applied_entries),
        "memory_written": memory_written,
        "memory_noop": memory_noop,
        "behavior_controls_applied": behavior_applied,
        "behavior_controls_noop": behavior_noop,
    }


def build_memory_persona_ingestion_preview(
    seed: dict[str, Any],
    *,
    session_ids: BrainSessionIds,
) -> BrainMemoryPersonaIngestionReport:
    """Validate one curated seed and return a public-safe dry-run report."""
    if not isinstance(seed, dict):
        return rejected_seed_load_report("seed_payload_must_be_object")

    seed_hash = seed_sha256(seed)
    import_id = import_id_for_seed(seed_hash=seed_hash, session_ids=session_ids)
    rejected: list[IngestionRejectedEntry] = _contains_forbidden_key(seed)

    schema_version = seed.get("schema_version")
    if schema_version != INGESTION_SCHEMA_VERSION:
        _reject(
            rejected,
            path="schema_version",
            fatal=True,
            reason_codes=("unsupported_schema_version",),
        )

    for key in sorted(seed):
        normalized_key = _normalized_key(key)
        if normalized_key not in _ALLOWED_TOP_LEVEL and normalized_key not in _FORBIDDEN_KEYS:
            _reject(
                rejected,
                path=normalized_key,
                reason_codes=("unsupported_seed_section",),
            )

    candidates = [
        *_collect_profile_candidates(seed, import_id=import_id, rejected=rejected),
        *_collect_preference_candidates(seed, import_id=import_id, rejected=rejected),
        *_collect_namespace_candidates(
            seed,
            section="relationship_style",
            namespaces=_RELATIONSHIP_NAMESPACES,
            import_id=import_id,
            kind="relationship_style",
            confidence=0.82,
            rejected=rejected,
        ),
        *_collect_namespace_candidates(
            seed,
            section="teaching_profile",
            namespaces=_TEACHING_NAMESPACES,
            import_id=import_id,
            kind="teaching_profile",
            confidence=0.82,
            rejected=rejected,
        ),
    ]
    behavior_controls = seed.get("behavior_controls")
    if behavior_controls not in (None, ""):
        normalized_controls, rejected_fields = validate_behavior_control_update_payload(
            behavior_controls if isinstance(behavior_controls, dict) else None
        )
        if rejected_fields:
            for field in rejected_fields:
                forbidden = field in _FORBIDDEN_KEYS
                _reject(
                    rejected,
                    path="behavior_controls.forbidden_field"
                    if forbidden
                    else f"behavior_controls.{field}",
                    fatal=forbidden,
                    reason_codes=("behavior_controls_fields_invalid",),
                )
        elif normalized_controls:
            summary = "behavior controls update"
            candidates.append(
                IngestionCandidate(
                    entry_id=_entry_id(
                        import_id=import_id,
                        namespace="behavior_controls",
                        subject=session_ids.user_id,
                        summary=summary,
                    ),
                    kind="behavior_controls",
                    namespace="behavior_controls",
                    subject=session_ids.user_id,
                    value={key: value for key, value in sorted(normalized_controls.items())},
                    rendered_text=summary,
                    summary=summary,
                    confidence=1.0,
                    singleton=True,
                    reason_codes=_reason_codes("memory_persona_candidate:behavior_controls"),
                )
            )
        else:
            _reject(
                rejected,
                path="behavior_controls",
                reason_codes=("behavior_controls_empty",),
            )

    if len(candidates) > _MAX_CANDIDATES:
        _reject(
            rejected,
            path="seed",
            fatal=True,
            reason_codes=("too_many_candidates",),
        )
        candidates = candidates[:_MAX_CANDIDATES]

    candidate_tuple = tuple(candidates)
    rejected_tuple = tuple(rejected)
    has_fatal = any(entry.fatal for entry in rejected_tuple)
    accepted = bool(candidate_tuple) and not has_fatal
    reason_codes = ["memory_persona_preview_built"]
    if accepted:
        reason_codes.append("memory_persona_preview_accepted")
    else:
        reason_codes.append("memory_persona_preview_rejected")
    if rejected_tuple:
        reason_codes.append("memory_persona_preview_has_rejections")

    return BrainMemoryPersonaIngestionReport(
        schema_version=INGESTION_SCHEMA_VERSION,
        accepted=accepted,
        applied=False,
        import_id=import_id,
        seed_sha256=seed_hash,
        counts=_candidate_counts(candidates=candidate_tuple, rejected_entries=rejected_tuple),
        candidates=candidate_tuple,
        rejected_entries=rejected_tuple,
        applied_entries=(),
        behavior_control_result=None,
        reason_codes=_reason_codes(*reason_codes),
    )


def _approved_report_matches(
    *,
    approved_report: dict[str, Any] | None,
    preview: BrainMemoryPersonaIngestionReport,
) -> tuple[bool, tuple[str, ...]]:
    if not isinstance(approved_report, dict):
        return False, ("approved_report_missing",)
    if approved_report.get("schema_version") != INGESTION_SCHEMA_VERSION:
        return False, ("approved_report_schema_mismatch",)
    if approved_report.get("accepted") is not True:
        return False, ("approved_report_not_accepted",)
    if approved_report.get("applied") is not False:
        return False, ("approved_report_not_preview",)
    if approved_report.get("seed_sha256") != preview.seed_sha256:
        return False, ("approved_report_seed_mismatch",)
    if approved_report.get("import_id") != preview.import_id:
        return False, ("approved_report_import_mismatch",)
    return True, ()


def _memory_candidate_exists(
    *,
    store: BrainStore,
    user_id: str,
    candidate: IngestionCandidate,
) -> bool:
    records = store.semantic_memories(
        user_id=user_id,
        namespaces=(candidate.namespace,),
        limit=512,
        include_inactive=False,
    )
    return any(
        record.subject == candidate.subject
        and record.rendered_text == candidate.rendered_text
        and record.status == "active"
        for record in records
    )


def _apply_memory_candidate(
    *,
    store: BrainStore,
    session_ids: BrainSessionIds,
    import_id: str,
    candidate: IngestionCandidate,
) -> IngestionAppliedEntry:
    if _memory_candidate_exists(store=store, user_id=session_ids.user_id, candidate=candidate):
        return IngestionAppliedEntry(
            entry_id=candidate.entry_id,
            kind=candidate.kind,
            namespace=candidate.namespace,
            status="noop",
            reason_codes=_reason_codes("memory_persona_entry_noop", "memory_already_current"),
        )

    record = store.remember_fact(
        user_id=session_ids.user_id,
        namespace=candidate.namespace,
        subject=candidate.subject,
        value=candidate.value,
        rendered_text=candidate.rendered_text,
        confidence=candidate.confidence,
        singleton=candidate.singleton,
        source_episode_id=None,
        provenance={
            "source": INGESTION_SOURCE,
            "import_id": import_id,
            "entry_id": candidate.entry_id,
        },
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    if candidate.namespace == "profile.name":
        store.ensure_user(user_id=session_ids.user_id, display_name=candidate.value["value"])
    return IngestionAppliedEntry(
        entry_id=candidate.entry_id,
        kind=candidate.kind,
        namespace=candidate.namespace,
        status="written",
        reason_codes=_reason_codes("memory_persona_entry_written", f"fact_status:{record.status}"),
    )


def apply_memory_persona_ingestion(
    *,
    store: BrainStore,
    seed: dict[str, Any],
    session_ids: BrainSessionIds,
    approved_report: dict[str, Any] | None,
) -> BrainMemoryPersonaIngestionReport:
    """Apply a curated seed only when a matching accepted preview report is supplied."""
    preview = build_memory_persona_ingestion_preview(seed, session_ids=session_ids)
    matches, mismatch_reasons = _approved_report_matches(
        approved_report=approved_report,
        preview=preview,
    )
    if not preview.accepted or not matches:
        reasons = [
            "memory_persona_apply_rejected",
            *preview.reason_codes,
            *mismatch_reasons,
        ]
        return BrainMemoryPersonaIngestionReport(
            schema_version=INGESTION_SCHEMA_VERSION,
            accepted=False,
            applied=False,
            import_id=preview.import_id,
            seed_sha256=preview.seed_sha256,
            counts=preview.counts,
            candidates=preview.candidates,
            rejected_entries=preview.rejected_entries,
            applied_entries=(),
            behavior_control_result=None,
            reason_codes=_reason_codes(*reasons),
        )

    store.ensure_default_blocks(load_default_agent_blocks())
    store.ensure_user(user_id=session_ids.user_id, language=str(seed.get("language") or "") or None)
    applied_entries: list[IngestionAppliedEntry] = []
    behavior_control_result: dict[str, Any] | None = None

    for candidate in preview.candidates:
        if candidate.kind == "behavior_controls":
            result = apply_behavior_control_update(
                store=store,
                session_ids=session_ids,
                updates=candidate.value,
                source=INGESTION_SOURCE,
            )
            behavior_control_result = result.as_dict()
            continue
        applied_entries.append(
            _apply_memory_candidate(
                store=store,
                session_ids=session_ids,
                import_id=preview.import_id,
                candidate=candidate,
            )
        )

    applied_tuple = tuple(applied_entries)
    counts = _candidate_counts(
        candidates=preview.candidates,
        rejected_entries=preview.rejected_entries,
        applied_entries=applied_tuple,
        behavior_control_result=behavior_control_result,
    )
    applied = bool(
        counts["memory_written"] > 0 or counts["behavior_controls_applied"] > 0
    )
    reason_codes = ["memory_persona_apply_accepted"]
    if applied:
        reason_codes.append("memory_persona_apply_persisted")
    else:
        reason_codes.append("memory_persona_apply_noop")

    return BrainMemoryPersonaIngestionReport(
        schema_version=INGESTION_SCHEMA_VERSION,
        accepted=True,
        applied=applied,
        import_id=preview.import_id,
        seed_sha256=preview.seed_sha256,
        counts=counts,
        candidates=preview.candidates,
        rejected_entries=preview.rejected_entries,
        applied_entries=applied_tuple,
        behavior_control_result=behavior_control_result,
        reason_codes=_reason_codes(*reason_codes),
    )


__all__ = [
    "BrainMemoryPersonaIngestionReport",
    "INGESTION_SCHEMA_VERSION",
    "INGESTION_SOURCE",
    "IngestionAppliedEntry",
    "IngestionCandidate",
    "IngestionRejectedEntry",
    "apply_memory_persona_ingestion",
    "build_memory_persona_ingestion_preview",
    "import_id_for_seed",
    "rejected_seed_load_report",
    "seed_sha256",
]
