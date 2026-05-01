"""Privacy-safe performance preference pairs and policy proposals."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

PERFORMANCE_PREFERENCE_SCHEMA_VERSION = 3
PERFORMANCE_LEARNING_POLICY_SCHEMA_VERSION = 3
PERFORMANCE_PREFERENCE_ARTIFACT_DIR = Path("artifacts/performance_preferences_v3")
PERFORMANCE_PREFERENCE_DIMENSIONS = (
    "felt_heard",
    "state_clarity",
    "interruption_naturalness",
    "voice_pacing",
    "camera_honesty",
    "memory_usefulness",
    "persona_consistency",
    "enjoyment",
    "not_fake_human",
)
PERFORMANCE_PREFERENCE_WINNERS = ("a", "b", "same", "neither")
PERFORMANCE_PREFERENCE_PROFILES = ("browser-zh-melo", "browser-en-kokoro", "cross_profile")
PERFORMANCE_PREFERENCE_LANGUAGES = ("zh", "en", "mixed")
PERFORMANCE_PREFERENCE_TTS_LABELS = (
    "local-http-wav/MeloTTS",
    "kokoro/English",
    "mixed",
)
PERFORMANCE_LEARNING_PROPOSAL_TARGETS = (
    "persona_anchor_selection",
    "speech_chunking_bias",
    "memory_callback_threshold",
    "interruption_backchannel_policy",
)

_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
_HASH_PREFIX_LENGTH = 16
_MAX_LIST_ITEMS = 24
_MAX_REASON_CODES = 32
_MAX_LABELS = 24
_MAX_SUMMARY_CHARS = 180
_UNSAFE_KEY_EXACT = {
    "audio",
    "audio_bytes",
    "authorization",
    "candidate",
    "content",
    "credential",
    "credentials",
    "hidden_prompt",
    "ice",
    "ice_candidate",
    "image",
    "image_bytes",
    "messages",
    "notes",
    "password",
    "prompt",
    "raw",
    "raw_audio",
    "raw_image",
    "raw_memory",
    "raw_transcript",
    "response_text",
    "sdp",
    "secret",
    "system_prompt",
    "text",
    "token",
    "transcript",
    "url",
}
_UNSAFE_KEY_FRAGMENTS = {
    "authorization",
    "audio",
    "candidate",
    "credential",
    "hidden",
    "image",
    "message",
    "password",
    "prompt",
    "raw",
    "sdp",
    "secret",
    "source_ref",
    "transcript",
    "url",
}
_SAFE_KEY_EXACT = {
    "candidate_a",
    "candidate_b",
    "candidate_id",
    "candidate_kind",
    "candidate_label",
    "camera_honesty_states",
    "control_frame_ids",
    "created_at",
    "dimension",
    "evidence_id",
    "evidence_kind",
    "evidence_refs",
    "failure_labels",
    "improvement_labels",
    "language",
    "metric_counts",
    "pair_id",
    "plan_ids",
    "policy_labels",
    "profile",
    "public_summary",
    "ratings",
    "reason_codes",
    "schema_version",
    "segment_counts",
    "source_pair_ids",
    "summary_hash",
    "tts_runtime_label",
    "winner",
}
_SAFE_KEY_SUFFIXES = (
    "_count",
    "_counts",
    "_hash",
    "_id",
    "_ids",
    "_kind",
    "_label",
    "_labels",
    "_ms",
    "_ref",
    "_refs",
    "_state",
    "_states",
)
_UNSAFE_VALUE_TOKENS = (
    "-----begin",
    "[blink_brain_context]",
    "a=candidate",
    "authorization:",
    "base64,",
    "bearer ",
    "data:audio",
    "data:image",
    "developer prompt",
    "hidden prompt",
    "ice-ufrag",
    "m=audio",
    "m=video",
    "o=-",
    "password",
    "raw audio",
    "raw image",
    "secret",
    "sk-",
    "system prompt",
    "token",
    "traceback",
    "v=0",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _safe_token(value: Any, *, default: str = "unknown", limit: int = 96) -> str:
    text = _TOKEN_RE.sub("_", _text(value))
    text = "_".join(part for part in text.split("_") if part)
    return text[:limit] or default


def _safe_text(value: Any, *, limit: int = 120) -> str:
    text = _text(value)
    if not text:
        return ""
    lowered = text.lower()
    if any(token in lowered for token in _UNSAFE_VALUE_TOKENS):
        return "redacted"
    return text[:limit]


def _stable_id(prefix: str, payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{uuid5(NAMESPACE_URL, encoded).hex}"


def _hash_text(value: Any) -> str:
    import hashlib

    return hashlib.sha256(_text(value).encode("utf-8")).hexdigest()[:_HASH_PREFIX_LENGTH]


def _dedupe(values: Iterable[Any], *, limit: int = _MAX_LABELS) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _safe_token(value, default="", limit=96)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return tuple(result)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _profile_for(value: Any, *, language: str = "mixed", tts_label: str = "mixed") -> str:
    profile = _safe_token(value, default="", limit=96)
    if profile in PERFORMANCE_PREFERENCE_PROFILES:
        return profile
    if language == "zh" or tts_label == "local-http-wav/MeloTTS":
        return "browser-zh-melo"
    if language == "en" or tts_label == "kokoro/English":
        return "browser-en-kokoro"
    return "cross_profile"


def _language_for(value: Any, *, profile: str = "cross_profile") -> str:
    language = _safe_token(value, default="", limit=16)
    if language in PERFORMANCE_PREFERENCE_LANGUAGES:
        return language
    if profile == "browser-zh-melo":
        return "zh"
    if profile == "browser-en-kokoro":
        return "en"
    return "mixed"


def _tts_label_for(value: Any, *, profile: str = "cross_profile", language: str = "mixed") -> str:
    label = _safe_text(value, limit=96)
    if label in PERFORMANCE_PREFERENCE_TTS_LABELS:
        return label
    if profile == "browser-zh-melo" or language == "zh":
        return "local-http-wav/MeloTTS"
    if profile == "browser-en-kokoro" or language == "en":
        return "kokoro/English"
    return "mixed"


def _safe_key(key: Any) -> str:
    return _safe_token(key, default="", limit=96)


def _key_is_unsafe(key: Any) -> bool:
    normalized = _safe_key(key).lower()
    if not normalized:
        return True
    if normalized in _SAFE_KEY_EXACT or normalized.endswith(_SAFE_KEY_SUFFIXES):
        return False
    if normalized in _UNSAFE_KEY_EXACT:
        return True
    return any(fragment in normalized for fragment in _UNSAFE_KEY_FRAGMENTS)


def _value_has_unsafe_token(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith("candidate:") and (
            " udp " in lowered or " tcp " in lowered or " typ " in lowered
        ):
            return True
        return any(token in lowered for token in _UNSAFE_VALUE_TOKENS)
    if isinstance(value, dict):
        return any(_value_has_unsafe_token(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_value_has_unsafe_token(item) for item in value)
    return False


@dataclass(frozen=True)
class PerformancePreferenceSanitizerReport:
    """Public-safe sanitizer result for preference artifacts."""

    accepted: bool = True
    blocked_keys: tuple[str, ...] = ()
    blocked_values: tuple[str, ...] = ()
    omitted_keys: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable sanitizer payload."""
        return {
            "accepted": self.accepted,
            "blocked_keys": list(self.blocked_keys),
            "blocked_values": list(self.blocked_values),
            "omitted_keys": list(self.omitted_keys),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class PerformancePreferenceCandidateRef:
    """One public-safe candidate reference in a pairwise preference."""

    candidate_id: str
    candidate_kind: str = "build_trace"
    profile: str = "cross_profile"
    language: str = "mixed"
    tts_runtime_label: str = "mixed"
    candidate_label: str = ""
    episode_ids: tuple[str, ...] = ()
    plan_ids: tuple[str, ...] = ()
    control_frame_ids: tuple[str, ...] = ()
    public_summary: str = ""
    summary_hash: str = ""
    segment_counts: dict[str, int] = field(default_factory=dict)
    metric_counts: dict[str, int] = field(default_factory=dict)
    policy_labels: tuple[str, ...] = ()
    camera_honesty_states: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate reference."""
        return {
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "profile": self.profile,
            "language": self.language,
            "tts_runtime_label": self.tts_runtime_label,
            "candidate_label": self.candidate_label,
            "episode_ids": list(self.episode_ids),
            "plan_ids": list(self.plan_ids),
            "control_frame_ids": list(self.control_frame_ids),
            "public_summary": self.public_summary,
            "summary_hash": self.summary_hash,
            "segment_counts": dict(sorted(self.segment_counts.items())),
            "metric_counts": dict(sorted(self.metric_counts.items())),
            "policy_labels": list(self.policy_labels),
            "camera_honesty_states": list(self.camera_honesty_states),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class PerformancePreferenceEvidenceRef:
    """One public-safe evidence reference for a preference pair."""

    evidence_kind: str
    evidence_id: str
    summary: str = ""
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence reference."""
        return {
            "evidence_kind": self.evidence_kind,
            "evidence_id": self.evidence_id,
            "summary": self.summary,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class PerformancePreferencePair:
    """One local pairwise performance preference record."""

    pair_id: str
    profile: str
    language: str
    tts_runtime_label: str
    candidate_a: PerformancePreferenceCandidateRef
    candidate_b: PerformancePreferenceCandidateRef
    winner: str
    ratings: dict[str, int]
    created_at: str = field(default_factory=_utc_now)
    improvement_labels: tuple[str, ...] = ()
    failure_labels: tuple[str, ...] = ()
    evidence_refs: tuple[PerformancePreferenceEvidenceRef, ...] = ()
    sanitizer: PerformancePreferenceSanitizerReport = field(
        default_factory=PerformancePreferenceSanitizerReport
    )
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable preference pair."""
        return {
            "schema_version": PERFORMANCE_PREFERENCE_SCHEMA_VERSION,
            "pair_id": self.pair_id,
            "created_at": self.created_at,
            "profile": self.profile,
            "language": self.language,
            "tts_runtime_label": self.tts_runtime_label,
            "candidate_a": self.candidate_a.as_dict(),
            "candidate_b": self.candidate_b.as_dict(),
            "winner": self.winner,
            "ratings": {key: int(self.ratings[key]) for key in PERFORMANCE_PREFERENCE_DIMENSIONS},
            "improvement_labels": list(self.improvement_labels),
            "failure_labels": list(self.failure_labels),
            "evidence_refs": [ref.as_dict() for ref in self.evidence_refs],
            "sanitizer": self.sanitizer.as_dict(),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class PerformanceLearningPolicyProposalV3:
    """One evidence-cited proposal derived from preference pairs."""

    proposal_id: str
    created_at: str
    status: str
    target: str
    summary: str
    behavior_control_updates: dict[str, str]
    source_pair_ids: tuple[str, ...]
    evidence_refs: tuple[PerformancePreferenceEvidenceRef, ...] = ()
    dimension_scores: dict[str, float] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable policy proposal."""
        return {
            "schema_version": PERFORMANCE_LEARNING_POLICY_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "created_at": self.created_at,
            "status": self.status,
            "target": self.target,
            "summary": self.summary,
            "behavior_control_updates": dict(sorted(self.behavior_control_updates.items())),
            "source_pair_ids": list(self.source_pair_ids),
            "evidence_refs": [ref.as_dict() for ref in self.evidence_refs],
            "dimension_scores": dict(sorted(self.dimension_scores.items())),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PerformanceLearningPolicyProposalV3 | None":
        """Build a proposal from a JSON-like payload."""
        if not isinstance(payload, dict) or _safe_int(payload.get("schema_version")) != 3:
            return None
        updates = {
            _safe_token(key, default="", limit=64): _safe_token(value, default="", limit=64)
            for key, value in dict(payload.get("behavior_control_updates") or {}).items()
        }
        updates = {key: value for key, value in updates.items() if key and value}
        return cls(
            proposal_id=_safe_token(payload.get("proposal_id"), default="", limit=120),
            created_at=_safe_text(payload.get("created_at"), limit=96) or _utc_now(),
            status=_safe_token(payload.get("status"), default="proposed", limit=32),
            target=_safe_token(payload.get("target"), default="persona_anchor_selection", limit=80),
            summary=_safe_text(payload.get("summary"), limit=180),
            behavior_control_updates=updates,
            source_pair_ids=_dedupe(payload.get("source_pair_ids") or (), limit=16),
            evidence_refs=tuple(
                ref
                for item in list(payload.get("evidence_refs") or [])[:_MAX_LIST_ITEMS]
                if (ref := _evidence_from_payload(item)) is not None
            ),
            dimension_scores={
                dimension: _safe_float(dict(payload.get("dimension_scores") or {}).get(dimension))
                for dimension in PERFORMANCE_PREFERENCE_DIMENSIONS
                if dimension in dict(payload.get("dimension_scores") or {})
            },
            reason_codes=_dedupe(payload.get("reason_codes") or (), limit=_MAX_REASON_CODES),
        )


def _safe_count_mapping(value: Any, *, max_items: int = 16) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    rows: dict[str, int] = {}
    for key, count in list(value.items())[:max_items]:
        safe_key = _safe_token(key, default="", limit=80)
        if safe_key:
            rows[safe_key] = max(0, _safe_int(count))
    return dict(sorted(rows.items()))


def _candidate_from_payload(payload: Any, *, fallback_id: str) -> PerformancePreferenceCandidateRef:
    data = dict(payload) if isinstance(payload, dict) else {}
    profile = _profile_for(data.get("profile"), language=_text(data.get("language")))
    language = _language_for(data.get("language"), profile=profile)
    tts_label = _tts_label_for(data.get("tts_runtime_label"), profile=profile, language=language)
    public_summary = _safe_text(data.get("public_summary"), limit=_MAX_SUMMARY_CHARS)
    return PerformancePreferenceCandidateRef(
        candidate_id=_safe_token(data.get("candidate_id"), default=fallback_id, limit=120),
        candidate_kind=_safe_token(data.get("candidate_kind"), default="build_trace", limit=64),
        profile=profile,
        language=language,
        tts_runtime_label=tts_label,
        candidate_label=_safe_text(data.get("candidate_label"), limit=80),
        episode_ids=_dedupe(data.get("episode_ids") or (), limit=12),
        plan_ids=_dedupe(data.get("plan_ids") or (), limit=12),
        control_frame_ids=_dedupe(data.get("control_frame_ids") or (), limit=12),
        public_summary=public_summary,
        summary_hash=_safe_token(data.get("summary_hash"), default="", limit=64)
        or (_hash_text(public_summary) if public_summary else ""),
        segment_counts=_safe_count_mapping(data.get("segment_counts")),
        metric_counts=_safe_count_mapping(data.get("metric_counts")),
        policy_labels=_dedupe(data.get("policy_labels") or (), limit=16),
        camera_honesty_states=_dedupe(data.get("camera_honesty_states") or (), limit=8),
        reason_codes=_dedupe(data.get("reason_codes") or (), limit=_MAX_REASON_CODES),
    )


def _evidence_from_payload(payload: Any) -> PerformancePreferenceEvidenceRef | None:
    data = dict(payload) if isinstance(payload, dict) else {}
    evidence_id = _safe_token(data.get("evidence_id"), default="", limit=120)
    evidence_kind = _safe_token(data.get("evidence_kind"), default="", limit=80)
    if not evidence_id or not evidence_kind:
        return None
    return PerformancePreferenceEvidenceRef(
        evidence_kind=evidence_kind,
        evidence_id=evidence_id,
        summary=_safe_text(data.get("summary"), limit=160),
        reason_codes=_dedupe(data.get("reason_codes") or (), limit=12),
    )


def _ratings_from_payload(payload: Any) -> dict[str, int]:
    data = dict(payload) if isinstance(payload, dict) else {}
    ratings: dict[str, int] = {}
    for dimension in PERFORMANCE_PREFERENCE_DIMENSIONS:
        score = _safe_int(data.get(dimension), default=3)
        ratings[dimension] = max(1, min(5, score))
    return ratings


def _safety_report(payload: Any) -> PerformancePreferenceSanitizerReport:
    blocked_keys: list[str] = []
    blocked_values: list[str] = []

    def walk(value: Any, *, depth: int = 0):
        if depth > 7:
            return
        if isinstance(value, dict):
            for raw_key, nested in value.items():
                safe_key = _safe_key(raw_key)
                if _key_is_unsafe(raw_key):
                    blocked_keys.append(safe_key or "unsafe_key")
                walk(nested, depth=depth + 1)
            return
        if isinstance(value, (list, tuple, set)):
            for item in list(value)[:_MAX_LIST_ITEMS]:
                walk(item, depth=depth + 1)
            return
        if _value_has_unsafe_token(value):
            blocked_values.append("unsafe_value")

    walk(payload)
    accepted = not blocked_keys and not blocked_values
    reason_codes = [
        "performance_preference_sanitizer:v3",
        "performance_preference:safe" if accepted else "performance_preference:unsafe_payload",
    ]
    if blocked_keys:
        reason_codes.append("performance_preference:unsafe_key_present")
    if blocked_values:
        reason_codes.append("performance_preference:unsafe_value_present")
    return PerformancePreferenceSanitizerReport(
        accepted=accepted,
        blocked_keys=_dedupe(blocked_keys, limit=16),
        blocked_values=_dedupe(blocked_values, limit=16),
        reason_codes=tuple(reason_codes),
    )


def build_performance_preference_pair(
    payload: dict[str, Any],
    *,
    created_at: str | None = None,
) -> tuple[PerformancePreferencePair | None, PerformancePreferenceSanitizerReport]:
    """Build one sanitized preference pair from a JSON-like payload."""
    report = _safety_report(payload)
    if not report.accepted:
        return None, report
    candidate_a = _candidate_from_payload(payload.get("candidate_a"), fallback_id="candidate-a")
    candidate_b = _candidate_from_payload(payload.get("candidate_b"), fallback_id="candidate-b")
    language = _language_for(payload.get("language"), profile=_text(payload.get("profile")))
    tts_label = _tts_label_for(payload.get("tts_runtime_label"), language=language)
    profile = _profile_for(payload.get("profile"), language=language, tts_label=tts_label)
    if candidate_a.profile != candidate_b.profile:
        profile = "cross_profile"
    if candidate_a.language != candidate_b.language:
        language = "mixed"
    if candidate_a.tts_runtime_label != candidate_b.tts_runtime_label:
        tts_label = "mixed"
    winner = _safe_token(payload.get("winner"), default="same", limit=16)
    if winner not in PERFORMANCE_PREFERENCE_WINNERS:
        winner = "same"
    ratings = _ratings_from_payload(payload.get("ratings"))
    evidence_refs = tuple(
        ref
        for item in list(payload.get("evidence_refs") or [])[:_MAX_LIST_ITEMS]
        if (ref := _evidence_from_payload(item)) is not None
    )
    base_payload = {
        "profile": profile,
        "language": language,
        "tts_runtime_label": tts_label,
        "candidate_a": candidate_a.as_dict(),
        "candidate_b": candidate_b.as_dict(),
        "winner": winner,
        "ratings": ratings,
        "improvement_labels": list(_dedupe(payload.get("improvement_labels") or ())),
        "failure_labels": list(_dedupe(payload.get("failure_labels") or ())),
        "evidence_refs": [ref.as_dict() for ref in evidence_refs],
    }
    pair_id = _safe_token(payload.get("pair_id"), default="", limit=120)
    if not pair_id:
        pair_id = _stable_id("performance-pref-pair", base_payload)
    reason_codes = _dedupe(
        (
            "performance_preference_pair:v3",
            f"performance_preference_profile:{profile}",
            f"performance_preference_winner:{winner}",
            *report.reason_codes,
            *(payload.get("reason_codes") or ()),
        ),
        limit=_MAX_REASON_CODES,
    )
    pair = PerformancePreferencePair(
        pair_id=pair_id,
        created_at=_safe_text(payload.get("created_at") or created_at, limit=96) or _utc_now(),
        profile=profile,
        language=language,
        tts_runtime_label=tts_label,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        winner=winner,
        ratings=ratings,
        improvement_labels=_dedupe(payload.get("improvement_labels") or ()),
        failure_labels=_dedupe(payload.get("failure_labels") or ()),
        evidence_refs=evidence_refs,
        sanitizer=report,
        reason_codes=reason_codes,
    )
    return pair, report


def _proposal_update_for_pair(pair: PerformancePreferencePair) -> tuple[str, dict[str, str], tuple[str, ...]]:
    labels = set(pair.failure_labels) | set(pair.improvement_labels)
    ratings = pair.ratings
    updates: dict[str, str] = {}
    target = "persona_anchor_selection"
    reasons: list[str] = []
    if ratings["voice_pacing"] <= 3 or labels & {"voice_pacing_too_long", "long_monologue"}:
        updates.update({"voice_mode": "concise", "response_depth": "concise"})
        target = "speech_chunking_bias"
        reasons.append("preference_policy:shorter_voice_chunks")
    if ratings["memory_usefulness"] <= 3 or "memory_callback_missing" in labels:
        updates["memory_use"] = "continuity_rich"
        target = "memory_callback_threshold"
        reasons.append("preference_policy:increase_memory_callback")
    if ratings["interruption_naturalness"] <= 3 or labels & {"interruption_awkward", "backchannel_misread"}:
        updates.update({"correction_mode": "precise", "question_budget": "low"})
        target = "interruption_backchannel_policy"
        reasons.append("preference_policy:tighten_repair_turns")
    if ratings["camera_honesty"] <= 3 or labels & {"camera_claim_unclear", "false_camera_claim"}:
        updates["evidence_visibility"] = "rich"
        target = "persona_anchor_selection"
        reasons.append("preference_policy:increase_camera_honesty_visibility")
    if ratings["not_fake_human"] <= 3 or ratings["persona_consistency"] <= 3:
        updates.update({"character_presence": "balanced", "humor_mode": "subtle"})
        target = "persona_anchor_selection"
        reasons.append("preference_policy:reduce_fake_human_risk")
    if not updates:
        updates["evidence_visibility"] = "compact"
        reasons.append("preference_policy:inspection_only")
    return target, updates, tuple(reasons)


def compile_performance_learning_policy_proposals(
    pairs: Iterable[PerformancePreferencePair],
    *,
    created_at: str | None = None,
) -> tuple[PerformanceLearningPolicyProposalV3, ...]:
    """Compile deterministic proposal-first policy updates from preference pairs."""
    proposals: list[PerformanceLearningPolicyProposalV3] = []
    for pair in pairs:
        if pair.winner in {"same", "neither"} and all(score >= 4 for score in pair.ratings.values()):
            continue
        target, updates, reason_codes = _proposal_update_for_pair(pair)
        summary = {
            "speech_chunking_bias": "Shorten voice chunks and response depth for rated pacing.",
            "memory_callback_threshold": "Increase grounded memory callbacks for rated usefulness.",
            "interruption_backchannel_policy": "Prefer tighter repair turns after interruption feedback.",
            "persona_anchor_selection": "Adjust public style anchors without fake-human claims.",
        }.get(target, "Review public-safe behavior controls from preference evidence.")
        payload = {
            "target": target,
            "updates": updates,
            "source_pair_ids": [pair.pair_id],
            "dimension_scores": pair.ratings,
        }
        proposals.append(
            PerformanceLearningPolicyProposalV3(
                proposal_id=_stable_id("performance-policy-proposal", payload),
                created_at=created_at or pair.created_at or _utc_now(),
                status="proposed",
                target=target,
                summary=summary,
                behavior_control_updates=updates,
                source_pair_ids=(pair.pair_id,),
                evidence_refs=pair.evidence_refs,
                dimension_scores={key: float(value) for key, value in pair.ratings.items()},
                reason_codes=_dedupe(
                    (
                        "performance_learning_policy_proposal:v3",
                        f"performance_learning_target:{target}",
                        *reason_codes,
                    ),
                    limit=_MAX_REASON_CODES,
                ),
            )
        )
    deduped: dict[str, PerformanceLearningPolicyProposalV3] = {}
    for proposal in proposals:
        deduped[proposal.proposal_id] = proposal
    return tuple(sorted(deduped.values(), key=lambda item: item.proposal_id))


class PerformancePreferenceStore:
    """Local JSONL store for preference pairs and proposal records."""

    def __init__(self, root: Path | str = PERFORMANCE_PREFERENCE_ARTIFACT_DIR):
        """Initialize the local preference artifact store."""
        self.root = Path(root)
        self.pairs_path = self.root / "preferences.jsonl"
        self.proposals_path = self.root / "policy_proposals.jsonl"

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def append_pair(self, pair: PerformancePreferencePair) -> None:
        """Append one preference pair JSONL row."""
        self._append_jsonl(self.pairs_path, pair.as_dict())

    def append_proposal(self, proposal: PerformanceLearningPolicyProposalV3) -> None:
        """Append one policy proposal JSONL row."""
        self._append_jsonl(self.proposals_path, proposal.as_dict())

    def load_pairs(self, *, limit: int | None = None) -> tuple[PerformancePreferencePair, ...]:
        """Load recent preference pairs from JSONL."""
        pairs: list[PerformancePreferencePair] = []
        for payload in _load_jsonl(self.pairs_path):
            pair, report = build_performance_preference_pair(payload)
            if pair is not None and report.accepted:
                pairs.append(pair)
        rows = tuple(pairs)
        return rows[-limit:] if limit is not None else rows

    def load_proposals(
        self,
        *,
        limit: int | None = None,
    ) -> tuple[PerformanceLearningPolicyProposalV3, ...]:
        """Load recent policy proposals from JSONL."""
        proposals = tuple(
            proposal
            for payload in _load_jsonl(self.proposals_path)
            if (proposal := PerformanceLearningPolicyProposalV3.from_dict(payload)) is not None
        )
        return proposals[-limit:] if limit is not None else proposals

    def find_proposal(self, proposal_id: str) -> PerformanceLearningPolicyProposalV3 | None:
        """Find the newest proposal with the given public ID."""
        safe_id = _safe_token(proposal_id, default="", limit=120)
        for proposal in reversed(self.load_proposals()):
            if proposal.proposal_id == safe_id:
                return proposal
        return None

    def record_pair(
        self,
        payload: dict[str, Any],
        *,
        created_at: str | None = None,
    ) -> tuple[PerformancePreferencePair | None, tuple[PerformanceLearningPolicyProposalV3, ...], PerformancePreferenceSanitizerReport]:
        """Sanitize, append, and derive proposals for one preference payload."""
        pair, report = build_performance_preference_pair(payload, created_at=created_at)
        if pair is None:
            return None, (), report
        self.append_pair(pair)
        proposals = compile_performance_learning_policy_proposals((pair,), created_at=pair.created_at)
        for proposal in proposals:
            self.append_proposal(proposal)
        return pair, proposals, report

    def mark_proposal_applied(
        self,
        proposal: PerformanceLearningPolicyProposalV3,
    ) -> PerformanceLearningPolicyProposalV3:
        """Append an applied-status copy of an existing proposal."""
        applied = replace(
            proposal,
            status="applied",
            reason_codes=_dedupe(
                (*proposal.reason_codes, "performance_learning_policy:applied"),
                limit=_MAX_REASON_CODES,
            ),
        )
        self.append_proposal(applied)
        return applied


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return tuple(rows)


def build_performance_learning_inspection(
    *,
    preferences_dir: Path | str = PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
    recent_limit: int = 8,
) -> dict[str, Any]:
    """Return a compact public-safe workbench payload for performance learning."""
    store = PerformancePreferenceStore(preferences_dir)
    pairs = store.load_pairs(limit=recent_limit)
    proposals = store.load_proposals(limit=recent_limit)
    pair_counts: dict[str, int] = {}
    for pair in pairs:
        pair_counts[pair.profile] = pair_counts.get(pair.profile, 0) + 1
    return {
        "schema_version": PERFORMANCE_PREFERENCE_SCHEMA_VERSION,
        "available": True,
        "summary": (
            f"{len(pairs)} recent preference pairs; {len(proposals)} policy proposals."
        ),
        "preferences_path_kind": "jsonl_file",
        "pair_count": len(pairs),
        "proposal_count": len(proposals),
        "profile_counts": dict(sorted(pair_counts.items())),
        "dimensions": list(PERFORMANCE_PREFERENCE_DIMENSIONS),
        "recent_pairs": [pair.as_dict() for pair in pairs],
        "policy_proposals": [proposal.as_dict() for proposal in proposals],
        "reason_codes": [
            "performance_learning:v3",
            "performance_learning:available",
            "performance_preferences:local_jsonl",
        ],
    }


def render_performance_preference_comparison(
    *,
    preferences_dir: Path | str = PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
) -> dict[str, Any]:
    """Render compact pairwise outcomes for release-candidate review."""
    store = PerformancePreferenceStore(preferences_dir)
    pairs = store.load_pairs()
    winner_counts = {winner: 0 for winner in PERFORMANCE_PREFERENCE_WINNERS}
    dimension_totals = {dimension: 0.0 for dimension in PERFORMANCE_PREFERENCE_DIMENSIONS}
    for pair in pairs:
        winner_counts[pair.winner] += 1
        for dimension, value in pair.ratings.items():
            dimension_totals[dimension] += float(value)
    count = max(1, len(pairs))
    return {
        "schema_version": PERFORMANCE_PREFERENCE_SCHEMA_VERSION,
        "pair_count": len(pairs),
        "winner_counts": winner_counts,
        "average_ratings": {
            dimension: round(total / count, 3)
            for dimension, total in sorted(dimension_totals.items())
        },
        "proposal_count": len(store.load_proposals()),
        "reason_codes": [
            "performance_preference_comparison:v3",
            "performance_preference_comparison:available",
        ],
    }


def render_performance_preference_comparison_markdown(payload: dict[str, Any]) -> str:
    """Render a compact Markdown comparison report."""
    lines = [
        "# Performance Preference Comparison",
        "",
        f"- schema: `{payload.get('schema_version', PERFORMANCE_PREFERENCE_SCHEMA_VERSION)}`",
        f"- pairs: `{payload.get('pair_count', 0)}`",
        f"- proposals: `{payload.get('proposal_count', 0)}`",
        "",
        "## Winners",
        "",
    ]
    for winner, count in dict(payload.get("winner_counts") or {}).items():
        lines.append(f"- `{winner}`: {count}")
    lines.extend(["", "## Average ratings", ""])
    for dimension, value in dict(payload.get("average_ratings") or {}).items():
        lines.append(f"- `{dimension}`: {value}")
    return "\n".join(lines)


__all__ = [
    "PERFORMANCE_LEARNING_POLICY_SCHEMA_VERSION",
    "PERFORMANCE_LEARNING_PROPOSAL_TARGETS",
    "PERFORMANCE_PREFERENCE_ARTIFACT_DIR",
    "PERFORMANCE_PREFERENCE_DIMENSIONS",
    "PERFORMANCE_PREFERENCE_LANGUAGES",
    "PERFORMANCE_PREFERENCE_PROFILES",
    "PERFORMANCE_PREFERENCE_SCHEMA_VERSION",
    "PERFORMANCE_PREFERENCE_TTS_LABELS",
    "PERFORMANCE_PREFERENCE_WINNERS",
    "PerformanceLearningPolicyProposalV3",
    "PerformancePreferenceCandidateRef",
    "PerformancePreferenceEvidenceRef",
    "PerformancePreferencePair",
    "PerformancePreferenceSanitizerReport",
    "PerformancePreferenceStore",
    "build_performance_learning_inspection",
    "build_performance_preference_pair",
    "compile_performance_learning_policy_proposals",
    "render_performance_preference_comparison",
    "render_performance_preference_comparison_markdown",
]
