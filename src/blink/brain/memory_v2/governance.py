"""Pure claim-governance helpers for continuity memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable

from blink.brain.projections import (
    BrainClaimCurrentnessStatus,
    BrainClaimGovernanceProjection,
    BrainClaimGovernanceRecord,
    BrainClaimRetentionClass,
    BrainClaimReviewState,
    BrainGovernanceReasonCode,
)

_TERMINAL_TRUTH_STATUSES = {"revoked", "superseded"}
_REVALIDATION_CLEAR_REASON_CODES = {
    BrainGovernanceReasonCode.EXPIRED_BY_POLICY.value,
    BrainGovernanceReasonCode.STALE_WITHOUT_REFRESH.value,
    BrainGovernanceReasonCode.OPERATOR_HOLD.value,
    BrainGovernanceReasonCode.LOW_SUPPORT.value,
    BrainGovernanceReasonCode.REQUIRES_CONFIRMATION.value,
}
_KNOWN_REASON_CODES = {code.value for code in BrainGovernanceReasonCode}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _normalize_reason_code(value: str) -> str:
    normalized = value.strip()
    if normalized not in _KNOWN_REASON_CODES:
        raise ValueError(f"Unsupported governance reason code: {normalized}")
    return normalized


def _normalize_currentness_status(value: str) -> str:
    return BrainClaimCurrentnessStatus(str(value).strip()).value


def _normalize_review_state(value: str) -> str:
    return BrainClaimReviewState(str(value).strip()).value


def _normalize_retention_class(value: str) -> str:
    return BrainClaimRetentionClass(str(value).strip()).value


def normalize_reason_codes(reason_codes: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize one optional reason-code list into a stable tuple."""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in reason_codes or ():
        code = _normalize_reason_code(str(value))
        if code not in seen:
            seen.add(code)
            normalized.append(code)
    return tuple(normalized)


def append_reason_codes(
    existing_reason_codes: Iterable[str] | None,
    new_reason_codes: Iterable[str] | None,
) -> tuple[str, ...]:
    """Append normalized governance reason codes without duplicates."""
    ordered = list(normalize_reason_codes(existing_reason_codes))
    seen = set(ordered)
    for code in normalize_reason_codes(new_reason_codes):
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    return tuple(ordered)


def default_claim_retention_class(*, scope_type: str | None, predicate: str | None) -> str:
    """Return the default explicit retention class for one claim."""
    normalized_scope_type = str(scope_type or "").strip()
    normalized_predicate = str(predicate or "").strip()
    if normalized_scope_type == "thread" or normalized_predicate == "session.summary":
        return BrainClaimRetentionClass.SESSION.value
    if normalized_scope_type in {"user", "relationship", "agent"}:
        return BrainClaimRetentionClass.DURABLE.value
    return BrainClaimRetentionClass.DURABLE.value


def effective_claim_review_state(review_state: str | None) -> str:
    """Return the explicit review state or its compatibility default."""
    if review_state is None:
        return BrainClaimReviewState.NONE.value
    return _normalize_review_state(review_state)


def effective_claim_retention_class(
    retention_class: str | None,
    *,
    scope_type: str | None,
    predicate: str | None,
) -> str:
    """Return the explicit retention class or its compatibility default."""
    if retention_class is None:
        return default_claim_retention_class(scope_type=scope_type, predicate=predicate)
    return _normalize_retention_class(retention_class)


def effective_claim_currentness_status(
    currentness_status: str | None,
    *,
    truth_status: str,
    valid_from: str | None,
    valid_to: str | None,
    stale_after_seconds: int | None,
    now: str | None = None,
) -> str:
    """Return one effective currentness state with a legacy fallback."""
    if currentness_status is not None:
        return _normalize_currentness_status(currentness_status)

    if truth_status in _TERMINAL_TRUTH_STATUSES:
        return BrainClaimCurrentnessStatus.HISTORICAL.value

    now_ts = _parse_ts(now or _utc_now()) or datetime.now(UTC)
    valid_to_ts = _parse_ts(valid_to)
    if valid_to_ts is not None and valid_to_ts <= now_ts:
        return BrainClaimCurrentnessStatus.HISTORICAL.value

    if stale_after_seconds not in (None, 0):
        valid_from_ts = _parse_ts(valid_from)
        if valid_from_ts is not None and now_ts > (
            valid_from_ts + timedelta(seconds=int(stale_after_seconds))
        ):
            return BrainClaimCurrentnessStatus.STALE.value

    return BrainClaimCurrentnessStatus.CURRENT.value


def claim_matches_temporal_mode(
    *,
    temporal_mode: str,
    currentness_status: str | None,
    truth_status: str,
    valid_from: str | None,
    valid_to: str | None,
    stale_after_seconds: int | None,
    now: str | None = None,
) -> bool:
    """Return whether one claim belongs in the compatibility temporal partition."""
    mode = str(temporal_mode).strip().lower()
    effective_currentness = effective_claim_currentness_status(
        currentness_status,
        truth_status=truth_status,
        valid_from=valid_from,
        valid_to=valid_to,
        stale_after_seconds=stale_after_seconds,
        now=now,
    )
    if mode == "all":
        return True
    if mode == "historical":
        return effective_currentness == BrainClaimCurrentnessStatus.HISTORICAL.value
    return effective_currentness != BrainClaimCurrentnessStatus.HISTORICAL.value


@dataclass(frozen=True)
class ClaimGovernanceState:
    """Internal normalized governance state for one claim row."""

    currentness_status: str
    review_state: str
    retention_class: str
    reason_codes: tuple[str, ...]
    last_governance_event_id: str | None
    governance_updated_at: str


def seed_claim_governance(
    *,
    scope_type: str | None,
    predicate: str | None,
    truth_status: str,
    updated_at: str | None = None,
    currentness_status: str | None = None,
    review_state: str | None = None,
    retention_class: str | None = None,
    reason_codes: Iterable[str] | None = None,
    last_governance_event_id: str | None = None,
) -> ClaimGovernanceState:
    """Seed explicit governance fields for a newly persisted claim row."""
    seeded_currentness = (
        BrainClaimCurrentnessStatus.HISTORICAL.value
        if truth_status in _TERMINAL_TRUTH_STATUSES
        else BrainClaimCurrentnessStatus.CURRENT.value
    )
    return ClaimGovernanceState(
        currentness_status=_normalize_currentness_status(currentness_status or seeded_currentness),
        review_state=effective_claim_review_state(review_state),
        retention_class=effective_claim_retention_class(
            retention_class,
            scope_type=scope_type,
            predicate=predicate,
        ),
        reason_codes=normalize_reason_codes(reason_codes),
        last_governance_event_id=_optional_text(last_governance_event_id),
        governance_updated_at=str(updated_at or _utc_now()),
    )


def transition_claim_governance(
    *,
    truth_status: str,
    currentness_status: str | None,
    review_state: str | None,
    retention_class: str | None,
    existing_reason_codes: Iterable[str] | None,
    scope_type: str | None,
    predicate: str | None,
    transition: str,
    updated_at: str | None = None,
    last_governance_event_id: str | None = None,
    reason_codes: Iterable[str] | None = None,
    explicit_review_state: str | None = None,
    explicit_retention_class: str | None = None,
) -> ClaimGovernanceState:
    """Return the next explicit governance state for one legal claim transition."""
    next_updated_at = str(updated_at or _utc_now())
    next_currentness = effective_claim_currentness_status(
        currentness_status,
        truth_status=truth_status,
        valid_from=None,
        valid_to=None,
        stale_after_seconds=None,
    )
    next_review_state = effective_claim_review_state(review_state)
    next_retention_class = effective_claim_retention_class(
        retention_class,
        scope_type=scope_type,
        predicate=predicate,
    )
    next_reason_codes = normalize_reason_codes(existing_reason_codes)

    if transition == "superseded":
        next_currentness = BrainClaimCurrentnessStatus.HISTORICAL.value
        next_reason_codes = append_reason_codes(
            next_reason_codes,
            reason_codes or (BrainGovernanceReasonCode.SUPERSEDED.value,),
        )
    elif transition == "revoked":
        next_currentness = BrainClaimCurrentnessStatus.HISTORICAL.value
        next_reason_codes = append_reason_codes(
            next_reason_codes,
            reason_codes or (BrainGovernanceReasonCode.CONTRADICTION.value,),
        )
    elif transition == "review_requested":
        next_currentness = BrainClaimCurrentnessStatus.HELD.value
        next_review_state = BrainClaimReviewState.REQUESTED.value
        next_reason_codes = append_reason_codes(
            next_reason_codes,
            reason_codes or (BrainGovernanceReasonCode.OPERATOR_HOLD.value,),
        )
    elif transition == "expired":
        next_currentness = BrainClaimCurrentnessStatus.STALE.value
        next_review_state = (
            _normalize_review_state(explicit_review_state)
            if explicit_review_state is not None
            else next_review_state
        )
        next_reason_codes = append_reason_codes(
            next_reason_codes,
            reason_codes or (BrainGovernanceReasonCode.EXPIRED_BY_POLICY.value,),
        )
    elif transition == "revalidated":
        next_currentness = BrainClaimCurrentnessStatus.CURRENT.value
        next_review_state = BrainClaimReviewState.RESOLVED.value
        next_reason_codes = (
            normalize_reason_codes(reason_codes)
            if reason_codes is not None
            else tuple(
                code for code in next_reason_codes if code not in _REVALIDATION_CLEAR_REASON_CODES
            )
        )
    elif transition == "retention_reclassified":
        if explicit_retention_class is None:
            raise ValueError("retention_reclassified requires an explicit retention class.")
        next_retention_class = _normalize_retention_class(explicit_retention_class)
        next_reason_codes = (
            normalize_reason_codes(reason_codes) if reason_codes is not None else next_reason_codes
        )
    else:
        raise ValueError(f"Unsupported claim governance transition: {transition}")

    return ClaimGovernanceState(
        currentness_status=next_currentness,
        review_state=next_review_state,
        retention_class=next_retention_class,
        reason_codes=next_reason_codes,
        last_governance_event_id=_optional_text(last_governance_event_id),
        governance_updated_at=next_updated_at,
    )


def build_claim_governance_projection(
    *,
    scope_type: str,
    scope_id: str,
    claims: Iterable[Any],
    updated_at: str | None = None,
) -> BrainClaimGovernanceProjection:
    """Build one deterministic scoped claim-governance projection."""
    currentness_counts = {status.value: 0 for status in BrainClaimCurrentnessStatus}
    review_state_counts = {status.value: 0 for status in BrainClaimReviewState}
    retention_class_counts = {status.value: 0 for status in BrainClaimRetentionClass}
    records: list[BrainClaimGovernanceRecord] = []
    current_claim_ids: list[str] = []
    stale_claim_ids: list[str] = []
    historical_claim_ids: list[str] = []
    held_claim_ids: list[str] = []
    latest_updated_at = updated_at or ""

    ordered_claims = sorted(
        [
            claim
            for claim in claims
            if getattr(claim, "scope_type", None) == scope_type
            and getattr(claim, "scope_id", None) == scope_id
        ],
        key=lambda claim: (
            str(getattr(claim, "governance_updated_at", "") or getattr(claim, "updated_at", "")),
            str(getattr(claim, "updated_at", "")),
            str(getattr(claim, "claim_id", "")),
        ),
        reverse=True,
    )
    for claim in ordered_claims:
        currentness = effective_claim_currentness_status(
            getattr(claim, "currentness_status", None),
            truth_status=str(getattr(claim, "status", "")),
            valid_from=getattr(claim, "valid_from", None),
            valid_to=getattr(claim, "valid_to", None),
            stale_after_seconds=getattr(claim, "stale_after_seconds", None),
            now=updated_at,
        )
        review = effective_claim_review_state(getattr(claim, "review_state", None))
        retention = effective_claim_retention_class(
            getattr(claim, "retention_class", None),
            scope_type=getattr(claim, "scope_type", None),
            predicate=getattr(claim, "predicate", None),
        )
        reason_codes = normalize_reason_codes(getattr(claim, "governance_reason_codes", ()))
        record_updated_at = str(
            getattr(claim, "governance_updated_at", None)
            or getattr(claim, "updated_at", None)
            or updated_at
            or _utc_now()
        )
        latest_updated_at = max(latest_updated_at, record_updated_at)
        record = BrainClaimGovernanceRecord(
            claim_id=str(getattr(claim, "claim_id", "")),
            scope_type=scope_type,
            scope_id=scope_id,
            truth_status=str(getattr(claim, "status", "")),
            currentness_status=currentness,
            review_state=review,
            retention_class=retention,
            reason_codes=reason_codes,
            last_governance_event_id=_optional_text(
                getattr(claim, "last_governance_event_id", None)
            ),
            updated_at=record_updated_at,
        )
        records.append(record)
        currentness_counts[currentness] = currentness_counts.get(currentness, 0) + 1
        review_state_counts[review] = review_state_counts.get(review, 0) + 1
        retention_class_counts[retention] = retention_class_counts.get(retention, 0) + 1
        if currentness == BrainClaimCurrentnessStatus.CURRENT.value:
            current_claim_ids.append(record.claim_id)
        elif currentness == BrainClaimCurrentnessStatus.STALE.value:
            stale_claim_ids.append(record.claim_id)
        elif currentness == BrainClaimCurrentnessStatus.HISTORICAL.value:
            historical_claim_ids.append(record.claim_id)
        elif currentness == BrainClaimCurrentnessStatus.HELD.value:
            held_claim_ids.append(record.claim_id)

    return BrainClaimGovernanceProjection(
        scope_type=scope_type,
        scope_id=scope_id,
        records=records,
        currentness_counts=currentness_counts,
        review_state_counts=review_state_counts,
        retention_class_counts=retention_class_counts,
        current_claim_ids=sorted(current_claim_ids),
        stale_claim_ids=sorted(stale_claim_ids),
        historical_claim_ids=sorted(historical_claim_ids),
        held_claim_ids=sorted(held_claim_ids),
        updated_at=latest_updated_at or str(updated_at or _utc_now()),
    )


__all__ = [
    "ClaimGovernanceState",
    "append_reason_codes",
    "build_claim_governance_projection",
    "claim_matches_temporal_mode",
    "default_claim_retention_class",
    "effective_claim_currentness_status",
    "effective_claim_retention_class",
    "effective_claim_review_state",
    "normalize_reason_codes",
    "seed_claim_governance",
    "transition_claim_governance",
]
