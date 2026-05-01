"""Optional VLM enrichment over fresh browser camera frames."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from blink.frames.frames import ErrorFrame, UserImageRawFrame, VisionTextFrame

_SYMBOLIC_VISION_PROMPT = (
    "Inspect the current camera frame and return strict JSON only. "
    "Use exactly these keys: "
    "attention_to_camera, engagement_state, summary, confidence, scene_zones, scene_entities. "
    "Allowed values: "
    "attention_to_camera=toward_camera|away|unknown; "
    "engagement_state=engaged|speaking|listening|idle|away|unknown. "
    "The summary must be one short neutral sentence. "
    "Confidence must be a float between 0.0 and 1.0. "
    "scene_zones and scene_entities are optional arrays. "
    "Each scene zone may contain zone_key, label, summary, confidence. "
    "Each scene entity may contain entity_key, kind, label, summary, zone_key, confidence, and optional affordances. "
    "Each affordance may contain capability_family, summary, availability, confidence. "
    "Do not add markdown, explanations, or extra keys."
)


@dataclass(frozen=True)
class VisionEnrichmentResult:
    """One optional VLM semantic enrichment result."""

    attention_to_camera: str
    engagement_state: str
    summary: str
    confidence: float
    scene_zones: list[dict[str, Any]] = field(default_factory=list)
    scene_entities: list[dict[str, Any]] = field(default_factory=list)
    available: bool = True
    reason: str | None = None

    @classmethod
    def unavailable(cls, *, reason: str) -> "VisionEnrichmentResult":
        """Build an unavailable enrichment result."""
        return cls(
            attention_to_camera="unknown",
            engagement_state="unknown",
            summary="",
            confidence=0.0,
            scene_zones=[],
            scene_entities=[],
            available=False,
            reason=reason,
        )


class VisionEnrichmentEngine:
    """Run the optional low-cadence VLM enrichment query."""

    def __init__(self, *, vision):
        """Bind the optional vision service."""
        self._vision = vision

    @property
    def available(self) -> bool:
        """Return whether semantic VLM enrichment is available."""
        return self._vision is not None

    async def enrich(self, image_frame: UserImageRawFrame) -> VisionEnrichmentResult:
        """Run one strict JSON VLM enrichment query."""
        if self._vision is None:
            return VisionEnrichmentResult.unavailable(reason="vision_enrichment_unavailable")

        query_frame = UserImageRawFrame(
            user_id=image_frame.user_id,
            image=image_frame.image,
            size=image_frame.size,
            format=image_frame.format,
            text=_SYMBOLIC_VISION_PROMPT,
        )
        query_frame.transport_source = image_frame.transport_source
        query_frame.pts = image_frame.pts

        description: str | None = None
        error_text: str | None = None
        async for result_frame in self._vision.run_vision(query_frame):
            if isinstance(result_frame, VisionTextFrame):
                description = (result_frame.text or "").strip()
            elif isinstance(result_frame, ErrorFrame):
                error_text = result_frame.error

        if error_text:
            return VisionEnrichmentResult.unavailable(reason="vision_enrichment_unavailable")
        try:
            payload = _parse_json_object(description or "")
        except Exception:
            return VisionEnrichmentResult.unavailable(reason="vision_enrichment_parse_error")

        summary = " ".join(str(payload.get("summary", "")).split()).strip()
        if not summary:
            return VisionEnrichmentResult.unavailable(reason="vision_enrichment_parse_error")

        return VisionEnrichmentResult(
            attention_to_camera=_enum_value(
                payload.get("attention_to_camera"),
                allowed={"toward_camera", "away", "unknown"},
                fallback="unknown",
            ),
            engagement_state=_enum_value(
                payload.get("engagement_state"),
                allowed={"engaged", "speaking", "listening", "idle", "away", "unknown"},
                fallback="unknown",
            ),
            summary=summary,
            confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.0)))),
            scene_zones=_normalize_scene_zones(payload.get("scene_zones")),
            scene_entities=_normalize_scene_entities(payload.get("scene_entities")),
            available=True,
        )


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("Vision enrichment output was empty.")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Vision enrichment output did not contain JSON.") from None
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Vision enrichment output must be a JSON object.")
    return parsed


def _enum_value(value: Any, *, allowed: set[str], fallback: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else fallback


def _normalize_scene_zones(value: Any) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for item in list(value or [])[:3]:
        if not isinstance(item, dict):
            continue
        zones.append(
            {
                key: item.get(key)
                for key in ("zone_key", "label", "summary", "confidence", "fresh_for_secs", "expire_after_secs")
                if item.get(key) is not None
            }
        )
    return zones


def _normalize_scene_entities(value: Any) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for item in list(value or [])[:4]:
        if not isinstance(item, dict):
            continue
        affordances = []
        for affordance in list(item.get("affordances", []) or [])[:2]:
            if not isinstance(affordance, dict):
                continue
            affordances.append(
                {
                    key: affordance.get(key)
                    for key in (
                        "capability_family",
                        "summary",
                        "availability",
                        "confidence",
                        "fresh_for_secs",
                        "expire_after_secs",
                    )
                    if affordance.get(key) is not None
                }
            )
        normalized = {
            key: item.get(key)
            for key in (
                "entity_key",
                "kind",
                "entity_kind",
                "label",
                "canonical_label",
                "summary",
                "zone_key",
                "zone_id",
                "confidence",
                "present",
                "visible",
                "fresh_for_secs",
                "expire_after_secs",
            )
            if item.get(key) is not None
        }
        if affordances:
            normalized["affordances"] = affordances
        entities.append(normalized)
    return entities
