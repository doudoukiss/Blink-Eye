"""Brain-side perception adapter contracts and local implementations."""

from __future__ import annotations

from typing import Protocol

from blink.brain.adapters import LOCAL_PERCEPTION_DESCRIPTOR, BrainAdapterDescriptor
from blink.brain.perception.detector import OnnxFacePresenceDetector, PresenceDetectionResult
from blink.brain.perception.enrichment import VisionEnrichmentEngine, VisionEnrichmentResult
from blink.frames.frames import UserImageRawFrame


class PerceptionAdapter(Protocol):
    """Bounded backend seam for perception detection and scene enrichment."""

    @property
    def descriptor(self) -> BrainAdapterDescriptor:
        """Return the backend descriptor."""

    @property
    def presence_detection_backend(self) -> str:
        """Return the low-level presence detection backend id."""

    @property
    def presence_detection_available(self) -> bool:
        """Return whether deterministic presence detection is available."""

    @property
    def scene_enrichment_available(self) -> bool:
        """Return whether optional scene enrichment is available."""

    def detect_presence(self, frame: UserImageRawFrame) -> PresenceDetectionResult:
        """Detect presence from one camera frame."""

    async def enrich_scene(self, frame: UserImageRawFrame) -> VisionEnrichmentResult:
        """Enrich one camera frame with scene semantics."""


class LocalPerceptionAdapter:
    """Local perception adapter over packaged ONNX detection and optional VLM enrichment."""

    def __init__(
        self,
        *,
        vision=None,
        detector: OnnxFacePresenceDetector | None = None,
        enrichment: VisionEnrichmentEngine | None = None,
    ):
        """Initialize the local perception adapter."""
        self._detector = detector or OnnxFacePresenceDetector()
        self._enrichment = enrichment or VisionEnrichmentEngine(vision=vision)
        self._descriptor = LOCAL_PERCEPTION_DESCRIPTOR

    @property
    def descriptor(self) -> BrainAdapterDescriptor:
        """Return the backend descriptor."""
        return self._descriptor

    @property
    def presence_detection_backend(self) -> str:
        """Return the low-level presence detector backend id."""
        return str(getattr(self._detector, "backend", "presence_detection"))

    @property
    def presence_detection_available(self) -> bool:
        """Return whether local presence detection is available."""
        return bool(getattr(self._detector, "available", False))

    @property
    def scene_enrichment_available(self) -> bool:
        """Return whether local scene enrichment is available."""
        return bool(getattr(self._enrichment, "available", False))

    def detect_presence(self, frame: UserImageRawFrame) -> PresenceDetectionResult:
        """Run deterministic presence detection."""
        return self._detector.detect(frame)

    async def enrich_scene(self, frame: UserImageRawFrame) -> VisionEnrichmentResult:
        """Run optional scene enrichment."""
        return await self._enrichment.enrich(frame)


__all__ = ["LocalPerceptionAdapter", "PerceptionAdapter"]
