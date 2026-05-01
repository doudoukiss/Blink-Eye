"""Browser visual presence detection, enrichment, and transport health."""

from blink.brain.perception.detector import PresenceDetectionResult
from blink.brain.perception.enrichment import VisionEnrichmentResult
from blink.brain.perception.fusion import (
    FusedPresenceState,
    PerceptionBroker,
    PerceptionBrokerConfig,
    PerceptionObservation,
)
from blink.brain.perception.health import (
    DEFAULT_CAMERA_STALE_FRAME_SECS,
    CameraFeedHealth,
    CameraFeedHealthManager,
    CameraFeedHealthManagerConfig,
    CameraTrackHealthEvent,
)

__all__ = [
    "CameraFeedHealth",
    "CameraFeedHealthManager",
    "CameraFeedHealthManagerConfig",
    "CameraTrackHealthEvent",
    "DEFAULT_CAMERA_STALE_FRAME_SECS",
    "FusedPresenceState",
    "PerceptionBroker",
    "PerceptionBrokerConfig",
    "PerceptionObservation",
    "PresenceDetectionResult",
    "VisionEnrichmentResult",
]
