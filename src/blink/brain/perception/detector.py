"""Deterministic browser presence detection backed by a packaged ONNX face detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from blink.frames.frames import UserImageRawFrame

_MODEL_FILENAME = "version-RFB-320.onnx"
_INPUT_WIDTH = 320
_INPUT_HEIGHT = 240
_MIN_BOXES = ((10, 16, 24), (32, 48), (64, 96), (128, 192, 256))
_STRIDES = (8, 16, 32, 64)
_CENTER_VARIANCE = 0.1
_SIZE_VARIANCE = 0.2
_CANDIDATE_THRESHOLD = 0.05
_LOW_CONFIDENCE_THRESHOLD = 0.35


@dataclass(frozen=True)
class PresenceDetectionResult:
    """One deterministic presence-classification result."""

    state: str
    confidence: float
    backend: str
    available: bool = True
    reason: str | None = None
    face_count: int = 0
    boxes: tuple[tuple[float, float, float, float], ...] = field(default_factory=tuple)

    @classmethod
    def unavailable(cls, *, backend: str, reason: str) -> "PresenceDetectionResult":
        """Build an unavailable detector result."""
        return cls(
            state="uncertain",
            confidence=0.0,
            backend=backend,
            available=False,
            reason=reason,
        )


class OnnxFacePresenceDetector:
    """Run a packaged Ultra-Light face detector against raw camera frames."""

    backend = "onnx_face_detector"

    def __init__(self):
        """Lazily load the packaged ONNX model."""
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._load_error: str | None = None
        self._priors = _generate_priors()

    @property
    def available(self) -> bool:
        """Return whether the detector is ready to run."""
        self._ensure_loaded()
        return self._session is not None

    @property
    def load_error(self) -> str | None:
        """Return the last detector loading error, if any."""
        self._ensure_loaded()
        return self._load_error

    def detect(self, frame: UserImageRawFrame) -> PresenceDetectionResult:
        """Classify one latest cached camera frame."""
        self._ensure_loaded()
        if self._session is None or self._input_name is None:
            return PresenceDetectionResult.unavailable(
                backend=self.backend,
                reason=self._load_error or "presence_detector_unavailable",
            )

        image = _frame_to_image(frame)
        if image is None:
            return PresenceDetectionResult.unavailable(
                backend=self.backend,
                reason="presence_detector_invalid_frame",
            )

        resized = image.resize((_INPUT_WIDTH, _INPUT_HEIGHT), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)
        normalized = ((array - 127.0) / 128.0).transpose(2, 0, 1)[None, :, :, :]

        scores, boxes = self._session.run(None, {self._input_name: normalized})
        score_vector = scores[0, :, 1]
        candidate_indexes = np.where(score_vector > _CANDIDATE_THRESHOLD)[0]
        if candidate_indexes.size == 0:
            return PresenceDetectionResult(
                state="absent",
                confidence=0.0,
                backend=self.backend,
                reason="no_face_detected",
            )

        decoded_boxes = _decode_boxes(boxes[0], self._priors)
        selected_boxes = decoded_boxes[candidate_indexes]
        selected_scores = score_vector[candidate_indexes]
        kept_indexes = _hard_nms(selected_boxes, selected_scores, iou_threshold=0.3, top_k=3)
        if kept_indexes.size == 0:
            return PresenceDetectionResult(
                state="absent",
                confidence=0.0,
                backend=self.backend,
                reason="no_face_detected",
            )

        final_boxes = selected_boxes[kept_indexes]
        final_scores = selected_scores[kept_indexes]
        best_score = float(np.max(final_scores))
        normalized_boxes = tuple(
            (
                max(0.0, float(box[0])),
                max(0.0, float(box[1])),
                min(1.0, float(box[2])),
                min(1.0, float(box[3])),
            )
            for box in final_boxes
        )

        if best_score >= 0.65:
            return PresenceDetectionResult(
                state="present",
                confidence=best_score,
                backend=self.backend,
                reason="face_detected",
                face_count=len(normalized_boxes),
                boxes=normalized_boxes,
            )
        if best_score >= _LOW_CONFIDENCE_THRESHOLD:
            return PresenceDetectionResult(
                state="uncertain",
                confidence=best_score,
                backend=self.backend,
                reason="presence_detector_low_confidence",
                face_count=len(normalized_boxes),
                boxes=normalized_boxes,
            )
        return PresenceDetectionResult(
            state="absent",
            confidence=best_score,
            backend=self.backend,
            reason="no_face_detected",
        )

    def _ensure_loaded(self):
        if self._session is not None or self._load_error is not None:
            return
        try:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            model_path = (
                resources.files("blink.brain.perception")
                .joinpath("data")
                .joinpath(_MODEL_FILENAME)
            )
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:  # pragma: no cover - environment-specific
            self._load_error = str(exc)


def _frame_to_image(frame: UserImageRawFrame) -> Image.Image | None:
    if frame.format != "RGB":
        return None
    width, height = frame.size
    array = np.frombuffer(frame.image, dtype=np.uint8)
    expected = width * height * 3
    if array.size != expected:
        return None
    return Image.fromarray(array.reshape((height, width, 3)), mode="RGB")


def _generate_priors() -> np.ndarray:
    priors: list[list[float]] = []
    for stride, min_boxes in zip(_STRIDES, _MIN_BOXES, strict=True):
        feature_height = int(np.ceil(_INPUT_HEIGHT / stride))
        feature_width = int(np.ceil(_INPUT_WIDTH / stride))
        for row in range(feature_height):
            for col in range(feature_width):
                x_center = (col + 0.5) * stride / _INPUT_WIDTH
                y_center = (row + 0.5) * stride / _INPUT_HEIGHT
                for min_box in min_boxes:
                    priors.append(
                        [
                            x_center,
                            y_center,
                            min_box / _INPUT_WIDTH,
                            min_box / _INPUT_HEIGHT,
                        ]
                    )
    return np.asarray(priors, dtype=np.float32)


def _decode_boxes(raw_boxes: np.ndarray, priors: np.ndarray) -> np.ndarray:
    centers = priors[:, :2] + raw_boxes[:, :2] * _CENTER_VARIANCE * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(raw_boxes[:, 2:] * _SIZE_VARIANCE)
    top_left = centers - (sizes / 2.0)
    bottom_right = centers + (sizes / 2.0)
    return np.concatenate([top_left, bottom_right], axis=1)


def _hard_nms(boxes: np.ndarray, scores: np.ndarray, *, iou_threshold: float, top_k: int) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    kept: list[int] = []
    while order.size > 0 and len(kept) < top_k:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        current_box = boxes[current]
        remaining = boxes[order[1:]]
        ious = _iou(current_box, remaining)
        order = order[1:][ious <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inter_top_left = np.maximum(box[:2], boxes[:, :2])
    inter_bottom_right = np.minimum(box[2:], boxes[:, 2:])
    inter_sizes = np.maximum(0.0, inter_bottom_right - inter_top_left)
    inter_area = inter_sizes[:, 0] * inter_sizes[:, 1]

    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter_area
    return np.divide(inter_area, union, out=np.zeros_like(inter_area), where=union > 0)
