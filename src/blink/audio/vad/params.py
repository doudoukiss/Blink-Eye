#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Provider-light VAD parameter types.

This module intentionally stays free of analyzer/runtime audio helpers so core
frame and runtime imports can reference ``VADParams`` without pulling in
optional loudness dependencies.
"""

from __future__ import annotations

from pydantic import BaseModel

VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6


class VADParams(BaseModel):
    """Configuration parameters for Voice Activity Detection.

    Parameters:
        confidence: Minimum confidence threshold for voice detection.
        start_secs: Duration to wait before confirming voice start.
        stop_secs: Duration to wait before confirming voice stop.
        min_volume: Minimum audio volume threshold for voice detection.
    """

    confidence: float = VAD_CONFIDENCE
    start_secs: float = VAD_START_SECS
    stop_secs: float = VAD_STOP_SECS
    min_volume: float = VAD_MIN_VOLUME


__all__ = [
    "VAD_CONFIDENCE",
    "VAD_MIN_VOLUME",
    "VAD_START_SECS",
    "VAD_STOP_SECS",
    "VADParams",
]
