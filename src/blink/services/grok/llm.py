#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Grok LLM service implementation.

.. deprecated:: 0.0.108
    This module is deprecated. Please use GrokLLMService from
    blink.services.xai.llm instead.
"""

import warnings

from blink.services.xai.llm import *  # noqa: F401,F403

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "blink.services.grok.llm is deprecated. Please use blink.services.xai.llm instead.",
        DeprecationWarning,
        stacklevel=2,
    )
