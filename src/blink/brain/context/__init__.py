"""Lazy public exports for Blink context compilation."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "BrainActiveContextAnchorCandidate",
    "BrainActiveContextCompiler",
    "BrainActiveContextExpansionRecord",
    "BrainActiveContextPacket",
    "BrainActiveContextPacketItemRecord",
    "BrainActiveContextSectionDecision",
    "BrainActiveContextTrace",
    "BrainCompiledContextPacket",
    "BrainContextBudgetProfile",
    "BrainContextCompiler",
    "BrainContextModePolicy",
    "BrainContextSceneEpisodePolicy",
    "BrainContextSelectionDecision",
    "BrainContextSelectionTrace",
    "BrainContextSelector",
    "BrainContextTask",
    "BrainContextTemporalMode",
    "BrainContextTraceVerbosity",
    "BrainSelectedContext",
    "BrainSelectedSection",
    "approximate_token_count",
    "compile_context_packet_from_surface",
    "get_brain_context_mode_policy",
]

_EXPORTS = {
    "BrainContextBudgetProfile": ("blink.brain.context.budgets", "BrainContextBudgetProfile"),
    "approximate_token_count": ("blink.brain.context.budgets", "approximate_token_count"),
    "BrainActiveContextAnchorCandidate": (
        "blink.brain.context.compiler",
        "BrainActiveContextAnchorCandidate",
    ),
    "BrainActiveContextCompiler": ("blink.brain.context.compiler", "BrainActiveContextCompiler"),
    "BrainActiveContextExpansionRecord": (
        "blink.brain.context.compiler",
        "BrainActiveContextExpansionRecord",
    ),
    "BrainActiveContextPacket": ("blink.brain.context.compiler", "BrainActiveContextPacket"),
    "BrainActiveContextPacketItemRecord": (
        "blink.brain.context.compiler",
        "BrainActiveContextPacketItemRecord",
    ),
    "BrainActiveContextSectionDecision": (
        "blink.brain.context.compiler",
        "BrainActiveContextSectionDecision",
    ),
    "BrainActiveContextTrace": ("blink.brain.context.compiler", "BrainActiveContextTrace"),
    "BrainCompiledContextPacket": ("blink.brain.context.compiler", "BrainCompiledContextPacket"),
    "BrainContextCompiler": ("blink.brain.context.compiler", "BrainContextCompiler"),
    "BrainContextTemporalMode": ("blink.brain.context.compiler", "BrainContextTemporalMode"),
    "compile_context_packet_from_surface": (
        "blink.brain.context.compiler",
        "compile_context_packet_from_surface",
    ),
    "BrainContextModePolicy": ("blink.brain.context.policy", "BrainContextModePolicy"),
    "BrainContextSceneEpisodePolicy": (
        "blink.brain.context.policy",
        "BrainContextSceneEpisodePolicy",
    ),
    "BrainContextTask": ("blink.brain.context.policy", "BrainContextTask"),
    "BrainContextTraceVerbosity": ("blink.brain.context.policy", "BrainContextTraceVerbosity"),
    "get_brain_context_mode_policy": (
        "blink.brain.context.policy",
        "get_brain_context_mode_policy",
    ),
    "BrainContextSelectionDecision": (
        "blink.brain.context.selectors",
        "BrainContextSelectionDecision",
    ),
    "BrainContextSelectionTrace": ("blink.brain.context.selectors", "BrainContextSelectionTrace"),
    "BrainContextSelector": ("blink.brain.context.selectors", "BrainContextSelector"),
    "BrainSelectedContext": ("blink.brain.context.selectors", "BrainSelectedContext"),
    "BrainSelectedSection": ("blink.brain.context.selectors", "BrainSelectedSection"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
