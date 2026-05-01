"""Layered memory compatibility surface with lazy exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "BrainEpisodicMemoryCandidate",
    "BrainEpisodicMemoryRecord",
    "BrainFactCandidate",
    "BrainMemoryEmbeddingProvider",
    "BrainMemoryExportArtifact",
    "BrainMemoryExporter",
    "BrainMemoryQuery",
    "BrainMemoryRetriever",
    "BrainMemorySearchResult",
    "BrainNarrativeMemoryRecord",
    "BrainSemanticMemoryRecord",
    "BrainTaskCandidate",
    "BrainWorkingMemorySnapshot",
    "build_episodic_candidates_from_event",
    "build_thread_summary_text",
    "build_user_profile_summary",
    "build_working_memory_snapshot",
    "extract_memory_candidates",
    "extract_task_candidates",
    "render_preference_fact",
    "render_profile_fact",
    "semantic_contradiction_key",
    "semantic_default_staleness",
]

_EXPORTS = {
    "BrainEpisodicMemoryCandidate": (
        "blink.brain.memory_layers.episodic",
        "BrainEpisodicMemoryCandidate",
    ),
    "BrainEpisodicMemoryRecord": (
        "blink.brain.memory_layers.episodic",
        "BrainEpisodicMemoryRecord",
    ),
    "build_episodic_candidates_from_event": (
        "blink.brain.memory_layers.episodic",
        "build_episodic_candidates_from_event",
    ),
    "BrainMemoryExportArtifact": (
        "blink.brain.memory_layers.exports",
        "BrainMemoryExportArtifact",
    ),
    "BrainMemoryExporter": ("blink.brain.memory_layers.exports", "BrainMemoryExporter"),
    "BrainNarrativeMemoryRecord": (
        "blink.brain.memory_layers.narrative",
        "BrainNarrativeMemoryRecord",
    ),
    "BrainTaskCandidate": ("blink.brain.memory_layers.narrative", "BrainTaskCandidate"),
    "build_thread_summary_text": (
        "blink.brain.memory_layers.narrative",
        "build_thread_summary_text",
    ),
    "extract_task_candidates": (
        "blink.brain.memory_layers.narrative",
        "extract_task_candidates",
    ),
    "BrainMemoryEmbeddingProvider": (
        "blink.brain.memory_layers.retrieval",
        "BrainMemoryEmbeddingProvider",
    ),
    "BrainMemoryQuery": ("blink.brain.memory_layers.retrieval", "BrainMemoryQuery"),
    "BrainMemoryRetriever": ("blink.brain.memory_layers.retrieval", "BrainMemoryRetriever"),
    "BrainMemorySearchResult": (
        "blink.brain.memory_layers.retrieval",
        "BrainMemorySearchResult",
    ),
    "BrainFactCandidate": ("blink.brain.memory_layers.semantic", "BrainFactCandidate"),
    "BrainSemanticMemoryRecord": (
        "blink.brain.memory_layers.semantic",
        "BrainSemanticMemoryRecord",
    ),
    "build_user_profile_summary": (
        "blink.brain.memory_layers.semantic",
        "build_user_profile_summary",
    ),
    "extract_memory_candidates": (
        "blink.brain.memory_layers.semantic",
        "extract_memory_candidates",
    ),
    "render_preference_fact": (
        "blink.brain.memory_layers.semantic",
        "render_preference_fact",
    ),
    "render_profile_fact": (
        "blink.brain.memory_layers.semantic",
        "render_profile_fact",
    ),
    "semantic_contradiction_key": (
        "blink.brain.memory_layers.semantic",
        "semantic_contradiction_key",
    ),
    "semantic_default_staleness": (
        "blink.brain.memory_layers.semantic",
        "semantic_default_staleness",
    ),
    "BrainWorkingMemorySnapshot": (
        "blink.brain.memory_layers.working",
        "BrainWorkingMemorySnapshot",
    ),
    "build_working_memory_snapshot": (
        "blink.brain.memory_layers.working",
        "build_working_memory_snapshot",
    ),
}


def __getattr__(name: str):
    """Resolve compatibility exports lazily to avoid import-time cycles."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose stable lazy-export names to interactive tooling."""
    return sorted(set(globals()) | set(__all__))
