"""Public API for Blink's bounded teaching knowledge scaffold."""

from __future__ import annotations

from blink.brain.knowledge.registry import KnowledgeReserveRegistry
from blink.brain.knowledge.schema import (
    ConceptMapEntry,
    ExplanationExemplar,
    KnowledgeReserveEntry,
    TeachingSequence,
)
from blink.brain.knowledge.selectors import (
    BrainKnowledgeRoutingDecision,
    BrainKnowledgeRoutingItem,
    KnowledgeSelectionRequest,
    KnowledgeSelectionResult,
    explicit_knowledge_routing_decision,
    knowledge_routing_decision_from_selection,
    render_selected_teaching_knowledge,
    select_teaching_knowledge,
    unavailable_knowledge_routing_decision,
)
from blink.brain.knowledge.teaching_canon import (
    DEFAULT_TEACHING_CANON,
    build_default_teaching_canon,
)

__all__ = [
    "BrainKnowledgeRoutingDecision",
    "BrainKnowledgeRoutingItem",
    "ConceptMapEntry",
    "DEFAULT_TEACHING_CANON",
    "ExplanationExemplar",
    "KnowledgeReserveEntry",
    "KnowledgeReserveRegistry",
    "KnowledgeSelectionRequest",
    "KnowledgeSelectionResult",
    "TeachingSequence",
    "build_default_teaching_canon",
    "explicit_knowledge_routing_decision",
    "knowledge_routing_decision_from_selection",
    "render_selected_teaching_knowledge",
    "select_teaching_knowledge",
    "unavailable_knowledge_routing_decision",
]
