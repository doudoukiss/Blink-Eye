from __future__ import annotations

import pytest
from pydantic import ValidationError

from blink.brain.context.budgets import approximate_token_count
from blink.brain.knowledge import (
    ConceptMapEntry,
    ExplanationExemplar,
    KnowledgeReserveEntry,
    KnowledgeReserveRegistry,
    KnowledgeSelectionRequest,
    TeachingSequence,
    build_default_teaching_canon,
    knowledge_routing_decision_from_selection,
    render_selected_teaching_knowledge,
    select_teaching_knowledge,
)

_BANNED_CITATION_TOKENS = (
    "https://",
    "http://",
    "doi:",
    "arxiv:",
    "Smith et al.",
    "[1]",
)


def _entry(entry_id: str = "reserve-bayes") -> KnowledgeReserveEntry:
    return KnowledgeReserveEntry(
        entry_id=entry_id,
        title="Bayesian updating",
        summary="Use prior evidence and new observations to revise belief.",
        body="Longer reserve note that should stay out of compact prompt rendering.",
        tags=("bayes", "probability"),
        languages=("en",),
        task_modes=("reply",),
        source="teaching-note://bayes",
        provenance={"author": "unit-test"},
        priority=0.7,
    )


def _exemplar(exemplar_id: str = "exemplar-bayes") -> ExplanationExemplar:
    return ExplanationExemplar(
        exemplar_id=exemplar_id,
        title="Coin-flip Bayesian walkthrough",
        query_terms=("bayes", "probability", "coin"),
        teaching_modes=("walkthrough",),
        task_modes=("reply",),
        language="en",
        prompt_pattern="When asked for Bayesian intuition, use a small coin example.",
        explanation="Start with a prior, observe flips, then revise the posterior.",
        source="teaching-exemplar://bayes-coin",
        provenance={"author": "unit-test"},
        priority=0.8,
    )


def _sequence(sequence_id: str = "sequence-bayes") -> TeachingSequence:
    return TeachingSequence(
        sequence_id=sequence_id,
        title="Bayesian explanation sequence",
        query_terms=("bayes", "probability", "posterior"),
        teaching_modes=("walkthrough",),
        task_modes=("reply",),
        language="en",
        steps=(
            "Name the prior.",
            "Introduce the new observation.",
            "Explain the posterior change.",
        ),
        source="teaching-sequence://bayes",
        provenance={"author": "unit-test"},
        priority=0.6,
    )


def test_knowledge_schema_accepts_valid_records_and_requires_provenance():
    concept = ConceptMapEntry(
        concept_id="concept-bayes",
        label="Bayesian updating",
        summary="Belief revision from prior and evidence.",
        prerequisites=("probability",),
        related_concepts=("posterior",),
        tags=("bayes",),
        source="concept-map://bayes",
        provenance={"author": "unit-test"},
    )

    assert _entry().source == "teaching-note://bayes"
    assert _exemplar().query_terms == ("bayes", "probability", "coin")
    assert _sequence().steps[0] == "Name the prior."
    assert concept.related_concepts == ("posterior",)

    with pytest.raises(ValidationError):
        KnowledgeReserveEntry(
            entry_id="missing-source",
            title="Missing source",
            summary="Invalid.",
            body="Invalid.",
            provenance={"author": "unit-test"},
        )
    with pytest.raises(ValidationError):
        ExplanationExemplar(
            exemplar_id="missing-provenance",
            title="Missing provenance",
            query_terms=("bayes",),
            prompt_pattern="Invalid.",
            explanation="Invalid.",
            source="teaching-exemplar://invalid",
        )


def test_teaching_knowledge_selection_is_deterministic_and_stable():
    registry = KnowledgeReserveRegistry.from_entries(
        entries=(_entry("reserve-b"), _entry("reserve-a")),
        exemplars=(_exemplar(),),
        teaching_sequences=(_sequence(),),
    )
    request = KnowledgeSelectionRequest(
        query_text="Explain bayes probability with a coin.",
        task_mode="reply",
        language="en",
        teaching_mode="walkthrough",
        max_items=3,
        max_tokens=180,
    )

    first = select_teaching_knowledge(registry, request)
    second = select_teaching_knowledge(registry, request)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.estimated_tokens <= request.max_tokens
    assert first.reason_codes[:2] == (
        "knowledge_selection_candidate",
        "knowledge_selection_selected",
    )
    assert render_selected_teaching_knowledge(first) == first.rendered_text


def test_teaching_knowledge_routing_decision_is_public_safe():
    registry = KnowledgeReserveRegistry.from_entries(
        entries=(_entry(),),
        exemplars=(_exemplar(),),
        teaching_sequences=(_sequence(),),
    )
    request = KnowledgeSelectionRequest(
        query_text="Explain bayes probability with a coin.",
        task_mode="reply",
        language="en",
        teaching_mode="walkthrough",
        max_items=3,
        max_tokens=180,
    )
    result = select_teaching_knowledge(registry, request)
    decision = knowledge_routing_decision_from_selection(
        result=result,
        request=request,
        selection_kind="auto",
    )
    payload = decision.as_dict()
    encoded = str(payload)

    assert decision.available is True
    assert payload["summary"].startswith("3 teaching knowledge items selected")
    assert payload["selected_items"][0]["item_id"] == "reserve-bayes"
    assert payload["selected_items"][0]["source_label"] == "teaching-note"
    assert "knowledge_route:reserve:reserve-bayes" in payload["reason_codes"]
    assert result.rendered_text not in encoded
    assert "Longer reserve note" not in encoded
    assert "prompt_pattern" not in encoded
    assert "steps" not in encoded


def test_teaching_selection_requires_explicit_task_query_language_and_mode_match():
    registry = KnowledgeReserveRegistry.from_entries(exemplars=(_exemplar(),))

    no_query = select_teaching_knowledge(
        registry,
        KnowledgeSelectionRequest(query_text="", task_mode="reply"),
    )
    wrong_task = select_teaching_knowledge(
        registry,
        KnowledgeSelectionRequest(query_text="bayes", task_mode="planning"),
    )
    wrong_language = select_teaching_knowledge(
        registry,
        KnowledgeSelectionRequest(query_text="bayes", task_mode="reply", language="zh"),
    )
    wrong_mode = select_teaching_knowledge(
        registry,
        KnowledgeSelectionRequest(
            query_text="bayes",
            task_mode="reply",
            teaching_mode="drill",
        ),
    )
    selected = select_teaching_knowledge(
        registry,
        KnowledgeSelectionRequest(
            query_text="bayes",
            task_mode="reply",
            teaching_mode="walkthrough",
        ),
    )

    assert no_query.selected_exemplars == ()
    assert wrong_task.selected_exemplars == ()
    assert wrong_language.selected_exemplars == ()
    assert wrong_mode.selected_exemplars == ()
    assert selected.selected_exemplars[0].exemplar_id == "exemplar-bayes"


def test_teaching_knowledge_rendering_is_compact_and_omits_full_reserve_body():
    registry = KnowledgeReserveRegistry.from_entries(
        entries=(_entry(),),
        exemplars=(_exemplar(),),
        teaching_sequences=(_sequence(),),
    )
    result = select_teaching_knowledge(
        registry,
        KnowledgeSelectionRequest(
            query_text="bayes posterior probability",
            task_mode="reply",
            teaching_mode="walkthrough",
            max_items=3,
            max_tokens=120,
        ),
    )

    assert result.rendered_text
    assert approximate_token_count(result.rendered_text) == result.estimated_tokens
    assert result.estimated_tokens <= 120
    assert "Longer reserve note" not in result.rendered_text
    assert "source=" in result.rendered_text
    assert "provenance=" in result.rendered_text


def test_default_teaching_canon_is_small_non_empty_and_source_anchored():
    registry = build_default_teaching_canon()
    result = select_teaching_knowledge(
        registry,
        KnowledgeSelectionRequest(
            query_text="Explain recursion with a small example.",
            task_mode="reply",
            language="en",
            teaching_mode="clarify",
            max_items=2,
            max_tokens=96,
        ),
    )

    assert registry.is_empty is False
    assert len(registry.entries) == 1
    assert len(registry.exemplars) == 6
    assert len(registry.teaching_sequences) == 3
    assert (
        result.rendered_text
        == select_teaching_knowledge(
            registry,
            KnowledgeSelectionRequest(
                query_text="Explain recursion with a small example.",
                task_mode="reply",
                language="en",
                teaching_mode="clarify",
                max_items=2,
                max_tokens=96,
            ),
        ).rendered_text
    )
    assert result.estimated_tokens <= 96
    assert "blink-default-teaching-canon" in result.rendered_text
    assert "provenance=curator:blink,kind:internal-pedagogy,version:2026-04" in (
        result.rendered_text
    )
    assert "Use only available sources or repo-visible context" not in result.rendered_text
    assert all(token not in result.rendered_text for token in _BANNED_CITATION_TOKENS)
    assert "exemplar:walkthrough_small_example" in result.rendered_text


def test_default_teaching_canon_selects_new_compact_pedagogy_records():
    registry = build_default_teaching_canon()
    cases = (
        (
            "Debug this failing function with one hypothesis and one minimal repro.",
            "exemplar:debugging_hypothesis_one_change",
        ),
        (
            "I think recursion means an infinite loop; correct my misconception.",
            "exemplar:misconception_repair_without_shame",
        ),
        (
            "Answer from sources and cite the documentation if evidence is uncertain.",
            "exemplar:source_grounded_answer_with_limits",
        ),
        (
            "Give me one practice prompt with an answer key for recursion.",
            "sequence:practice_prompt_with_answer_key",
        ),
    )

    for query_text, expected_id in cases:
        result = select_teaching_knowledge(
            registry,
            KnowledgeSelectionRequest(
                query_text=query_text,
                task_mode="reply",
                language="en",
                teaching_mode="clarify",
                max_items=2,
                max_tokens=96,
            ),
        )

        assert expected_id in result.rendered_text
        assert result.estimated_tokens <= 96
        assert "source=blink-default-teaching-canon" in result.rendered_text
        assert "provenance=curator:blink,kind:internal-pedagogy,version:2026-04" in (
            result.rendered_text
        )
        assert "Use only available sources or repo-visible context" not in result.rendered_text
        assert all(token not in result.rendered_text for token in _BANNED_CITATION_TOKENS)


def test_default_teaching_canon_selects_chinese_technical_bridge():
    registry = build_default_teaching_canon()

    result = select_teaching_knowledge(
        registry,
        KnowledgeSelectionRequest(
            query_text="请解释递归调试思路",
            task_mode="reply",
            language="zh",
            teaching_mode="clarify",
            max_items=2,
            max_tokens=96,
        ),
    )

    assert result.rendered_text
    assert result.estimated_tokens <= 96
    assert "exemplar:chinese_technical_explanation_bridge" in result.rendered_text
    assert "source=blink-default-teaching-canon" in result.rendered_text
    assert "provenance=" in result.rendered_text
