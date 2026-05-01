from __future__ import annotations

from dataclasses import dataclass

from blink.brain.knowledge import (
    KnowledgeSelectionRequest,
    build_default_teaching_canon,
    select_teaching_knowledge,
)


@dataclass(frozen=True)
class TeachingQualityCase:
    case_id: str
    query_text: str
    language: str
    teaching_mode: str
    expected_id: str
    max_tokens: int
    must_include: tuple[str, ...]
    must_not_include: tuple[str, ...]


_BANNED_EXTERNAL_CITATION_PATTERNS = (
    "https://",
    "http://",
    "doi:",
    "arxiv:",
    "Smith et al.",
    "[1]",
)


TEACHING_QUALITY_CASES = (
    TeachingQualityCase(
        case_id="debugging_one_hypothesis",
        query_text="Debug this failing function with one hypothesis and one minimal repro.",
        language="en",
        teaching_mode="clarify",
        expected_id="exemplar:debugging_hypothesis_one_change",
        max_tokens=96,
        must_include=("observed symptom", "one testable change"),
        must_not_include=("Use only available sources or repo-visible context",),
    ),
    TeachingQualityCase(
        case_id="misconception_repair_direct",
        query_text="I think recursion means an infinite loop; correct my misconception.",
        language="en",
        teaching_mode="clarify",
        expected_id="exemplar:misconception_repair_without_shame",
        max_tokens=96,
        must_include=("State the correction directly", "avoid shaming language"),
        must_not_include=("you are wrong", "Use only available sources or repo-visible context"),
    ),
    TeachingQualityCase(
        case_id="source_grounded_limits",
        query_text="Answer from sources and cite the documentation if evidence is uncertain.",
        language="en",
        teaching_mode="clarify",
        expected_id="exemplar:source_grounded_answer_with_limits",
        max_tokens=96,
        must_include=("available context", "avoid invented citations"),
        must_not_include=("Use only available sources or repo-visible context",),
    ),
    TeachingQualityCase(
        case_id="chinese_technical_bridge",
        query_text="请解释递归调试思路",
        language="zh",
        teaching_mode="clarify",
        expected_id="exemplar:chinese_technical_explanation_bridge",
        max_tokens=96,
        must_include=("用中文解释核心思路", "technical example"),
        must_not_include=("Use only available sources or repo-visible context",),
    ),
    TeachingQualityCase(
        case_id="practice_prompt_answer_key",
        query_text="Give me one practice prompt with an answer key for recursion.",
        language="en",
        teaching_mode="clarify",
        expected_id="sequence:practice_prompt_with_answer_key",
        max_tokens=96,
        must_include=("practice prompt", "expected answer"),
        must_not_include=("Use only available sources or repo-visible context",),
    ),
)


def _selected_ids(result) -> tuple[str, ...]:
    return (
        *(entry.entry_id for entry in result.selected_entries),
        *(exemplar.exemplar_id for exemplar in result.selected_exemplars),
        *(sequence.sequence_id for sequence in result.selected_sequences),
    )


def test_teaching_quality_case_ids_are_stable():
    assert [case.case_id for case in TEACHING_QUALITY_CASES] == [
        "debugging_one_hypothesis",
        "misconception_repair_direct",
        "source_grounded_limits",
        "chinese_technical_bridge",
        "practice_prompt_answer_key",
    ]


def test_teaching_quality_cases_select_expected_compact_records():
    registry = build_default_teaching_canon()

    for case in TEACHING_QUALITY_CASES:
        first = select_teaching_knowledge(
            registry,
            KnowledgeSelectionRequest(
                query_text=case.query_text,
                task_mode="reply",
                language=case.language,
                teaching_mode=case.teaching_mode,
                max_items=2,
                max_tokens=case.max_tokens,
            ),
        )
        second = select_teaching_knowledge(
            registry,
            KnowledgeSelectionRequest(
                query_text=case.query_text,
                task_mode="reply",
                language=case.language,
                teaching_mode=case.teaching_mode,
                max_items=2,
                max_tokens=case.max_tokens,
            ),
        )

        assert first.model_dump(mode="json") == second.model_dump(mode="json")
        assert case.expected_id in _selected_ids(first)
        assert case.expected_id in first.rendered_text
        assert first.estimated_tokens <= case.max_tokens
        assert "source=blink-default-teaching-canon" in first.rendered_text
        assert "provenance=curator:blink,kind:internal-pedagogy,version:2026-04" in (
            first.rendered_text
        )
        assert all(text in first.rendered_text for text in case.must_include)
        assert all(text not in first.rendered_text for text in case.must_not_include)
        assert all(token not in first.rendered_text for token in _BANNED_EXTERNAL_CITATION_PATTERNS)


def test_teaching_quality_cases_are_reply_and_query_opt_in():
    registry = build_default_teaching_canon()
    case = TEACHING_QUALITY_CASES[0]

    wrong_task = select_teaching_knowledge(
        registry,
        KnowledgeSelectionRequest(
            query_text=case.query_text,
            task_mode="planning",
            language=case.language,
            teaching_mode=case.teaching_mode,
        ),
    )
    empty_query = select_teaching_knowledge(
        registry,
        KnowledgeSelectionRequest(
            query_text="",
            task_mode="reply",
            language=case.language,
            teaching_mode=case.teaching_mode,
        ),
    )

    assert wrong_task.rendered_text == ""
    assert "knowledge_selection_empty" in wrong_task.reason_codes
    assert empty_query.rendered_text == ""
    assert "query_empty" in empty_query.reason_codes
