from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from blink.brain.evals.persona_memory import (
    PERSONA_MEMORY_REQUIRED_METRICS,
    build_persona_memory_eval_suite,
    evaluate_persona_memory_eval_suite,
    render_persona_memory_metrics_rows,
)

pytestmark = pytest.mark.brain_property


@given(order=st.permutations((0, 1, 2, 3, 4)))
def test_persona_memory_eval_rows_are_order_invariant_and_bounded(order):
    suite = build_persona_memory_eval_suite()[:5]
    baseline = render_persona_memory_metrics_rows(evaluate_persona_memory_eval_suite(suite))
    shuffled = render_persona_memory_metrics_rows(
        evaluate_persona_memory_eval_suite([suite[index] for index in order])
    )

    assert shuffled == baseline
    for row in baseline:
        for metric_name in PERSONA_MEMORY_REQUIRED_METRICS:
            assert 0.0 <= row[metric_name] <= 1.0
