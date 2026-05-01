"""Deterministic context-budget profiles for Blink continuity state."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

from blink.brain.context.policy import (
    BrainContextTask,
    BrainContextTraceVerbosity,
    get_brain_context_mode_policy,
)


def approximate_token_count(text: str) -> int:
    """Return a deterministic approximate token count for plain text."""
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return max(1, ceil(len(normalized) / 4))


@dataclass(frozen=True)
class BrainContextBudgetProfile:
    """Budget profile for one context-selection task."""

    task: str
    max_tokens: int
    section_caps: dict[str, int] = field(default_factory=dict)
    dynamic_token_reserve: int | None = None
    trace_verbosity: BrainContextTraceVerbosity | None = None

    def resolved(
        self,
        task: BrainContextTask | str | None = None,
    ) -> "BrainContextBudgetProfile":
        """Return this profile with policy defaults filled in."""
        task_value = (
            task.value
            if isinstance(task, BrainContextTask)
            else str(task).strip()
            if task is not None
            else self.task
        )
        policy = get_brain_context_mode_policy(task_value)
        if task_value != self.task:
            raise ValueError(
                f"Budget profile task mismatch: profile={self.task} requested={task_value}"
            )
        return BrainContextBudgetProfile(
            task=task_value,
            max_tokens=self.max_tokens,
            section_caps={**policy.section_caps, **self.section_caps},
            dynamic_token_reserve=(
                self.dynamic_token_reserve
                if self.dynamic_token_reserve is not None
                else policy.dynamic_token_reserve
            ),
            trace_verbosity=self.trace_verbosity or policy.trace_verbosity,
        )

    @classmethod
    def for_task(cls, task: str) -> "BrainContextBudgetProfile":
        """Return the default budget profile for one context task."""
        policy = get_brain_context_mode_policy(task)
        return cls(
            task=policy.task.value,
            max_tokens=policy.max_tokens,
            section_caps=dict(policy.section_caps),
            dynamic_token_reserve=policy.dynamic_token_reserve,
            trace_verbosity=policy.trace_verbosity,
        )
