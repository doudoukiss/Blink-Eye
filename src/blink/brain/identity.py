"""Pinned Blink brain identity and policy blocks."""

from __future__ import annotations

from importlib import resources

from blink.project_identity import PROJECT_IDENTITY
from blink.transcriptions.language import Language

BRAIN_CONTEXT_HEADER = "[BLINK_BRAIN_CONTEXT]"

_DEFAULT_AGENT_BLOCK_FILES = {
    "identity": "IDENTITY.md",
    "policy": "POLICY.md",
    "style": "STYLE.md",
    "action_library": "ACTION_LIBRARY.md",
    "persona": "PERSONA.md",
    "voice": "VOICE.md",
    "relationship_style": "RELATIONSHIP_STYLE.md",
    "teaching_style": "TEACHING_STYLE.md",
}


def _default_path(name: str):
    return resources.files("blink.brain.defaults").joinpath(name)


def load_default_agent_blocks() -> dict[str, str]:
    """Load the checked-in default brain blocks."""
    return {
        name: _default_path(filename).read_text(encoding="utf-8").strip()
        for name, filename in _DEFAULT_AGENT_BLOCK_FILES.items()
    }


def base_brain_system_prompt(language: Language) -> str:
    """Return the thin static system prompt for brain-enabled local runtimes."""
    if language.value.lower().startswith(("zh", "cmn")):
        return (
            f"你是 {PROJECT_IDENTITY.display_name}。"
            "请始终遵循系统消息中的 BLINK_BRAIN_CONTEXT。"
            "不要虚构未声明的能力。"
        )
    return (
        f"You are {PROJECT_IDENTITY.display_name}. "
        "Always follow the system-provided BLINK_BRAIN_CONTEXT. "
        "Do not invent undeclared capabilities."
    )
