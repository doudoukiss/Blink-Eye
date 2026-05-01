import json

import pytest

from blink.brain.context import BrainContextTask, compile_context_packet_from_surface
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.identity import base_brain_system_prompt, load_default_agent_blocks
from blink.brain.persona import (
    PERSONA_ALLOWED_ONTOLOGICAL_STATUSES,
    PERSONA_CANONICAL_NAME,
    BrainPersonaFrame,
    BrainPersonaModality,
    BrainPersonaState,
    BrainPersonaTaskMode,
    RelationshipStyleDefaultsSpec,
    SelfPersonaCoreSpec,
    TeachingStyleDefaultsSpec,
    compile_persona_frame,
    compile_self_persona_core,
    load_persona_defaults,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language

_DEFAULT_PERSONA_BLOCKS = load_default_agent_blocks()
_DEFAULT_PERSONA_DEFAULTS = load_persona_defaults(_DEFAULT_PERSONA_BLOCKS)


def _structured_doc(payload: dict, *, title: str = "Structured Doc") -> str:
    return (
        f"# {title}\n\n"
        "This document contains a structured payload for the Blink persona kernel.\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"
    )


def _default_blocks_with_persona_payload(payload: dict) -> dict[str, str]:
    blocks = dict(_DEFAULT_PERSONA_BLOCKS)
    blocks["persona"] = _structured_doc(payload, title="Persona")
    return blocks


def test_load_default_agent_blocks_includes_persona_kernel_docs():
    blocks = load_default_agent_blocks()

    assert {
        "identity",
        "policy",
        "style",
        "action_library",
        "persona",
        "voice",
        "relationship_style",
        "teaching_style",
    } == set(blocks)
    assert all(content.strip() for content in blocks.values())


def test_compile_persona_frame_is_deterministic_and_roundtrips():
    blocks = load_default_agent_blocks()
    first = compile_persona_frame(
        agent_blocks=blocks,
        task_mode=BrainPersonaTaskMode.REPLY,
        modality=BrainPersonaModality.TEXT,
        state=BrainPersonaState.neutral(current_mode="reply"),
    )
    second = compile_persona_frame(
        agent_blocks=blocks,
        task_mode=BrainPersonaTaskMode.REPLY,
        modality=BrainPersonaModality.TEXT,
        state=BrainPersonaState.neutral(current_mode="reply"),
    )
    restored = BrainPersonaFrame.from_dict(first.as_dict())

    assert first.as_dict() == second.as_dict()
    assert restored is not None
    assert restored.as_dict() == first.as_dict()
    assert restored.compiled_from_blocks == ("persona", "voice")
    assert restored.guardrails


def test_persona_defaults_and_frame_preserve_canonical_identity_invariants():
    blocks = load_default_agent_blocks()
    defaults = load_persona_defaults(blocks)
    frame = compile_persona_frame(
        agent_blocks=blocks,
        task_mode=BrainPersonaTaskMode.PLANNING,
        modality=BrainPersonaModality.VOICE,
    )

    assert defaults.charter.canonical_name == PERSONA_CANONICAL_NAME
    assert defaults.charter.ontological_status in PERSONA_ALLOWED_ONTOLOGICAL_STATUSES
    assert defaults.charter.human_identity_claims_allowed is False
    assert frame.charter.canonical_name == PERSONA_CANONICAL_NAME
    assert frame.charter.ontological_status in PERSONA_ALLOWED_ONTOLOGICAL_STATUSES
    assert frame.charter.human_identity_claims_allowed is False
    assert frame.task_mode == BrainPersonaTaskMode.PLANNING.value
    assert frame.modality == BrainPersonaModality.VOICE.value


def test_persona_defaults_promote_public_schema_and_self_persona_core():
    defaults = load_persona_defaults(load_default_agent_blocks())
    persona_core = compile_self_persona_core(load_default_agent_blocks())

    assert isinstance(defaults.relationship_defaults, RelationshipStyleDefaultsSpec)
    assert isinstance(defaults.teaching_defaults, TeachingStyleDefaultsSpec)
    assert isinstance(persona_core, SelfPersonaCoreSpec)
    assert persona_core.canonical_name == PERSONA_CANONICAL_NAME
    assert set(persona_core.source_block_fingerprints) == {
        "persona",
        "voice",
        "relationship_style",
        "teaching_style",
    }


@pytest.mark.parametrize(
    ("blocks", "error_fragment"),
    [
        pytest.param(
            _default_blocks_with_persona_payload(
                {
                    "schema_version": 1,
                    "charter": {
                        **_DEFAULT_PERSONA_DEFAULTS.charter.model_dump(),
                        "canonical_name": "Not Blink",
                    },
                    "traits": _DEFAULT_PERSONA_DEFAULTS.traits.model_dump(),
                }
            ),
            "Invalid structured persona defaults payload",
            id="wrong_canonical_name",
        ),
        pytest.param(
            _default_blocks_with_persona_payload(
                {
                    "schema_version": 1,
                    "charter": {
                        **_DEFAULT_PERSONA_DEFAULTS.charter.model_dump(),
                        "human_identity_claims_allowed": True,
                    },
                    "traits": _DEFAULT_PERSONA_DEFAULTS.traits.model_dump(),
                }
            ),
            "Invalid structured persona defaults payload",
            id="human_identity_enabled",
        ),
        pytest.param(
            _default_blocks_with_persona_payload(
                {
                    "schema_version": 1,
                    "charter": _DEFAULT_PERSONA_DEFAULTS.charter.model_dump(),
                    "traits": {
                        **_DEFAULT_PERSONA_DEFAULTS.traits.model_dump(),
                        "warmth": 1.2,
                    },
                }
            ),
            "Invalid structured persona defaults payload",
            id="trait_out_of_range",
        ),
        pytest.param(
            _default_blocks_with_persona_payload(
                {
                    "schema_version": 1,
                    "charter": {
                        **_DEFAULT_PERSONA_DEFAULTS.charter.model_dump(),
                        "age": 19,
                    },
                    "traits": _DEFAULT_PERSONA_DEFAULTS.traits.model_dump(),
                }
            ),
            "Invalid structured persona defaults payload",
            id="unknown_field",
        ),
        pytest.param(
            {
                **_DEFAULT_PERSONA_BLOCKS,
                "persona": "# Persona\n\nNo structured payload here.\n",
            },
            "exactly one fenced json block",
            id="missing_json_fence",
        ),
        pytest.param(
            {
                **_DEFAULT_PERSONA_BLOCKS,
                "persona": (
                    _structured_doc(
                        {
                            "schema_version": 1,
                            "charter": _DEFAULT_PERSONA_DEFAULTS.charter.model_dump(),
                            "traits": _DEFAULT_PERSONA_DEFAULTS.traits.model_dump(),
                        },
                        title="Persona",
                    )
                    + "\n```json\n{}\n```\n"
                ),
            },
            "exactly one fenced json block",
            id="multiple_json_fences",
        ),
    ],
)
def test_invalid_persona_docs_fail_fast(blocks, error_fragment):
    with pytest.raises(ValueError, match=error_fragment):
        load_persona_defaults(blocks)


def test_persona_kernel_preserves_thin_prompt_and_adds_no_context_section(tmp_path):
    assert base_brain_system_prompt(Language.EN) == (
        "You are Blink. Always follow the system-provided BLINK_BRAIN_CONTEXT. "
        "Do not invent undeclared capabilities."
    )
    assert base_brain_system_prompt(Language.ZH) == (
        "你是 Blink。请始终遵循系统消息中的 BLINK_BRAIN_CONTEXT。不要虚构未声明的能力。"
    )

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="persona-kernel")
    store.ensure_default_blocks(load_default_agent_blocks())
    surface = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
    ).build(latest_user_text="Describe your identity.")

    packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Describe your identity.",
        task=BrainContextTask.REPLY,
        language=Language.EN,
        base_prompt=base_brain_system_prompt(Language.EN),
    )

    assert packet.selected_context.section("persona") is None
    assert "Blink Scholar-Companion" not in packet.prompt
    assert "Voice Profile" not in packet.prompt
