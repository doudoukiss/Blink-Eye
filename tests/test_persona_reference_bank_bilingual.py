import json
from pathlib import Path

from jsonschema import Draft202012Validator

from blink.brain.persona import (
    persona_reference_anchors_by_situation_v3,
    persona_reference_bank,
    persona_reference_bank_v3,
    required_persona_reference_anchor_keys_v3,
    required_persona_reference_scenarios,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "persona_reference.schema.json"
ANCHOR_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "persona_reference_anchor_v3.schema.json"
)


def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def _anchor_validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(ANCHOR_SCHEMA_PATH.read_text(encoding="utf-8")))


def _assert_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _assert_anchor_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_anchor_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def test_reference_bank_has_required_bilingual_scenario_coverage():
    references = [reference.as_dict() for reference in persona_reference_bank()]
    required = set(required_persona_reference_scenarios())

    assert {item["locale"] for item in references} == {"zh", "en"}
    for locale in ("zh", "en"):
        scenarios = {item["scenario"] for item in references if item["locale"] == locale}
        assert required <= scenarios

    for payload in references:
        _assert_schema_valid(payload)


def test_bilingual_references_are_equivalent_but_not_direct_copies():
    zh = {item.scenario: item.as_dict() for item in persona_reference_bank(locale="zh")}
    en = {item.scenario: item.as_dict() for item in persona_reference_bank(locale="en")}

    assert set(zh) == set(en) == set(required_persona_reference_scenarios())
    for scenario in required_persona_reference_scenarios():
        assert zh[scenario]["example_output"] != en[scenario]["example_output"]
        assert zh[scenario]["stance"] != en[scenario]["stance"]
        assert zh[scenario]["response_shape"]
        assert en[scenario]["response_shape"]


def test_references_do_not_expose_private_prompts_or_fake_biography():
    encoded = json.dumps(
        [reference.as_dict() for reference in persona_reference_bank()],
        ensure_ascii=False,
        sort_keys=True,
    ).lower()

    for banned in (
        "system_prompt",
        "developer_prompt",
        "hidden prompt",
        "api_key",
        "bearer ",
        "human_identity",
        "identity cloning",
        "romantic",
        "exclusive",
    ):
        assert banned not in encoded


def test_runtime_reference_summaries_exclude_examples_and_forbidden_moves():
    reference = persona_reference_bank(locale="en")[0]
    summary = reference.public_summary()

    assert "example_input" not in summary
    assert "example_output" not in summary
    assert "forbidden_moves" not in summary
    assert summary["reference_id"] == reference.id
    assert summary["scenario"] == reference.scenario


def test_reference_bank_v3_has_required_public_anchor_coverage():
    bank = persona_reference_bank_v3().as_dict()
    anchors = bank["anchors"]
    required = set(required_persona_reference_anchor_keys_v3())

    assert bank["schema_version"] == 3
    assert bank["anchor_count"] == len(required)
    assert {anchor["situation_key"] for anchor in anchors} == required
    assert set(persona_reference_anchors_by_situation_v3()) == required
    for anchor in anchors:
        _assert_anchor_schema_valid(anchor)
        assert anchor["zh_example"]
        assert anchor["en_example"]
        assert anchor["zh_example"] != anchor["en_example"]
        assert anchor["behavior_constraints"]
        assert anchor["negative_examples"]
        assert f"persona_anchor:{anchor['situation_key']}" in anchor["reason_codes"]


def test_reference_bank_v3_summaries_are_text_free_and_public_safe():
    encoded = json.dumps(persona_reference_bank_v3().as_dict(), ensure_ascii=False).lower()
    for required_marker in (
        "invent a human backstory",
        "repetitive catchphrases",
        "raw image data",
        "unsupported camera",
        "unsupported tts",
        "raw memory records",
    ):
        assert required_marker in encoded

    for banned in (
        "system_prompt",
        "developer_prompt",
        "hidden prompt",
        "api_key",
        "bearer ",
        "raw_audio",
        "raw_image",
        "sdp_offer",
        "ice_candidate",
        "memory_body",
    ):
        assert banned not in encoded

    summary = persona_reference_anchors_by_situation_v3()["visual_grounding"].public_summary()
    assert "zh_example" not in summary
    assert "en_example" not in summary
    assert "behavior_constraints" not in summary
    assert "negative_examples" not in summary
    assert summary["situation_key"] == "visual_grounding"
