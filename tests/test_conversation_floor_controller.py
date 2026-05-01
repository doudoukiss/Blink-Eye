import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from blink.cli import local_browser
from blink.cli.local_browser import LocalBrowserConfig, create_app
from blink.interaction import (
    ActorEventContext,
    BrowserPerformanceEventBus,
    ConversationFloorController,
    ConversationFloorInput,
    ConversationFloorStatus,
    classify_floor_phrase,
    classify_floor_text,
)
from blink.interaction.performance_events import BrowserInteractionMode

ROOT = Path(__file__).resolve().parents[1]
FLOOR_SCHEMA_PATH = ROOT / "schemas" / "conversation_floor_state.schema.json"
ACTOR_STATE_SCHEMA_PATH = ROOT / "schemas" / "browser_actor_state_v2.schema.json"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "floor_cases_zh_en.jsonl"

PRIMARY_BROWSER_PROFILES = (
    (
        "browser-zh-melo",
        local_browser.Language.ZH,
        "local-http-wav",
        "local-http-wav/MeloTTS",
    ),
    (
        "browser-en-kokoro",
        local_browser.Language.EN,
        "kokoro",
        "kokoro/English",
    ),
)


def _validator(path: Path) -> Draft202012Validator:
    return Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))


def _assert_valid(path: Path, payload: dict[str, object]) -> None:
    errors = sorted(_validator(path).iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _fixture_cases() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("case", _fixture_cases(), ids=lambda case: case["name"])
def test_conversation_floor_controller_distinguishes_deterministic_fixtures(case):
    controller = ConversationFloorController(
        profile=case["profile"],
        language=case["language"],
        protected_playback=case.get("protected_playback", True),
        barge_in_armed=case.get("barge_in_armed", False),
        echo_safe=case.get("echo_safe", case.get("barge_in_armed", False)),
    )

    update = None
    for raw_input in case["inputs"]:
        update = controller.apply(ConversationFloorInput(**raw_input))

    assert update is not None
    payload = update.state.as_dict()
    _assert_valid(FLOOR_SCHEMA_PATH, payload)
    assert payload["state"] == case["expected_state"]
    assert payload[case["expected_state"]] is True
    assert payload["floor_model_version"] == 3
    assert payload["sub_state"] == case["expected_sub_state"]
    assert payload["last_text_kind"] == case["expected_text_kind"]
    assert payload["phrase_class"] == case["expected_phrase_class"]
    assert payload["yield_decision"] == case["expected_yield_decision"]
    assert case["expected_reason"] in payload["reason_codes"]
    if "expected_confidence_bucket" in case:
        assert payload["phrase_confidence_bucket"] == case["expected_confidence_bucket"]
        assert payload["low_confidence_transcript"] is True
    assert payload["profile"] == case["profile"]
    assert payload["language"] == case["language"]
    encoded = json.dumps(payload, ensure_ascii=False)
    for raw_input in case["inputs"]:
        if raw_input.get("text"):
            assert raw_input["text"] not in encoded


def test_backchannels_do_not_terminate_assistant_floor_when_policy_continues():
    for text in ("嗯", "对", "继续", "ok", "yeah", "right", "go on"):
        controller = ConversationFloorController(
            profile="browser-zh-melo",
            language="zh",
            protected_playback=True,
            barge_in_armed=False,
        )
        controller.apply(ConversationFloorInput(input_type="tts_started"))
        update = controller.apply(ConversationFloorInput(input_type="stt_interim", text=text))
        payload = update.state.as_dict()

        assert payload["state"] == ConversationFloorStatus.ASSISTANT_HAS_FLOOR.value
        assert payload["assistant_has_floor"] is True
        assert payload["sub_state"] in {"ignored_backchannel", "assistant_holding_floor"}
        assert "floor:continued" in payload["reason_codes"]


def test_explicit_interruption_phrases_trigger_repair_yield_when_armed_or_echo_safe():
    armed = ConversationFloorController(
        profile="browser-en-kokoro",
        language="en",
        protected_playback=False,
        barge_in_armed=True,
    )
    armed.apply(ConversationFloorInput(input_type="tts_started"))
    armed_update = armed.apply(ConversationFloorInput(input_type="stt_interim", text="hold on"))

    echo_safe = ConversationFloorController(
        profile="browser-en-kokoro",
        language="en",
        protected_playback=True,
        barge_in_armed=False,
        echo_safe=True,
    )
    echo_safe.apply(ConversationFloorInput(input_type="tts_started"))
    echo_update = echo_safe.apply(
        ConversationFloorInput(
            input_type="stt_interim",
            text="wait",
            echo_safe=True,
        )
    )

    assert armed_update.state.as_dict()["state"] == "repair"
    assert armed_update.state.as_dict()["sub_state"] == "repair_requested"
    assert "floor:yielded" in armed_update.state.as_dict()["reason_codes"]
    assert echo_update.state.as_dict()["state"] == "repair"
    assert echo_update.state.as_dict()["sub_state"] == "repair_requested"
    assert "floor:yielded" in echo_update.state.as_dict()["reason_codes"]


def test_text_classifier_covers_chinese_and_english_floor_phrases():
    assert classify_floor_text("嗯").value == "backchannel"
    assert classify_floor_text("right").value == "confirmation"
    assert classify_floor_phrase("继续").phrase_class == "continuer"
    assert classify_floor_phrase("go on").phrase_class == "continuer"
    assert classify_floor_text("让我想想").value == "hesitation"
    assert classify_floor_text("不对，我不是说这个").value == "correction"
    assert classify_floor_text("wait a second").value == "explicit_interruption"


def test_floor_transition_actor_event_is_public_safe_and_schema_v2():
    bus = BrowserPerformanceEventBus(
        max_events=10,
        actor_context_provider=lambda: ActorEventContext(
            profile="browser-en-kokoro",
            language="en",
            tts_backend="kokoro",
            tts_label="kokoro/English",
            vision_backend="moondream",
        ),
    )

    bus.emit(
        event_type="floor.transition",
        source="floor",
        mode=BrowserInteractionMode.INTERRUPTED,
        metadata={
            "floor_state": "overlap",
            "floor_sub_state": "overlap_candidate",
            "previous_floor_state": "assistant_has_floor",
            "input_type": "stt_interim",
            "text_kind": "explicit_interruption",
            "phrase_class": "explicit_interrupt",
            "phrase_confidence_bucket": "high",
            "yield_decision": "yield_to_user",
            "raw_text": "hold on with private transcript",
        },
        reason_codes=["floor:explicit_interruption", "floor:yielded"],
    )
    actor_payload = bus.actor_payload()
    event = actor_payload["events"][0]
    encoded = json.dumps(event, ensure_ascii=False)

    assert actor_payload["schema_version"] == 2
    assert event["event_type"] == "floor_transition"
    assert event["mode"] == "interrupted"
    assert event["metadata"]["floor_state"] == "overlap"
    assert event["metadata"]["floor_sub_state"] == "overlap_candidate"
    assert event["metadata"]["text_kind"] == "explicit_interruption"
    assert event["metadata"]["phrase_class"] == "explicit_interrupt"
    assert event["metadata"]["yield_decision"] == "yield_to_user"
    assert "raw_text" not in event["metadata"]
    assert "private transcript" not in encoded


def _browser_config(
    *,
    profile: str,
    language: local_browser.Language,
    tts_backend: str,
    tts_label: str,
) -> LocalBrowserConfig:
    return LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=language,
        stt_backend="mlx-whisper",
        tts_backend=tts_backend,
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001" if tts_backend == "local-http-wav" else None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=False,
        allow_barge_in=False,
        tts_runtime_label=tts_label,
        config_profile=profile,
    )


def _test_client(config: LocalBrowserConfig):
    pytest.importorskip("fastapi")
    pytest.importorskip("aiortc")
    pytest.importorskip("av")
    from fastapi.testclient import TestClient

    app, _ = create_app(config)
    return app, TestClient(app)


def test_actor_state_exposes_conversation_floor_for_both_primary_profiles():
    states = []
    for profile, language, tts_backend, tts_label in PRIMARY_BROWSER_PROFILES:
        app, client = _test_client(
            _browser_config(
                profile=profile,
                language=language,
                tts_backend=tts_backend,
                tts_label=tts_label,
            )
        )
        app.state.blink_apply_conversation_floor_input(
            ConversationFloorInput(input_type="tts_started")
        )
        payload = client.get("/api/runtime/actor-state").json()
        _assert_valid(ACTOR_STATE_SCHEMA_PATH, payload)
        _assert_valid(FLOOR_SCHEMA_PATH, payload["conversation_floor"])
        assert payload["conversation_floor"]["state"] == "assistant_has_floor"
        assert payload["conversation_floor"]["sub_state"] == "assistant_holding_floor"
        assert payload["conversation_floor"]["floor_model_version"] == 3
        assert payload["conversation_floor"]["profile"] == profile
        assert payload["conversation_floor"]["language"] == language.value
        states.append(payload)

    assert set(states[0]["conversation_floor"]) == set(states[1]["conversation_floor"])
