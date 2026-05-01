import json
from pathlib import Path

import pytest

from blink.cli import local_browser
from blink.cli.local_browser import LocalBrowserConfig, create_app

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


def _browser_config(
    *,
    profile: str = "browser-zh-melo",
    language: local_browser.Language = local_browser.Language.ZH,
    tts_backend: str = "local-http-wav",
    tts_label: str = "local-http-wav/MeloTTS",
    actor_surface_v2_enabled: bool = True,
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
        actor_surface_v2_enabled=actor_surface_v2_enabled,
    )


def _test_client(config: LocalBrowserConfig):
    pytest.importorskip("fastapi")
    pytest.importorskip("aiortc")
    pytest.importorskip("av")
    from fastapi.testclient import TestClient

    app, _ = create_app(config)
    return TestClient(app)


def _client_config_payload(script: str) -> dict[str, object]:
    prefix = "globalThis.BlinkRuntimeConfig = Object.freeze("
    assert prefix in script
    return json.loads(script.split(prefix, 1)[1].rsplit(");", 1)[0])


def test_browser_actor_surface_client_config_default_on_and_explicitly_disableable():
    enabled_client = _test_client(_browser_config())
    enabled_script = enabled_client.get("/api/runtime/client-config.js").text
    enabled_payload = _client_config_payload(enabled_script)

    assert enabled_payload["actor_surface_v2_enabled"] is True
    assert "actor_surface_v2:on" in enabled_payload["reason_codes"]
    assert "Secret prompt" not in enabled_script

    disabled_client = _test_client(_browser_config(actor_surface_v2_enabled=False))
    disabled_script = disabled_client.get("/api/runtime/client-config.js").text
    disabled_payload = _client_config_payload(disabled_script)

    assert disabled_payload["actor_surface_v2_enabled"] is False
    assert "actor_surface_v2:off" in disabled_payload["reason_codes"]


def test_browser_actor_surface_resolve_config_env_and_cli_precedence(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_ACTOR_SURFACE_V2", "0")
    env_off = local_browser.resolve_config(
        local_browser.build_parser().parse_args(["--config-profile", "browser-en-kokoro"])
    )
    cli_on = local_browser.resolve_config(
        local_browser.build_parser().parse_args(
            ["--config-profile", "browser-en-kokoro", "--actor-surface-v2"]
        )
    )

    monkeypatch.setenv("BLINK_LOCAL_ACTOR_SURFACE_V2", "1")
    cli_off = local_browser.resolve_config(
        local_browser.build_parser().parse_args(
            ["--config-profile", "browser-en-kokoro", "--no-actor-surface-v2"]
        )
    )

    assert env_off.actor_surface_v2_enabled is False
    assert cli_on.actor_surface_v2_enabled is True
    assert cli_off.actor_surface_v2_enabled is False


@pytest.mark.parametrize(
    ("profile", "language", "tts_backend", "tts_label"),
    PRIMARY_BROWSER_PROFILES,
)
def test_browser_actor_state_exposes_every_field_actor_surface_reads(
    profile,
    language,
    tts_backend,
    tts_label,
):
    client = _test_client(
        _browser_config(
            profile=profile,
            language=language,
            tts_backend=tts_backend,
            tts_label=tts_label,
        )
    )

    actor_state = client.get("/api/runtime/actor-state").json()
    performance_state = client.get("/api/runtime/performance-state").json()

    assert performance_state["schema_version"] == 1
    assert actor_state["schema_version"] == 2
    assert actor_state["profile"] == profile
    assert actor_state["language"] == language.value
    assert actor_state["tts"]["label"] == tts_label
    assert actor_state["vision"]["backend"] == "moondream"
    assert actor_state["protected_playback"] is True

    for top_level_key in (
        "mode",
        "tts",
        "vision",
        "interruption",
        "webrtc_audio_health",
        "active_listening",
        "speech",
        "camera_scene",
        "memory_persona",
        "degradation",
        "live_text",
    ):
        assert top_level_key in actor_state

    for live_text_key in ("partial_transcript", "final_transcript", "assistant_subtitle"):
        assert live_text_key in actor_state["live_text"]

    for camera_key in (
        "state",
        "freshness_state",
        "latest_frame_sequence",
        "latest_frame_age_ms",
        "grounding_mode",
        "current_answer_used_vision",
        "scene_social_state_v2",
    ):
        assert camera_key in actor_state["camera_scene"]

    scene_social = actor_state["camera_scene"]["scene_social_state_v2"]
    assert scene_social["schema_version"] == 2
    assert scene_social["camera_honesty_state"] in {
        "can_see_now",
        "recent_frame_available",
        "available_not_used",
        "unavailable",
    }
    assert scene_social["scene_transition"] in {
        "none",
        "camera_ready",
        "looking_requested",
        "frame_captured",
        "vision_answered",
        "vision_stale",
        "vision_unavailable",
    }

    semantic = actor_state["active_listening"]["semantic_state_v3"]
    assert semantic["schema_version"] == 3
    assert semantic["language"] == language.value
    assert semantic["detected_intent"] == "unknown"
    assert "safe_live_summary" in semantic
    assert "listener_chips" in semantic
    assert "still_listening" in [chip["chip_id"] for chip in semantic["listener_chips"]]


def test_browser_actor_surface_assets_use_public_actor_api_without_media_or_prompt_leaks():
    root = Path(__file__).resolve().parents[1]
    asset_paths = [
        root / "web/client_src/src/assets/blink-expression-panel.js",
    ]
    required_strings = (
        "/api/runtime/actor-state",
        "/api/runtime/actor-events",
        "/api/runtime/performance-state",
        "/api/runtime/performance-events",
        "actor_surface_v2_enabled",
        "renderActorSurface",
        "placePanel(panel)",
        "operatorPanel.appendChild(panel)",
        "profileBadge",
        "Debug timeline",
        "Heard",
        "I heard...",
        "constraint detected",
        "question detected",
        "showing object",
        "still listening",
        "ready to answer",
        "Blink is saying",
        "Looking",
        "honesty",
        "presence",
        "Used memory/persona",
        "Interruption",
        "听到",
        "我听到...",
        "检测到约束",
        "检测到问题",
        "正在看物体",
        "继续听",
        "可以回答",
        "Blink 正在说",
        "正在看",
        "使用的记忆/风格",
        "打断",
        "actorEventLimit = 50",
        "slice(-12)",
        "appendTranscriptText",
        "lastFinalTranscript: appendTranscriptText(",
        'lastFinalTranscript: ""',
    )
    forbidden_strings = (
        "getUserMedia",
        "createOffer",
        "raw_image",
        "raw_audio",
        "system_prompt",
        "developer_prompt",
        "hidden_prompt",
        "ice_candidate",
        "sdp_offer",
    )

    for asset_path in asset_paths:
        text = asset_path.read_text(encoding="utf-8")
        for required in required_strings:
            assert required in text
        for forbidden in forbidden_strings:
            assert forbidden not in text
