import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from blink.cli import local_browser, local_runtime_profiles
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


def _clear_profile_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "BLINK_LOCAL_LANGUAGE",
        "BLINK_LOCAL_TTS_BACKEND",
        "BLINK_LOCAL_BROWSER_VISION",
        "BLINK_LOCAL_CONTINUOUS_PERCEPTION",
        "BLINK_LOCAL_ALLOW_BARGE_IN",
        "BLINK_LOCAL_LLM_SYSTEM_PROMPT",
        "OLLAMA_SYSTEM_PROMPT",
    ):
        monkeypatch.setenv(name, "")


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


def test_primary_browser_profiles_keep_bilingual_vision_parity() -> None:
    profiles = local_runtime_profiles.load_local_runtime_profiles(include_local_override=False)

    assert set(profiles) >= {"browser-zh-melo", "browser-en-kokoro"}
    assert profiles["browser-zh-melo"].runtime == "browser"
    assert profiles["browser-zh-melo"].get("language") == "zh"
    assert profiles["browser-zh-melo"].get("tts_backend") == "local-http-wav"
    assert profiles["browser-zh-melo"].get("browser_vision") is True
    assert profiles["browser-en-kokoro"].runtime == "browser"
    assert profiles["browser-en-kokoro"].get("language") == "en"
    assert profiles["browser-en-kokoro"].get("tts_backend") == "kokoro"
    assert profiles["browser-en-kokoro"].get("browser_vision") is True

    for profile_id in ("browser-zh-melo", "browser-en-kokoro"):
        profile = profiles[profile_id]
        assert profile.get("continuous_perception") is False
        assert profile.get("allow_barge_in") is False


@pytest.mark.parametrize(
    ("profile", "language", "tts_backend", "tts_label"),
    PRIMARY_BROWSER_PROFILES,
)
def test_primary_browser_profiles_resolve_to_equal_camera_defaults(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    language: local_browser.Language,
    tts_backend: str,
    tts_label: str,
) -> None:
    _clear_profile_override_env(monkeypatch)
    monkeypatch.setenv("BLINK_LOCAL_TTS_RUNTIME_LABEL", tts_label)

    config = local_browser.resolve_config(
        local_browser.build_parser().parse_args(["--config-profile", profile])
    )

    assert config.config_profile == profile
    assert config.language == language
    assert config.tts_backend == tts_backend
    assert config.vision_enabled is True
    assert config.continuous_perception_enabled is False
    assert config.allow_barge_in is False
    assert config.tts_runtime_label == tts_label


def test_browser_kokoro_profile_has_explicit_no_vision_escape_hatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_profile_override_env(monkeypatch)
    monkeypatch.setenv("BLINK_LOCAL_BROWSER_VISION", "0")

    env_config = local_browser.resolve_config(
        local_browser.build_parser().parse_args(["--config-profile", "browser-en-kokoro"])
    )
    cli_config = local_browser.resolve_config(
        local_browser.build_parser().parse_args(
            ["--config-profile", "browser-en-kokoro", "--no-vision"]
        )
    )

    assert env_config.vision_enabled is False
    assert cli_config.vision_enabled is False


def test_primary_browser_profiles_expose_same_runtime_state_schema() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("aiortc")
    pytest.importorskip("av")
    from fastapi.testclient import TestClient

    states = []
    stacks = []
    client_configs = []
    for profile, language, tts_backend, tts_label in PRIMARY_BROWSER_PROFILES:
        app, _ = create_app(
            _browser_config(
                profile=profile,
                language=language,
                tts_backend=tts_backend,
                tts_label=tts_label,
            )
        )
        client = TestClient(app)
        state = client.get("/api/runtime/performance-state").json()
        stack = client.get("/api/runtime/stack").json()
        script = client.get("/api/runtime/client-config.js").text
        payload = json.loads(
            script.removeprefix("globalThis.BlinkRuntimeConfig = Object.freeze(")
            .removesuffix(");\n")
        )
        states.append(state)
        stacks.append(stack)
        client_configs.append(payload)

    assert set(states[0]) == set(states[1])
    assert set(stacks[0]) == set(stacks[1])
    for state, stack, client_config in zip(states, stacks, client_configs, strict=True):
        assert state["runtime"] == "browser"
        assert state["transport"] == "WebRTC"
        assert state["vision"]["enabled"] is True
        assert state["vision"]["continuous_perception_enabled"] is False
        assert state["camera_presence"]["enabled"] is True
        assert state["camera_presence"]["state"] == "disconnected"
        assert state["protected_playback"] is True
        assert state["interruption"]["barge_in_state"] == "protected"
        assert stack["vision_enabled"] is True
        assert stack["continuous_perception_enabled"] is False
        assert client_config["enableCam"] is True
        assert client_config["enableMic"] is True


def test_primary_browser_launchers_advertise_parity_and_safe_logs() -> None:
    root = Path(__file__).resolve().parents[1]
    melo = (root / "scripts/run-local-browser-melo.sh").read_text(encoding="utf-8")
    kokoro = (root / "scripts/run-local-browser-kokoro-en.sh").read_text(encoding="utf-8")

    for text, profile, language, tts_label, latest_log in (
        (melo, "browser-zh-melo", "zh", "local-http-wav/MeloTTS", "latest-browser-melo.log"),
        (kokoro, "browser-en-kokoro", "en", "kokoro/English", "latest-browser-kokoro-en.log"),
    ):
        assert f"profile={profile}" in text
        assert f"language={language}" in text
        assert tts_label in text
        assert "webrtc=on" in text
        assert "camera_vision=$(launcher_state_label \"$DEFAULT_BROWSER_VISION\")" in text
        assert (
            "continuous_perception=$(launcher_state_label \"$DEFAULT_CONTINUOUS_PERCEPTION\")"
            in text
        )
        assert "protected_playback=$(launcher_protected_playback_label \"$DEFAULT_ALLOW_BARGE_IN\")" in text
        assert "barge_in_policy=$(launcher_barge_in_policy_label \"$DEFAULT_ALLOW_BARGE_IN\")" in text
        assert "--no-vision)" in text
        assert "DEFAULT_BROWSER_VISION=0" in text
        assert "--allow-barge-in)" in text
        assert "DEFAULT_ALLOW_BARGE_IN=1" in text
        assert "client=${CLIENT_URL}" in text
        assert "http://${CLIENT_HOST}:${CLIENT_PORT}/client/" in text
        assert latest_log in text
        assert "OPENAI_API_KEY" not in text
        assert "Authorization" not in text

    assert 'DEFAULT_BROWSER_VISION="${BLINK_LOCAL_BROWSER_VISION:-1}"' in kokoro
    assert "BLINK_LOCAL_BROWSER_VISION=0" not in kokoro
    assert "run-melotts" not in kokoro.lower()


def test_bilingual_browser_runtime_debug_script_is_provider_free() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("aiortc")
    pytest.importorskip("av")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["BLINK_IMPORT_BANNER"] = "0"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evals/debug-bilingual-browser-runtime.py",
            "--skip-port-check",
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["provider_free"] is True
    assert payload["opens_browser"] is False
    assert payload["calls_models"] is False
    assert payload["calls_tts"] is False
    assert payload["calls_moondream"] is False
    assert payload["accesses_raw_media"] is False
    assert [profile["profile"] for profile in payload["profiles"]] == [
        "browser-zh-melo",
        "browser-en-kokoro",
    ]
    for profile in payload["profiles"]:
        assert profile["passed"] is True
        assert all(status == 200 for status in profile["endpoint_statuses"].values())
        assert profile["checks"]["vision_default"] is True
        assert profile["checks"]["continuous_perception_default"] is True
        assert profile["checks"]["protected_playback_default"] is True
