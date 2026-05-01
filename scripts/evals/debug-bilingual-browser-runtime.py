#!/usr/bin/env python3
"""Provider-free diagnostics for Blink's primary bilingual browser paths."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("BLINK_IMPORT_BANNER", "0")

from blink.cli import local_browser  # noqa: E402
from blink.cli.local_browser import LocalBrowserConfig, create_app  # noqa: E402

PRIMARY_BROWSER_PATHS = (
    {
        "profile": "browser-zh-melo",
        "language": local_browser.Language.ZH,
        "tts_backend": "local-http-wav",
        "tts_runtime_label": "local-http-wav/MeloTTS",
        "tts_base_url": "http://127.0.0.1:8001",
        "launcher": "./scripts/run-local-browser-melo.sh",
    },
    {
        "profile": "browser-en-kokoro",
        "language": local_browser.Language.EN,
        "tts_backend": "kokoro",
        "tts_runtime_label": "kokoro/English",
        "tts_base_url": None,
        "launcher": "./scripts/run-local-browser-kokoro-en.sh",
    },
)

RUNTIME_ENDPOINTS = (
    "/client/",
    "/api/runtime/client-config.js",
    "/api/runtime/actor-state",
    "/api/runtime/actor-events",
    "/api/runtime/performance-state",
    "/api/runtime/performance-events",
    "/api/runtime/client-media",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run provider-free diagnostics for the Chinese/Melo and English/Kokoro "
            "browser runtime paths."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--skip-port-check",
        action="store_true",
        help="Skip live port occupancy probing. Useful in hermetic tests.",
    )
    parser.add_argument(
        "--fail-on-port-occupied",
        action="store_true",
        help="Treat an occupied browser port as a diagnostic failure.",
    )
    return parser


def _browser_config(path: dict[str, Any], *, host: str, port: int) -> LocalBrowserConfig:
    return LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="provider-free diagnostic prompt",
        language=path["language"],
        stt_backend="mlx-whisper",
        tts_backend=path["tts_backend"],
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url=path["tts_base_url"],
        host=host,
        port=port,
        vision_enabled=True,
        continuous_perception_enabled=False,
        allow_barge_in=False,
        tts_runtime_label=path["tts_runtime_label"],
        config_profile=path["profile"],
    )


def _parse_client_config(script: str) -> dict[str, Any]:
    prefix = "globalThis.BlinkRuntimeConfig = Object.freeze("
    suffix = ");\n"
    if not script.startswith(prefix) or not script.endswith(suffix):
        return {}
    payload = script.removeprefix(prefix).removesuffix(suffix)
    parsed = json.loads(payload)
    return parsed if isinstance(parsed, dict) else {}


def _profile_diagnostics(path: dict[str, Any], *, host: str, port: int) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    config = _browser_config(path, host=host, port=port)
    app, _uvicorn = create_app(config)
    client = TestClient(app)
    endpoint_statuses: dict[str, int] = {}
    payloads: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for endpoint in RUNTIME_ENDPOINTS:
        response = client.get(endpoint)
        endpoint_statuses[endpoint] = response.status_code
        if response.status_code != 200:
            failures.append(f"endpoint_status:{endpoint}:{response.status_code}")
            continue
        if endpoint == "/api/runtime/client-config.js":
            payloads[endpoint] = _parse_client_config(response.text)
        elif endpoint != "/client/":
            try:
                parsed = response.json()
            except Exception:
                parsed = {}
            payloads[endpoint] = parsed if isinstance(parsed, dict) else {}

    client_config = payloads.get("/api/runtime/client-config.js", {})
    actor_state = payloads.get("/api/runtime/actor-state", {})
    performance_state = payloads.get("/api/runtime/performance-state", {})
    client_media = payloads.get("/api/runtime/client-media", {})

    actor_tts = actor_state.get("tts") if isinstance(actor_state.get("tts"), dict) else {}
    actor_vision = (
        actor_state.get("vision") if isinstance(actor_state.get("vision"), dict) else {}
    )
    checks = {
        "profile": actor_state.get("profile") == path["profile"],
        "language": actor_state.get("language") == path["language"].value,
        "tts_label": actor_tts.get("label") == path["tts_runtime_label"],
        "vision_default": actor_vision.get("enabled") is True
        and actor_vision.get("backend") == "moondream"
        and client_config.get("enableCam") is True,
        "continuous_perception_default": (
            actor_vision.get("continuous_perception_enabled") is False
        ),
        "protected_playback_default": actor_state.get("protected_playback") is True,
        "client_media_public_shape": {
            "camera_state",
            "microphone_state",
            "mode",
            "reason_codes",
        }.issubset(client_media),
        "performance_state_profile": performance_state.get("profile") == path["profile"],
    }
    failures.extend(f"check_failed:{name}" for name, passed in checks.items() if not passed)

    return {
        "profile": path["profile"],
        "launcher": path["launcher"],
        "language": path["language"].value,
        "tts_runtime_label": path["tts_runtime_label"],
        "provider_free": True,
        "endpoint_statuses": endpoint_statuses,
        "checks": checks,
        "client_media": {
            "mode": client_media.get("mode"),
            "camera_state": client_media.get("camera_state"),
            "microphone_state": client_media.get("microphone_state"),
            "reason_codes": client_media.get("reason_codes", []),
        },
        "failures": failures,
        "passed": not failures,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    failures: list[str] = []
    try:
        port_occupied = (
            False
            if args.skip_port_check
            else local_browser._browser_port_is_occupied(args.host, args.port)
        )
    except Exception:
        port_occupied = False
        failures.append("port_check:unavailable")
    port_check = {
        "host": args.host,
        "port": args.port,
        "checked": not args.skip_port_check,
        "occupied": bool(port_occupied),
        "blocking": bool(port_occupied and args.fail_on_port_occupied),
    }
    if port_check["blocking"]:
        failures.append("port_check:occupied")

    profile_results: list[dict[str, Any]] = []
    try:
        for path in PRIMARY_BROWSER_PATHS:
            result = _profile_diagnostics(path, host=args.host, port=args.port)
            profile_results.append(result)
            failures.extend(
                f"{result['profile']}:{failure}" for failure in result.get("failures", [])
            )
    except ImportError as exc:
        failures.append(f"dependency_missing:{type(exc).__name__}")
    except Exception as exc:
        failures.append(f"diagnostic_runtime_error:{type(exc).__name__}")

    payload = {
        "schema_version": 1,
        "diagnostic_id": "bilingual_browser_runtime_debug/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "provider_free": True,
        "opens_browser": False,
        "calls_models": False,
        "calls_tts": False,
        "calls_moondream": False,
        "accesses_raw_media": False,
        "port_check": port_check,
        "profiles": profile_results,
        "failures": failures,
        "passed": not failures,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
