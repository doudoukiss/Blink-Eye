#!/usr/bin/env python3
"""Prepare local Phase 11 dogfooding session plans without launching models."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PRIMARY_PROFILES = {
    "browser-zh-melo": {
        "language": "zh",
        "tts_runtime_label": "local-http-wav/MeloTTS",
        "launcher": "./scripts/run-local-browser-melo.sh",
    },
    "browser-en-kokoro": {
        "language": "en",
        "tts_runtime_label": "kokoro/English",
        "launcher": "./scripts/run-local-browser-kokoro-en.sh",
    },
}

SCENARIOS = (
    {
        "scenario_id": "long_planning",
        "task_label": "Long planning turn with constraints",
        "reason_codes": ["dogfood_task:planning", "dogfood_dimension:felt_heard"],
    },
    {
        "scenario_id": "visual_question",
        "task_label": "Ask a grounded visual question after showing an object",
        "reason_codes": ["dogfood_task:camera_honesty", "dogfood_task:visual_grounding"],
    },
    {
        "scenario_id": "interruption",
        "task_label": "Interrupt during assistant speech with an explicit correction",
        "reason_codes": ["dogfood_task:interruption", "dogfood_task:repair"],
    },
    {
        "scenario_id": "correction",
        "task_label": "Correct a previous preference and check stale-memory handling",
        "reason_codes": ["dogfood_task:memory_correction", "dogfood_task:conflict"],
    },
    {
        "scenario_id": "memory_callback",
        "task_label": "Ask why Blink responded that way and inspect public trace",
        "reason_codes": ["dogfood_task:memory_callback", "dogfood_task:plan_trace"],
    },
)


def _build_plan(profile: str, *, duration_seconds: int, output_dir: Path) -> dict[str, Any]:
    profile_data = PRIMARY_PROFILES[profile]
    preference_dir = output_dir / profile / "preferences"
    episode_dir = output_dir / profile / "episodes"
    command = [
        profile_data["launcher"],
        "--performance-episode-v3",
        "--performance-episode-v3-dir",
        str(episode_dir),
        "--performance-preferences-v3-dir",
        str(preference_dir),
    ]
    return {
        "schema_version": 3,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "language": profile_data["language"],
        "tts_runtime_label": profile_data["tts_runtime_label"],
        "duration_seconds": duration_seconds,
        "launcher": profile_data["launcher"],
        "command": command,
        "client_url": "http://127.0.0.1:7860/client/",
        "preference_dir": str(preference_dir),
        "episode_dir": str(episode_dir),
        "scenarios": list(SCENARIOS),
        "artifact_policy": {
            "stores_raw_audio": False,
            "stores_raw_images": False,
            "stores_hidden_prompts": False,
            "stores_freeform_notes": False,
            "reason_codes": [
                "performance_learning_dogfood:public_safe",
                "performance_learning_dogfood:local_only",
            ],
        },
        "reason_codes": [
            "performance_learning_dogfood:v3",
            "browser_primary_path",
            "native_pyaudio_isolation_lane",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(*PRIMARY_PROFILES.keys(), "both"),
        default="both",
        help="Primary browser profile to prepare.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=300,
        help="Dogfooding duration target per profile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/performance_preferences_v3/dogfood_runs"),
        help="Directory where the dogfood plan JSON is written.",
    )
    args = parser.parse_args()

    profiles = tuple(PRIMARY_PROFILES) if args.profile == "both" else (args.profile,)
    payload = {
        "schema_version": 3,
        "created_at": datetime.now(UTC).isoformat(),
        "profiles": [
            _build_plan(profile, duration_seconds=args.duration_seconds, output_dir=args.output_dir)
            for profile in profiles
        ],
        "reason_codes": [
            "performance_learning_dogfood_plan:v3",
            "dogfood_no_browser_launch",
            "dogfood_no_model_call",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "dogfood_plan.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), **payload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
