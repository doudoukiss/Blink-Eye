#!/usr/bin/env python3
"""Compare local Phase 11 performance preference JSONL artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blink.brain.evals.performance_preferences import (
    PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
    render_performance_preference_comparison,
    render_performance_preference_comparison_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preferences-dir",
        type=Path,
        default=PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
        help="Directory containing preferences.jsonl and policy_proposals.jsonl.",
    )
    parser.add_argument("--output-json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--output-md", type=Path, help="Optional Markdown output path.")
    args = parser.parse_args()

    payload = render_performance_preference_comparison(preferences_dir=args.preferences_dir)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(
            render_performance_preference_comparison_markdown(payload),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
