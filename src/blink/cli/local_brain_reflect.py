"""Run one deterministic Blink brain reflection cycle against the local SQLite store."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from blink.brain.runtime_shell import BrainRuntimeShell
from blink.brain.session import resolve_brain_session_ids
from blink.project_identity import PROJECT_IDENTITY, local_env_name


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=f"Run one structured reflection cycle for {PROJECT_IDENTITY.display_name}."
    )
    parser.add_argument("--brain-db-path", help="Optional SQLite store path override.")
    parser.add_argument("--runtime-kind", default="browser", help="Runtime kind used for default ids.")
    parser.add_argument("--client-id", help="Optional client id used when resolving default ids.")
    parser.add_argument("--user-id", help="Explicit user id override.")
    parser.add_argument("--thread-id", help="Explicit thread id override.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for bounded reflection maintenance."""
    args = build_parser().parse_args(argv)
    session_ids = resolve_brain_session_ids(
        runtime_kind=args.runtime_kind,
        client_id=args.client_id,
    )
    user_id = args.user_id or session_ids.user_id
    thread_id = args.thread_id or session_ids.thread_id
    db_path = args.brain_db_path or os.getenv(local_env_name("BRAIN_DB_PATH"))

    shell = BrainRuntimeShell.open(
        brain_db_path=Path(db_path) if db_path else None,
        runtime_kind=args.runtime_kind,
        client_id=args.client_id,
        user_id=user_id,
        thread_id=thread_id,
    )
    try:
        result = shell.run_reflection_once(trigger="manual")
    finally:
        shell.close()

    print(f"cycle_id={result.cycle_id}")
    print(f"draft_artifact={result.draft_artifact_path}")
    print(f"health_report_id={result.health_report_id or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
