"""Generate continuity audit artifacts for the local Blink brain store."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from blink.brain.runtime_shell import BrainRuntimeShell
from blink.brain.session import resolve_brain_session_ids
from blink.project_identity import PROJECT_IDENTITY, local_env_name
from blink.transcriptions.language import Language


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=f"Write a continuity audit for {PROJECT_IDENTITY.display_name}."
    )
    parser.add_argument("--brain-db-path", help="Optional SQLite store path override.")
    parser.add_argument("--runtime-kind", default="browser", help="Runtime kind used for default ids.")
    parser.add_argument("--client-id", help="Optional client id used when resolving default ids.")
    parser.add_argument("--user-id", help="Explicit user id override.")
    parser.add_argument("--thread-id", help="Explicit thread id override.")
    parser.add_argument(
        "--presence-scope-key",
        help="Optional presence scope key override. Defaults to <runtime-kind>:presence.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory override. Defaults to artifacts/brain_audit.",
    )
    parser.add_argument(
        "--replay-cases-dir",
        help="Optional replay-fixture directory override. Defaults to tests/fixtures/brain_evals.",
    )
    parser.add_argument("--language", default="en", help="Language used for context selection traces.")
    parser.add_argument(
        "--reply-query",
        help="Optional explicit reply query used for context packet compilation.",
    )
    parser.add_argument(
        "--planning-query",
        help="Optional explicit planning query used for context packet compilation.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for continuity audits."""
    args = build_parser().parse_args(argv)
    session_ids = resolve_brain_session_ids(
        runtime_kind=args.runtime_kind,
        client_id=args.client_id,
    )
    resolved_session_ids = session_ids.__class__(
        agent_id=session_ids.agent_id,
        user_id=args.user_id or session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=args.thread_id or session_ids.thread_id,
    )
    db_path = args.brain_db_path or os.getenv(local_env_name("BRAIN_DB_PATH"))
    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/brain_audit")
    replay_cases_dir = (
        Path(args.replay_cases_dir) if args.replay_cases_dir else Path("tests/fixtures/brain_evals")
    )
    presence_scope_key = args.presence_scope_key or f"{args.runtime_kind}:presence"
    language = Language(str(args.language).lower())

    shell = BrainRuntimeShell.open(
        brain_db_path=Path(db_path) if db_path else None,
        runtime_kind=args.runtime_kind,
        client_id=args.client_id,
        user_id=resolved_session_ids.user_id,
        thread_id=resolved_session_ids.thread_id,
        presence_scope_key=presence_scope_key,
        language=language,
    )
    try:
        report = shell.export_audit(
            output_dir=output_dir,
            replay_cases_dir=replay_cases_dir,
            context_queries={
                key: value
                for key, value in {
                    "reply": args.reply_query,
                    "planning": args.planning_query,
                }.items()
                if value
            }
            or None,
        )
    finally:
        shell.close()

    print(f"audit_json={report.json_path}")
    print(f"audit_markdown={report.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
