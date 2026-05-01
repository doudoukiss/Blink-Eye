"""Inspect and control Blink's narrow runtime shell from the local CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from blink.brain.context import BrainContextTask
from blink.brain.runtime_shell import BrainRuntimeShell
from blink.project_identity import PROJECT_IDENTITY, local_env_name
from blink.transcriptions.language import Language


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--brain-db-path", help="Optional SQLite store path override.")
    parser.add_argument("--runtime-kind", default="browser", help="Runtime kind used for default ids.")
    parser.add_argument("--client-id", help="Optional client id used when resolving default ids.")
    parser.add_argument("--user-id", help="Explicit user id override.")
    parser.add_argument("--thread-id", help="Explicit thread id override.")
    parser.add_argument(
        "--presence-scope-key",
        help="Optional presence scope key override. Defaults to <runtime-kind>:presence.",
    )
    parser.add_argument("--language", default="en", help="Language used for packet inspection.")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=f"Inspect and control {PROJECT_IDENTITY.display_name}'s runtime shell."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="Inspect the current runtime shell snapshot.")
    _add_common_args(snapshot)

    packet = subparsers.add_parser("packet", help="Inspect one compiled context packet.")
    _add_common_args(packet)
    packet.add_argument(
        "--task",
        required=True,
        choices=[task.value for task in BrainContextTask],
        help="Context task mode to compile.",
    )
    packet.add_argument("--query", required=True, help="Explicit query text for packet compilation.")

    waits = subparsers.add_parser("waits", help="Inspect waiting commitments and wake routing.")
    _add_common_args(waits)

    interrupt = subparsers.add_parser("interrupt", help="Interrupt one active commitment.")
    _add_common_args(interrupt)
    interrupt.add_argument("--commitment-id", required=True, help="Commitment id to interrupt.")
    interrupt.add_argument("--reason", required=True, help="Operator-facing reason summary.")

    suppress = subparsers.add_parser(
        "suppress",
        help="Move one commitment into explicit-resume-only holding state.",
    )
    _add_common_args(suppress)
    suppress.add_argument("--commitment-id", required=True, help="Commitment id to suppress.")
    suppress.add_argument("--reason", required=True, help="Operator-facing reason summary.")

    resume = subparsers.add_parser("resume", help="Resume one deferred or blocked commitment.")
    _add_common_args(resume)
    resume.add_argument("--commitment-id", required=True, help="Commitment id to resume.")
    resume.add_argument("--reason", help="Optional operator-facing reason summary.")

    return parser


def _open_shell(args: argparse.Namespace) -> BrainRuntimeShell:
    db_path = args.brain_db_path or os.getenv(local_env_name("BRAIN_DB_PATH"))
    return BrainRuntimeShell.open(
        brain_db_path=Path(db_path) if db_path else None,
        runtime_kind=args.runtime_kind,
        client_id=args.client_id,
        user_id=args.user_id,
        thread_id=args.thread_id,
        presence_scope_key=args.presence_scope_key or f"{args.runtime_kind}:presence",
        language=Language(str(args.language).lower()),
    )


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for the runtime shell."""
    args = build_parser().parse_args(argv)
    shell = _open_shell(args)
    try:
        if args.command == "snapshot":
            payload = shell.snapshot().as_dict()
        elif args.command == "packet":
            payload = shell.inspect_packet(
                task=BrainContextTask(str(args.task)),
                query_text=str(args.query),
            ).as_dict()
        elif args.command == "waits":
            payload = shell.inspect_pending_wakes().as_dict()
        elif args.command == "interrupt":
            payload = shell.interrupt_commitment(
                commitment_id=str(args.commitment_id),
                reason_summary=str(args.reason),
            ).as_dict()
        elif args.command == "suppress":
            payload = shell.suppress_commitment(
                commitment_id=str(args.commitment_id),
                reason_summary=str(args.reason),
            ).as_dict()
        elif args.command == "resume":
            payload = shell.resume_commitment(
                commitment_id=str(args.commitment_id),
                reason_summary=args.reason,
            ).as_dict()
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    finally:
        shell.close()

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
