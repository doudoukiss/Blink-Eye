"""CLI for preview-first curated Blink memory/personality ingestion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from blink.brain.memory_persona_ingestion import (
    BrainMemoryPersonaIngestionReport,
    apply_memory_persona_ingestion,
    build_memory_persona_ingestion_preview,
    rejected_seed_load_report,
)
from blink.brain.persona import build_witty_sophisticated_memory_story_seed
from blink.brain.session import BrainSessionIds, resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.project_identity import PROJECT_IDENTITY, local_env_name


def build_parser() -> argparse.ArgumentParser:
    """Build the memory/personality ingestion CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            f"Preview or apply curated local memory/personality seeds for "
            f"{PROJECT_IDENTITY.display_name}."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate and preview without writes.")
    mode.add_argument("--apply", action="store_true", help="Persist after matching preview approval.")
    seed_source = parser.add_mutually_exclusive_group()
    seed_source.add_argument("--seed", help="Path to the JSON seed file.")
    seed_source.add_argument(
        "--preset",
        choices=("witty-sophisticated", "witty_sophisticated"),
        help="Use a built-in curated seed without reading a seed file.",
    )
    parser.add_argument("--report", help="Optional path to write the JSON report.")
    parser.add_argument(
        "--approved-report",
        help="Required for --apply; must match the seed hash and import id.",
    )
    parser.add_argument("--brain-db-path", help="Optional SQLite store path override.")
    parser.add_argument("--runtime-kind", default="browser", help="Runtime kind for default ids.")
    parser.add_argument("--client-id", help="Optional client id used when resolving default ids.")
    parser.add_argument("--agent-id", default="blink/main", help="Agent id for the import scope.")
    parser.add_argument("--user-id", help="Explicit user id override.")
    parser.add_argument("--session-id", help="Explicit session id override.")
    parser.add_argument("--thread-id", help="Explicit thread id override.")
    return parser


def _session_ids(args: argparse.Namespace) -> BrainSessionIds:
    resolved = resolve_brain_session_ids(
        runtime_kind=str(args.runtime_kind),
        client_id=args.client_id,
        agent_id=str(args.agent_id or "blink/main"),
    )
    return BrainSessionIds(
        agent_id=str(args.agent_id or resolved.agent_id),
        user_id=str(args.user_id or resolved.user_id),
        session_id=str(args.session_id or resolved.session_id),
        thread_id=str(args.thread_id or resolved.thread_id),
    )


def _read_json_object(path: str, *, kind: str) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, (f"{kind}_file_missing",)
    except json.JSONDecodeError:
        return None, (f"{kind}_json_invalid",)
    except OSError:
        return None, (f"{kind}_read_failed",)
    if not isinstance(payload, dict):
        return None, (f"{kind}_payload_must_be_object",)
    return payload, ()


def _resolve_seed_payload(
    args: argparse.Namespace,
    *,
    session_ids: BrainSessionIds,
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    if args.seed:
        return _read_json_object(args.seed, kind="seed")
    if args.preset in {"witty-sophisticated", "witty_sophisticated"}:
        return (
            build_witty_sophisticated_memory_story_seed(
                user_name=str(args.user_id or args.client_id or session_ids.user_id),
                agent_id=session_ids.agent_id,
            ),
            (),
        )
    return None, ("seed_or_preset_required",)


def _emit_report(report: BrainMemoryPersonaIngestionReport, path: str | None) -> None:
    payload = json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    if path:
        Path(path).write_text(f"{payload}\n", encoding="utf-8")
    else:
        print(payload)


def _open_store(args: argparse.Namespace) -> BrainStore:
    db_path = args.brain_db_path or os.getenv(local_env_name("BRAIN_DB_PATH"))
    return BrainStore(path=Path(db_path) if db_path else None)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for curated memory/personality ingestion."""
    args = build_parser().parse_args(argv)
    session_ids = _session_ids(args)
    seed_payload, seed_errors = _resolve_seed_payload(args, session_ids=session_ids)
    if seed_payload is None:
        report = rejected_seed_load_report(*seed_errors)
        _emit_report(report, args.report)
        return 1

    if args.dry_run:
        report = build_memory_persona_ingestion_preview(seed_payload, session_ids=session_ids)
        _emit_report(report, args.report)
        return 0 if report.accepted else 1

    if not args.approved_report:
        report = rejected_seed_load_report("approved_report_required")
        _emit_report(report, args.report)
        return 1

    approved_payload, approved_errors = _read_json_object(args.approved_report, kind="approved_report")
    if approved_payload is None:
        report = rejected_seed_load_report(*approved_errors)
        _emit_report(report, args.report)
        return 1

    store = _open_store(args)
    try:
        report = apply_memory_persona_ingestion(
            store=store,
            seed=seed_payload,
            session_ids=session_ids,
            approved_report=approved_payload,
        )
    finally:
        store.close()
    _emit_report(report, args.report)
    return 0 if report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
