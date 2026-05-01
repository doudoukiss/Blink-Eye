"""File-first episode dataset export orchestration for Phase 22."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blink.brain.evals.dataset_manifest import (
    BrainEpisodeDatasetManifest,
    build_episode_dataset_manifest,
)
from blink.brain.evals.episode_export import (
    BrainEpisodeRecord,
    build_episode_from_embodied_eval_run_payload,
    build_episode_from_replay_artifact_payload,
)
from blink.brain.evals.live_episode_export import build_episode_from_live_runtime_payload
from blink.brain.evals.practice_episode_export import build_episodes_from_practice_plan_payload
from blink.brain.store import BrainStore


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def _looks_like_eval_run(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("run_id"), str) and isinstance(payload.get("metrics"), dict)


def _looks_like_eval_report(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("scenario"), dict) and isinstance(payload.get("runs"), list)


def _looks_like_eval_suite(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("suite"), dict) and isinstance(payload.get("reports"), list)


def _run_source_path(run_payload: dict[str, Any], fallback: Path) -> Path:
    artifact_paths = dict(run_payload.get("artifact_paths", {}))
    run_json = _optional_text(artifact_paths.get("run_json"))
    return Path(run_json) if run_json is not None else fallback


def _artifact_uri(record: Any, *, artifact_kind: str) -> str | None:
    for artifact in getattr(record, "artifact_refs", ()):
        if getattr(artifact, "artifact_kind", None) == artifact_kind:
            return _optional_text(getattr(artifact, "uri", None))
    return None


def _record_source_export(
    *,
    store_path: Path,
    episode_path: Path,
    episode: BrainEpisodeRecord,
) -> None:
    if not store_path.exists():
        return
    conn = sqlite3.connect(store_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT agent_id, user_id, thread_id
            FROM brain_events
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return
    store = BrainStore(path=store_path)
    try:
        store.record_memory_export(
            user_id=str(row["user_id"]),
            thread_id=str(row["thread_id"]),
            export_kind="brain_episode_export",
            path=episode_path,
            payload=episode.as_dict(),
            metadata={
                "episode_id": episode.episode_id,
                "scenario_id": episode.scenario_id,
                "origin": episode.origin,
                "agent_id": str(row["agent_id"]),
            },
        )
    finally:
        store.close()


@dataclass(frozen=True)
class BrainEpisodeDatasetExportResult:
    """One bounded dataset-export result."""

    source: str
    input_path: str
    output_dir: str
    episode_paths: tuple[str, ...]
    manifest: BrainEpisodeDatasetManifest
    manifest_json_path: str
    manifest_markdown_path: str

    def as_dict(self) -> dict[str, Any]:
        """Serialize the dataset-export result."""
        return {
            "source": self.source,
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "episode_paths": list(self.episode_paths),
            "manifest": self.manifest.as_dict(),
            "manifest_json_path": self.manifest_json_path,
            "manifest_markdown_path": self.manifest_markdown_path,
        }


def load_episode_records_from_source(
    *, source: str, input_path: Path
) -> tuple[BrainEpisodeRecord, ...]:
    """Load canonical episode records from one supported source artifact."""
    payload = _read_json(input_path)
    if source == "embodied-eval":
        if _looks_like_eval_run(payload):
            return (build_episode_from_embodied_eval_run_payload(payload, source_path=input_path),)
        if _looks_like_eval_report(payload):
            return tuple(
                build_episode_from_embodied_eval_run_payload(
                    dict(run_payload),
                    source_path=_run_source_path(dict(run_payload), input_path),
                )
                for run_payload in payload.get("runs", [])
                if isinstance(run_payload, dict)
            )
        if _looks_like_eval_suite(payload):
            episodes: list[BrainEpisodeRecord] = []
            for report_payload in payload.get("reports", []):
                if not isinstance(report_payload, dict):
                    continue
                for run_payload in report_payload.get("runs", []):
                    if not isinstance(run_payload, dict):
                        continue
                    episodes.append(
                        build_episode_from_embodied_eval_run_payload(
                            dict(run_payload),
                            source_path=_run_source_path(dict(run_payload), input_path),
                        )
                    )
            return tuple(episodes)
        raise ValueError(f"Unsupported embodied-eval artifact payload: {input_path}")
    if source == "replay":
        return (build_episode_from_replay_artifact_payload(payload, source_path=input_path),)
    if source == "live":
        if isinstance(payload.get("episodes"), list):
            return tuple(
                sorted(
                    (
                        build_episode_from_live_runtime_payload(
                            dict(item),
                            source_path=input_path,
                        )
                        for item in payload.get("episodes", [])
                        if isinstance(item, dict)
                    ),
                    key=lambda record: record.episode_id,
                )
            )
        return (build_episode_from_live_runtime_payload(payload, source_path=input_path),)
    if source == "practice":
        if isinstance(payload.get("practice_plans"), list) or isinstance(
            payload.get("plans"), list
        ):
            plan_payloads = payload.get("practice_plans") or payload.get("plans") or []
            episodes: list[BrainEpisodeRecord] = []
            for item in plan_payloads:
                if not isinstance(item, dict):
                    continue
                episodes.extend(
                    build_episodes_from_practice_plan_payload(
                        dict(item),
                        source_path=input_path,
                    )
                )
            return tuple(sorted(episodes, key=lambda record: record.episode_id))
        return build_episodes_from_practice_plan_payload(payload, source_path=input_path)
    raise ValueError(f"Unsupported episode-export source '{source}'.")


def export_episode_dataset(
    *,
    source: str,
    input_path: Path,
    output_dir: Path,
) -> BrainEpisodeDatasetExportResult:
    """Export one deterministic episode dataset from one source artifact."""
    episodes = load_episode_records_from_source(source=source, input_path=input_path)
    manifest = build_episode_dataset_manifest(episodes)
    resolved_output_dir = output_dir
    episodes_dir = resolved_output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_paths: list[str] = []
    for episode in episodes:
        episode_path = episodes_dir / f"{episode.episode_id}.json"
        _write_json(episode_path, episode.as_dict())
        episode_paths.append(str(episode_path))
        source_store_path = _artifact_uri(episode, artifact_kind="brain_db")
        if source_store_path is not None:
            _record_source_export(
                store_path=Path(source_store_path),
                episode_path=episode_path,
                episode=episode,
            )
    manifest_json_path = resolved_output_dir / "dataset_manifest.json"
    manifest_markdown_path = resolved_output_dir / "dataset_manifest.md"
    _write_json(manifest_json_path, manifest.as_dict())
    manifest_markdown_path.write_text(manifest.render_markdown(), encoding="utf-8")
    return BrainEpisodeDatasetExportResult(
        source=source,
        input_path=str(input_path),
        output_dir=str(resolved_output_dir),
        episode_paths=tuple(sorted(episode_paths)),
        manifest=manifest,
        manifest_json_path=str(manifest_json_path),
        manifest_markdown_path=str(manifest_markdown_path),
    )


__all__ = [
    "BrainEpisodeDatasetExportResult",
    "export_episode_dataset",
    "load_episode_records_from_source",
]
