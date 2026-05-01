"""Operator-facing robot-head proof shows for Blink."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from blink.embodiment.robot_head.controller import RobotHeadController

ShowStepKind = Literal["motif", "state", "neutral"]


@dataclass(frozen=True)
class RobotHeadShowStep:
    """One operator-facing step inside a robot-head proof show."""

    kind: ShowStepKind
    target: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class RobotHeadShowDefinition:
    """Blink-owned proof-show definition."""

    name: str
    title: str
    description: str
    steps: list[RobotHeadShowStep] = field(default_factory=list)
    sensitive_motion: bool = False
    expected_live: bool = True


def build_robot_head_show_catalog() -> dict[str, RobotHeadShowDefinition]:
    """Build the Blink-owned robot-head proof-show catalog."""
    return {
        "investor_head_motion_v3": RobotHeadShowDefinition(
            name="investor_head_motion_v3",
            title="V3 Head Motion",
            description="Focused head-yaw evidence ladder with max rotations and strong sweeps.",
            steps=[
                RobotHeadShowStep(kind="motif", target="investor_head_turn_left_v3"),
                RobotHeadShowStep(kind="motif", target="investor_head_turn_right_v3"),
                RobotHeadShowStep(kind="motif", target="investor_head_sweep_left_v3"),
                RobotHeadShowStep(kind="motif", target="investor_head_sweep_right_v3"),
            ],
        ),
        "investor_eye_motion_v4": RobotHeadShowDefinition(
            name="investor_eye_motion_v4",
            title="V4 Eye Motion",
            description="Focused eye yaw and pitch proof.",
            steps=[
                RobotHeadShowStep(kind="motif", target="investor_eye_yaw_left_v4"),
                RobotHeadShowStep(kind="motif", target="investor_eye_yaw_right_v4"),
                RobotHeadShowStep(kind="motif", target="investor_eye_pitch_up_v4"),
                RobotHeadShowStep(kind="motif", target="investor_eye_pitch_down_v4"),
            ],
        ),
        "investor_lid_motion_v5": RobotHeadShowDefinition(
            name="investor_lid_motion_v5",
            title="V5 Lid Motion",
            description="Focused lids and blink proof.",
            steps=[
                RobotHeadShowStep(kind="motif", target="investor_both_lids_v5"),
                RobotHeadShowStep(kind="motif", target="investor_left_eye_lids_v5"),
                RobotHeadShowStep(kind="motif", target="investor_right_eye_lids_v5"),
                RobotHeadShowStep(kind="motif", target="investor_blink_v5"),
            ],
        ),
        "investor_brow_motion_v6": RobotHeadShowDefinition(
            name="investor_brow_motion_v6",
            title="V6 Brow Motion",
            description="Focused brow proof.",
            steps=[
                RobotHeadShowStep(kind="motif", target="investor_brows_both_v6"),
                RobotHeadShowStep(kind="motif", target="investor_brow_left_v6"),
                RobotHeadShowStep(kind="motif", target="investor_brow_right_v6"),
            ],
        ),
        "investor_neck_motion_v7": RobotHeadShowDefinition(
            name="investor_neck_motion_v7",
            title="V7 Neck Motion",
            description="Protective neck proof for tilt and pitch.",
            sensitive_motion=True,
            steps=[
                RobotHeadShowStep(kind="motif", target="investor_neck_tilt_left_v7"),
                RobotHeadShowStep(kind="motif", target="investor_neck_tilt_right_v7"),
                RobotHeadShowStep(kind="motif", target="investor_neck_pitch_up_v7"),
                RobotHeadShowStep(kind="motif", target="investor_neck_pitch_down_v7"),
            ],
        ),
        "investor_expressive_motion_v8": RobotHeadShowDefinition(
            name="investor_expressive_motion_v8",
            title="V8 Expressive Motion",
            description="Twelve motif expressive proof lane inspired by the maintained V8 hardware run.",
            sensitive_motion=True,
            steps=[
                RobotHeadShowStep(kind="motif", target="attentive_notice_right"),
                RobotHeadShowStep(kind="motif", target="attentive_notice_left"),
                RobotHeadShowStep(kind="motif", target="guarded_close_right"),
                RobotHeadShowStep(kind="motif", target="guarded_close_left"),
                RobotHeadShowStep(kind="motif", target="curious_lift"),
                RobotHeadShowStep(kind="motif", target="reflective_lower"),
                RobotHeadShowStep(kind="motif", target="skeptical_tilt_right"),
                RobotHeadShowStep(kind="motif", target="empathetic_tilt_left"),
                RobotHeadShowStep(kind="motif", target="playful_peek_right"),
                RobotHeadShowStep(kind="motif", target="playful_peek_left"),
                RobotHeadShowStep(kind="motif", target="bright_reengage"),
                RobotHeadShowStep(kind="motif", target="doubtful_side_glance"),
            ],
        ),
    }


SHOW_ALIASES = {
    "v3": "investor_head_motion_v3",
    "v4": "investor_eye_motion_v4",
    "v5": "investor_lid_motion_v5",
    "v6": "investor_brow_motion_v6",
    "v7": "investor_neck_motion_v7",
    "v8": "investor_expressive_motion_v8",
}


def resolve_robot_head_show(name: str) -> RobotHeadShowDefinition:
    """Resolve one operator-facing proof show."""
    normalized = SHOW_ALIASES.get(name.strip().lower(), name.strip())
    catalog = build_robot_head_show_catalog()
    if normalized not in catalog:
        raise ValueError(f"Unsupported robot-head show: {name}")
    return catalog[normalized]


def list_robot_head_shows() -> list[RobotHeadShowDefinition]:
    """Return all known robot-head proof shows."""
    return list(build_robot_head_show_catalog().values())


class RobotHeadShowRunner:
    """Run a Blink-owned robot-head proof show through the controller."""

    def __init__(
        self,
        *,
        controller: RobotHeadController,
        report_dir: Path | None = None,
        allow_sensitive_motion: bool = False,
    ):
        """Initialize the show runner."""
        self._controller = controller
        self._report_dir = report_dir or (Path.cwd() / "artifacts" / "robot_head_shows")
        self._report_dir.mkdir(parents=True, exist_ok=True)
        self._allow_sensitive_motion = allow_sensitive_motion

    async def run(self, show_name: str) -> dict:
        """Run one proof show and return the report payload."""
        definition = resolve_robot_head_show(show_name)
        if definition.sensitive_motion and not self._allow_sensitive_motion:
            raise ValueError(
                f"Show '{definition.name}' requires --allow-sensitive-motion because it uses neck motion."
            )

        started_at = datetime.now(UTC)
        report = {
            "show": asdict(definition),
            "started_at": started_at.isoformat(),
            "driver_mode": self._controller.driver_mode,
            "status_before": (await self._controller.status()).model_dump(),
            "steps": [],
        }

        try:
            report["neutral_start"] = (
                await self._controller.return_neutral(
                    source="operator",
                    reason=f"Prepare show '{definition.name}' from neutral.",
                )
            ).model_dump()

            for index, step in enumerate(definition.steps, start=1):
                if step.kind == "neutral":
                    result = await self._controller.return_neutral(
                        source="operator",
                        reason=f"Show '{definition.name}' neutral step {index}.",
                    )
                elif step.kind == "state":
                    if step.target is None:
                        raise ValueError(f"State step {index} requires a target.")
                    result = await self._controller.set_state(
                        step.target,
                        source="operator",
                        reason=f"Show '{definition.name}' state step {index}.",
                    )
                else:
                    if step.target is None:
                        raise ValueError(f"Motif step {index} requires a target.")
                    result = await self._controller.run_motif(
                        step.target,
                        source="operator",
                        reason=f"Show '{definition.name}' motif step {index}.",
                    )

                step_record = {
                    "index": index,
                    "label": step.label or step.target or step.kind,
                    "kind": step.kind,
                    "target": step.target,
                    "result": result.model_dump(),
                }
                report["steps"].append(step_record)
                if not result.accepted:
                    break

            report["neutral_end"] = (
                await self._controller.return_neutral(
                    source="operator",
                    reason=f"Finish show '{definition.name}' at neutral.",
                )
            ).model_dump()
            report["status_after"] = (await self._controller.status()).model_dump()
        finally:
            completed_at = datetime.now(UTC)
            report["completed_at"] = completed_at.isoformat()
            report_path = self._report_dir / (
                f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{definition.name}.json"
            )
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            report["report_path"] = str(report_path)

        return report
