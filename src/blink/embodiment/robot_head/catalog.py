"""Blink-owned semantic capability catalog for the robot head."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from blink.embodiment.robot_head.models import (
    RobotHeadMotif,
    RobotHeadMotifStep,
    RobotHeadPersistentState,
    RobotHeadUnitCategory,
    RobotHeadUnitProfile,
)


class RobotHeadCapabilityCatalog(BaseModel):
    """Planner-facing semantic catalog for robot-head embodiment."""

    model_config = ConfigDict(frozen=True)

    version: str
    description: str
    source_notes: list[str] = Field(default_factory=list)
    units: dict[str, RobotHeadUnitProfile]
    persistent_states: dict[str, RobotHeadPersistentState]
    motifs: dict[str, RobotHeadMotif]
    neutral_state_name: str = "neutral"
    safe_idle_state_name: str = "safe_idle"

    def public_state_names(self) -> list[str]:
        """Return user-facing persistent state names."""
        return sorted(
            state.name for state in self.persistent_states.values() if state.public
        )

    def public_motif_names(self) -> list[str]:
        """Return user-facing motif names."""
        return sorted(motif.name for motif in self.motifs.values() if motif.public)

    def get_state(self, name: str) -> RobotHeadPersistentState:
        """Return a persistent state by name."""
        if name not in self.persistent_states:
            raise ValueError(f"Unsupported robot-head state: {name}")
        return self.persistent_states[name]

    def get_motif(self, name: str) -> RobotHeadMotif:
        """Return a motif by name."""
        if name not in self.motifs:
            raise ValueError(f"Unsupported robot-head motif: {name}")
        return self.motifs[name]

    def validate_values(self, values: dict[str, float]) -> tuple[dict[str, float], list[str], bool]:
        """Clamp semantic values against the safe catalog ranges."""
        validated: dict[str, float] = {}
        warnings: list[str] = []
        preview_only = False

        for unit_name, raw_value in values.items():
            if unit_name not in self.units:
                raise ValueError(f"Unsupported robot-head unit: {unit_name}")

            unit = self.units[unit_name]
            clamped = max(unit.normalized_min, min(unit.normalized_max, float(raw_value)))
            if clamped != float(raw_value):
                warnings.append(
                    f"Clamped {unit_name} from {raw_value:.2f} to {clamped:.2f} within safe range."
                )
            validated[unit_name] = round(clamped, 3)
            preview_only = preview_only or unit.preview_only

        return validated, warnings, preview_only


def build_default_robot_head_catalog() -> RobotHeadCapabilityCatalog:
    """Build Blink's default semantic robot-head catalog."""
    return RobotHeadCapabilityCatalog(
        version="blink-robot-head-v1",
        description=(
            "Semantic capability catalog derived from the tracked robot_head live-limit, "
            "revalidation, and hardware handoff notes. Planner-facing values stay normalized "
            "and semantic; any future live calibration overlay will map them onto hardware."
        ),
        source_notes=[
            "Derived from the robot_head live-limit and revalidation notes in the working tree.",
            "Persistent states are eye-area-only held configurations.",
            "Structural motion belongs in motifs, not held states.",
            "Neck pitch and tilt are reserved for operator proof lanes and remain out of the "
            "normal public conversational surface.",
            "Focused proof lanes V3-V7 follow the slower 100/32 family-demo lane recovered "
            "from maintained hardware notes.",
            "Expressive proof lane V8 follows the slower 90/24 staged structural-then-"
            "expressive lane and is intentionally larger for readability.",
        ],
        units={
            "head_turn": RobotHeadUnitProfile(
                name="head_turn",
                category=RobotHeadUnitCategory.STRUCTURAL,
                normalized_min=-1.0,
                normalized_max=1.0,
                notes=(
                    "Head yaw uses the full verified family envelope for proof lanes; the public "
                    "conversation surface stays materially lower."
                ),
            ),
            "neck_pitch": RobotHeadUnitProfile(
                name="neck_pitch",
                category=RobotHeadUnitCategory.STRUCTURAL,
                normalized_min=-0.55,
                normalized_max=0.55,
                notes=(
                    "Coupled neck pair is reserved for operator proof lanes and expressive motifs. "
                    "The catalog caps it to the V7/V8 protective band instead of the full raw pair "
                    "range."
                ),
            ),
            "neck_tilt": RobotHeadUnitProfile(
                name="neck_tilt",
                category=RobotHeadUnitCategory.STRUCTURAL,
                normalized_min=-0.9,
                normalized_max=0.9,
                notes=(
                    "Isolated tilt compiles against the family-specific live tilt envelope rather "
                    "than the full neck-pair raw bounds."
                ),
            ),
            "eye_yaw": RobotHeadUnitProfile(
                name="eye_yaw",
                category=RobotHeadUnitCategory.EXPRESSIVE,
                normalized_min=-1.0,
                normalized_max=1.0,
            ),
            "eye_pitch": RobotHeadUnitProfile(
                name="eye_pitch",
                category=RobotHeadUnitCategory.EXPRESSIVE,
                normalized_min=-0.95,
                normalized_max=0.95,
            ),
            "left_lids": RobotHeadUnitProfile(
                name="left_lids",
                category=RobotHeadUnitCategory.EXPRESSIVE,
                normalized_min=-0.85,
                normalized_max=0.45,
            ),
            "right_lids": RobotHeadUnitProfile(
                name="right_lids",
                category=RobotHeadUnitCategory.EXPRESSIVE,
                normalized_min=-0.85,
                normalized_max=0.45,
            ),
            "left_brow": RobotHeadUnitProfile(
                name="left_brow",
                category=RobotHeadUnitCategory.EXPRESSIVE,
                normalized_min=-0.45,
                normalized_max=0.58,
            ),
            "right_brow": RobotHeadUnitProfile(
                name="right_brow",
                category=RobotHeadUnitCategory.EXPRESSIVE,
                normalized_min=-0.45,
                normalized_max=0.58,
            ),
        },
        persistent_states={
            "neutral": RobotHeadPersistentState(
                name="neutral",
                description="Return the head to the neutral expression baseline.",
            ),
            "friendly": RobotHeadPersistentState(
                name="friendly",
                description="A clearly readable friendly eye-area expression.",
                preset="conversation_readable",
                values={
                    "left_lids": 0.24,
                    "right_lids": 0.24,
                    "left_brow": 0.24,
                    "right_brow": 0.24,
                    "eye_pitch": 0.04,
                },
            ),
            "listen_attentively": RobotHeadPersistentState(
                name="listen_attentively",
                description="A visibly attentive listening expression while the user speaks.",
                preset="conversation_readable",
                values={
                    "left_lids": 0.28,
                    "right_lids": 0.28,
                    "left_brow": 0.26,
                    "right_brow": 0.26,
                    "eye_pitch": 0.12,
                },
            ),
            "thinking": RobotHeadPersistentState(
                name="thinking",
                description="A more legible reflective expression between user turn and answer.",
                preset="conversation_readable",
                values={
                    "left_lids": 0.06,
                    "right_lids": 0.06,
                    "left_brow": 0.06,
                    "right_brow": 0.2,
                    "eye_pitch": -0.14,
                },
            ),
            "focused_soft": RobotHeadPersistentState(
                name="focused_soft",
                description="A gentle attentive expression without structural motion.",
                preset="conversation_readable",
                values={
                    "left_lids": 0.16,
                    "right_lids": 0.16,
                    "left_brow": 0.12,
                    "right_brow": 0.12,
                    "eye_pitch": 0.04,
                },
            ),
            "confused": RobotHeadPersistentState(
                name="confused",
                description="A visibly puzzled eye-area expression.",
                preset="conversation_readable",
                values={
                    "left_lids": 0.08,
                    "right_lids": 0.04,
                    "left_brow": 0.24,
                    "right_brow": -0.18,
                    "eye_pitch": -0.12,
                },
            ),
            "safe_idle": RobotHeadPersistentState(
                name="safe_idle",
                description="A stable idle expression used before returning to neutral.",
                values={
                    "left_lids": 0.1,
                    "right_lids": 0.1,
                },
            ),
        },
        motifs={
            "acknowledge": RobotHeadMotif(
                name="acknowledge",
                description="A readable acknowledge motion when the bot begins speaking.",
                preset="conversation_readable",
                steps=[
                    RobotHeadMotifStep(
                        label="engage",
                        values={
                            "head_turn": 0.24,
                            "eye_yaw": 0.18,
                            "left_lids": 0.16,
                            "right_lids": 0.16,
                            "left_brow": 0.18,
                            "right_brow": 0.18,
                        },
                        hold_ms=320,
                    ),
                    RobotHeadMotifStep(
                        label="release",
                        values={
                            "head_turn": 0.1,
                            "eye_yaw": 0.08,
                            "left_lids": 0.14,
                            "right_lids": 0.14,
                        },
                        hold_ms=180,
                    ),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "blink": RobotHeadMotif(
                name="blink",
                description="Close and reopen both lids once.",
                preset="conversation_readable",
                steps=[
                    RobotHeadMotifStep(
                        label="close",
                        values={"left_lids": -0.72, "right_lids": -0.72},
                        hold_ms=140,
                    ),
                    RobotHeadMotifStep(
                        label="open",
                        values={"left_lids": 0.14, "right_lids": 0.14},
                        hold_ms=0,
                    ),
                ],
            ),
            "wink_left": RobotHeadMotif(
                name="wink_left",
                description="Close and reopen the left lids once.",
                preset="conversation_readable",
                steps=[
                    RobotHeadMotifStep(label="close", values={"left_lids": -0.72}, hold_ms=160),
                    RobotHeadMotifStep(label="open", values={"left_lids": 0.14}, hold_ms=0),
                ],
            ),
            "wink_right": RobotHeadMotif(
                name="wink_right",
                description="Close and reopen the right lids once.",
                preset="conversation_readable",
                steps=[
                    RobotHeadMotifStep(label="close", values={"right_lids": -0.72}, hold_ms=160),
                    RobotHeadMotifStep(label="open", values={"right_lids": 0.14}, hold_ms=0),
                ],
            ),
            "look_left": RobotHeadMotif(
                name="look_left",
                description="A strong leftward gaze cue with clearly visible head follow.",
                preset="conversation_readable",
                steps=[
                    RobotHeadMotifStep(
                        label="turn",
                        values={"head_turn": -0.6, "eye_yaw": -0.68},
                        hold_ms=460,
                    ),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "look_right": RobotHeadMotif(
                name="look_right",
                description="A strong rightward gaze cue with clearly visible head follow.",
                preset="conversation_readable",
                steps=[
                    RobotHeadMotifStep(
                        label="turn",
                        values={"head_turn": 0.6, "eye_yaw": 0.68},
                        hold_ms=460,
                    ),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "listen_engage": RobotHeadMotif(
                name="listen_engage",
                description="Policy-only listening engage pulse that reads clearly on hardware.",
                public=False,
                preset="conversation_readable",
                steps=[
                    RobotHeadMotifStep(
                        label="engage",
                        values={
                            "left_lids": 0.34,
                            "right_lids": 0.34,
                            "left_brow": 0.3,
                            "right_brow": 0.3,
                            "eye_pitch": 0.16,
                        },
                        hold_ms=240,
                    ),
                    RobotHeadMotifStep(
                        label="settle",
                        values={
                            "left_lids": 0.28,
                            "right_lids": 0.28,
                            "left_brow": 0.26,
                            "right_brow": 0.26,
                            "eye_pitch": 0.12,
                        },
                        hold_ms=80,
                    ),
                ],
            ),
            "thinking_shift": RobotHeadMotif(
                name="thinking_shift",
                description="Policy-only reflective shift used at the end of the user turn.",
                public=False,
                preset="conversation_readable",
                steps=[
                    RobotHeadMotifStep(
                        label="glance_down",
                        values={
                            "left_lids": 0.08,
                            "right_lids": 0.08,
                            "left_brow": 0.04,
                            "right_brow": 0.24,
                            "eye_pitch": -0.24,
                        },
                        hold_ms=260,
                    ),
                    RobotHeadMotifStep(
                        label="settle",
                        values={
                            "left_lids": 0.06,
                            "right_lids": 0.06,
                            "left_brow": 0.06,
                            "right_brow": 0.2,
                            "eye_pitch": -0.14,
                        },
                        hold_ms=80,
                    ),
                ],
            ),
            "curious_tilt": RobotHeadMotif(
                name="curious_tilt",
                description="Preview-only curious tilt motif for future hardware validation.",
                public=False,
                preview_only=True,
                preset="preview_safe",
                steps=[
                    RobotHeadMotifStep(
                        label="tilt",
                        values={"neck_tilt": 0.48, "left_brow": 0.18, "right_brow": 0.18},
                        hold_ms=320,
                    ),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_head_turn_left_v3": RobotHeadMotif(
                name="investor_head_turn_left_v3",
                description="Operator-only V3 head-yaw proof to the left.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(label="turn", values={"head_turn": -0.82}, hold_ms=720),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_head_turn_right_v3": RobotHeadMotif(
                name="investor_head_turn_right_v3",
                description="Operator-only V3 head-yaw proof to the right.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(label="turn", values={"head_turn": 0.82}, hold_ms=720),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_head_sweep_left_v3": RobotHeadMotif(
                name="investor_head_sweep_left_v3",
                description="Operator-only V3 strong head-yaw sweep to the left.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(label="prep_right", values={"head_turn": 0.42}, hold_ms=260),
                    RobotHeadMotifStep(label="sweep_left", values={"head_turn": -0.82}, hold_ms=960),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_head_sweep_right_v3": RobotHeadMotif(
                name="investor_head_sweep_right_v3",
                description="Operator-only V3 strong head-yaw sweep to the right.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(label="prep_left", values={"head_turn": -0.42}, hold_ms=260),
                    RobotHeadMotifStep(label="sweep_right", values={"head_turn": 0.86}, hold_ms=960),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_eye_yaw_left_v4": RobotHeadMotif(
                name="investor_eye_yaw_left_v4",
                description="Operator-only V4 eye-yaw proof to the left.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(label="turn", values={"eye_yaw": -0.82}, hold_ms=640),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_eye_yaw_right_v4": RobotHeadMotif(
                name="investor_eye_yaw_right_v4",
                description="Operator-only V4 eye-yaw proof to the right.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(label="turn", values={"eye_yaw": 0.82}, hold_ms=640),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_eye_pitch_up_v4": RobotHeadMotif(
                name="investor_eye_pitch_up_v4",
                description="Operator-only V4 eye-pitch proof upward.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(
                        label="lift",
                        values={"eye_pitch": 0.62, "left_lids": 0.14, "right_lids": 0.14},
                        hold_ms=620,
                    ),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_eye_pitch_down_v4": RobotHeadMotif(
                name="investor_eye_pitch_down_v4",
                description="Operator-only V4 eye-pitch proof downward.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(label="lower", values={"eye_pitch": -0.58}, hold_ms=620),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_both_lids_v5": RobotHeadMotif(
                name="investor_both_lids_v5",
                description="Operator-only V5 both-lids proof.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(
                        label="close",
                        values={"left_lids": -0.78, "right_lids": -0.78},
                        hold_ms=520,
                    ),
                    RobotHeadMotifStep(
                        label="open",
                        values={"left_lids": 0.18, "right_lids": 0.18},
                        hold_ms=0,
                    ),
                ],
            ),
            "investor_left_eye_lids_v5": RobotHeadMotif(
                name="investor_left_eye_lids_v5",
                description="Operator-only V5 left-eye lid proof.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(label="close", values={"left_lids": -0.78}, hold_ms=520),
                    RobotHeadMotifStep(label="open", values={"left_lids": 0.18}, hold_ms=0),
                ],
            ),
            "investor_right_eye_lids_v5": RobotHeadMotif(
                name="investor_right_eye_lids_v5",
                description="Operator-only V5 right-eye lid proof.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(label="close", values={"right_lids": -0.78}, hold_ms=520),
                    RobotHeadMotifStep(label="open", values={"right_lids": 0.18}, hold_ms=0),
                ],
            ),
            "investor_blink_v5": RobotHeadMotif(
                name="investor_blink_v5",
                description="Operator-only V5 blink proof.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(
                        label="close",
                        values={"left_lids": -0.78, "right_lids": -0.78},
                        hold_ms=180,
                    ),
                    RobotHeadMotifStep(
                        label="open",
                        values={"left_lids": 0.16, "right_lids": 0.16},
                        hold_ms=0,
                    ),
                ],
            ),
            "investor_brows_both_v6": RobotHeadMotif(
                name="investor_brows_both_v6",
                description="Operator-only V6 both-brows proof.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(
                        label="raise",
                        values={"left_brow": 0.48, "right_brow": 0.48},
                        hold_ms=620,
                    ),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_brow_left_v6": RobotHeadMotif(
                name="investor_brow_left_v6",
                description="Operator-only V6 left-brow proof.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(label="raise", values={"left_brow": 0.52}, hold_ms=620),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_brow_right_v6": RobotHeadMotif(
                name="investor_brow_right_v6",
                description="Operator-only V6 right-brow proof.",
                public=False,
                preset="operator_proof_safe",
                steps=[
                    RobotHeadMotifStep(label="raise", values={"right_brow": 0.52}, hold_ms=620),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_neck_tilt_left_v7": RobotHeadMotif(
                name="investor_neck_tilt_left_v7",
                description="Operator-only V7 protective neck-tilt proof to the left.",
                public=False,
                preset="operator_neck_safe",
                steps=[
                    RobotHeadMotifStep(
                        label="tilt",
                        values={"neck_tilt": -0.78, "eye_yaw": -0.18},
                        hold_ms=760,
                    ),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_neck_tilt_right_v7": RobotHeadMotif(
                name="investor_neck_tilt_right_v7",
                description="Operator-only V7 protective neck-tilt proof to the right.",
                public=False,
                preset="operator_neck_safe",
                steps=[
                    RobotHeadMotifStep(
                        label="tilt",
                        values={"neck_tilt": 0.78, "eye_yaw": 0.18},
                        hold_ms=760,
                    ),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_neck_pitch_up_v7": RobotHeadMotif(
                name="investor_neck_pitch_up_v7",
                description="Operator-only V7 protective neck-pitch proof upward.",
                public=False,
                preset="operator_neck_safe",
                steps=[
                    RobotHeadMotifStep(
                        label="lift",
                        values={"neck_pitch": 0.42, "eye_pitch": 0.18},
                        hold_ms=760,
                    ),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "investor_neck_pitch_down_v7": RobotHeadMotif(
                name="investor_neck_pitch_down_v7",
                description="Operator-only V7 protective neck-pitch proof downward.",
                public=False,
                preset="operator_neck_safe",
                steps=[
                    RobotHeadMotifStep(
                        label="lower",
                        values={"neck_pitch": -0.42, "eye_pitch": -0.22},
                        hold_ms=760,
                    ),
                    RobotHeadMotifStep(label="return", values={}, hold_ms=0),
                ],
            ),
            "attentive_notice_right": RobotHeadMotif(
                name="attentive_notice_right",
                description="Operator-only V8 attentive notice motif to the right.",
                public=False,
                preset="operator_expressive_v8",
                steps=[
                    RobotHeadMotifStep(
                        label="structural_set",
                        values={"head_turn": 0.72},
                        hold_ms=820,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set",
                        values={
                            "head_turn": 0.72,
                            "eye_yaw": 0.58,
                            "left_brow": 0.36,
                            "right_brow": 0.36,
                            "left_lids": 0.16,
                            "right_lids": 0.16,
                        },
                        hold_ms=700,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release",
                        values={
                            "head_turn": 0.72,
                            "eye_yaw": 0.24,
                            "left_lids": 0.12,
                            "right_lids": 0.12,
                        },
                        hold_ms=420,
                    ),
                    RobotHeadMotifStep(label="return_to_neutral", values={}, hold_ms=0),
                ],
            ),
            "attentive_notice_left": RobotHeadMotif(
                name="attentive_notice_left",
                description="Operator-only V8 attentive notice motif to the left.",
                public=False,
                preset="operator_expressive_v8",
                steps=[
                    RobotHeadMotifStep(
                        label="structural_set",
                        values={"head_turn": -0.72},
                        hold_ms=820,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set",
                        values={
                            "head_turn": -0.72,
                            "eye_yaw": -0.58,
                            "left_brow": 0.36,
                            "right_brow": 0.36,
                            "left_lids": 0.16,
                            "right_lids": 0.16,
                        },
                        hold_ms=700,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release",
                        values={
                            "head_turn": -0.72,
                            "eye_yaw": -0.24,
                            "left_lids": 0.12,
                            "right_lids": 0.12,
                        },
                        hold_ms=420,
                    ),
                    RobotHeadMotifStep(label="return_to_neutral", values={}, hold_ms=0),
                ],
            ),
            "guarded_close_right": RobotHeadMotif(
                name="guarded_close_right",
                description="Operator-only V8 guarded close motif to the right.",
                public=False,
                preset="operator_expressive_v8",
                steps=[
                    RobotHeadMotifStep(
                        label="structural_set",
                        values={"head_turn": 0.74},
                        hold_ms=960,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set",
                        values={
                            "head_turn": 0.74,
                            "eye_yaw": 0.56,
                            "left_lids": -0.78,
                            "right_lids": -0.78,
                        },
                        hold_ms=760,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set_brows",
                        values={
                            "head_turn": 0.74,
                            "eye_yaw": 0.56,
                            "left_lids": -0.78,
                            "right_lids": -0.78,
                            "left_brow": -0.22,
                            "right_brow": -0.22,
                        },
                        hold_ms=760,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release_brows",
                        values={
                            "head_turn": 0.74,
                            "eye_yaw": 0.56,
                            "left_lids": -0.78,
                            "right_lids": -0.78,
                        },
                        hold_ms=500,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release_lids",
                        values={
                            "head_turn": 0.74,
                            "eye_yaw": 0.32,
                            "left_lids": 0.14,
                            "right_lids": 0.14,
                        },
                        hold_ms=500,
                    ),
                    RobotHeadMotifStep(label="return_to_neutral", values={}, hold_ms=0),
                ],
            ),
            "guarded_close_left": RobotHeadMotif(
                name="guarded_close_left",
                description="Operator-only V8 guarded close motif to the left.",
                public=False,
                preset="operator_expressive_v8",
                steps=[
                    RobotHeadMotifStep(
                        label="structural_set",
                        values={"head_turn": -0.74},
                        hold_ms=960,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set",
                        values={
                            "head_turn": -0.74,
                            "eye_yaw": -0.56,
                            "left_lids": -0.78,
                            "right_lids": -0.78,
                        },
                        hold_ms=760,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set_brows",
                        values={
                            "head_turn": -0.74,
                            "eye_yaw": -0.56,
                            "left_lids": -0.78,
                            "right_lids": -0.78,
                            "left_brow": -0.22,
                            "right_brow": -0.22,
                        },
                        hold_ms=760,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release_brows",
                        values={
                            "head_turn": -0.74,
                            "eye_yaw": -0.56,
                            "left_lids": -0.78,
                            "right_lids": -0.78,
                        },
                        hold_ms=500,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release_lids",
                        values={
                            "head_turn": -0.74,
                            "eye_yaw": -0.32,
                            "left_lids": 0.14,
                            "right_lids": 0.14,
                        },
                        hold_ms=500,
                    ),
                    RobotHeadMotifStep(label="return_to_neutral", values={}, hold_ms=0),
                ],
            ),
            "curious_lift": RobotHeadMotif(
                name="curious_lift",
                description="Operator-only V8 curious lift motif.",
                public=False,
                preset="operator_expressive_v8",
                steps=[
                    RobotHeadMotifStep(
                        label="structural_set",
                        values={"neck_pitch": 0.34},
                        hold_ms=900,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set",
                        values={
                            "neck_pitch": 0.34,
                            "eye_pitch": 0.32,
                            "left_brow": 0.42,
                            "right_brow": 0.42,
                            "left_lids": 0.18,
                            "right_lids": 0.18,
                        },
                        hold_ms=700,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release",
                        values={
                            "neck_pitch": 0.14,
                            "eye_pitch": 0.14,
                            "left_brow": 0.18,
                            "right_brow": 0.18,
                        },
                        hold_ms=420,
                    ),
                    RobotHeadMotifStep(label="return_to_neutral", values={}, hold_ms=0),
                ],
            ),
            "reflective_lower": RobotHeadMotif(
                name="reflective_lower",
                description="Operator-only V8 reflective lower motif.",
                public=False,
                preset="operator_expressive_v8",
                steps=[
                    RobotHeadMotifStep(
                        label="structural_set",
                        values={"neck_pitch": -0.34},
                        hold_ms=900,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set",
                        values={
                            "neck_pitch": -0.34,
                            "eye_pitch": -0.3,
                            "left_brow": -0.18,
                            "right_brow": -0.18,
                            "left_lids": 0.08,
                            "right_lids": 0.08,
                        },
                        hold_ms=700,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release",
                        values={
                            "neck_pitch": -0.14,
                            "eye_pitch": -0.14,
                            "left_brow": -0.08,
                            "right_brow": -0.08,
                        },
                        hold_ms=420,
                    ),
                    RobotHeadMotifStep(label="return_to_neutral", values={}, hold_ms=0),
                ],
            ),
            "skeptical_tilt_right": RobotHeadMotif(
                name="skeptical_tilt_right",
                description="Operator-only V8 skeptical tilt-right motif.",
                public=False,
                preset="operator_neck_safe",
                steps=[
                    RobotHeadMotifStep(
                        label="structural_set",
                        values={"neck_tilt": 0.72},
                        hold_ms=900,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set",
                        values={
                            "neck_tilt": 0.72,
                            "eye_yaw": 0.34,
                            "left_brow": 0.38,
                            "right_brow": -0.2,
                            "left_lids": 0.12,
                            "right_lids": 0.04,
                        },
                        hold_ms=700,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release",
                        values={"neck_tilt": 0.32, "eye_yaw": 0.12},
                        hold_ms=420,
                    ),
                    RobotHeadMotifStep(label="return_to_neutral", values={}, hold_ms=0),
                ],
            ),
            "empathetic_tilt_left": RobotHeadMotif(
                name="empathetic_tilt_left",
                description="Operator-only V8 empathetic tilt-left motif.",
                public=False,
                preset="operator_neck_safe",
                steps=[
                    RobotHeadMotifStep(
                        label="structural_set",
                        values={"neck_tilt": -0.72},
                        hold_ms=900,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set",
                        values={
                            "neck_tilt": -0.72,
                            "left_lids": 0.18,
                            "right_lids": 0.18,
                            "left_brow": 0.24,
                            "right_brow": 0.24,
                            "eye_pitch": -0.06,
                        },
                        hold_ms=700,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release",
                        values={
                            "neck_tilt": -0.32,
                            "left_lids": 0.12,
                            "right_lids": 0.12,
                            "left_brow": 0.12,
                            "right_brow": 0.12,
                        },
                        hold_ms=420,
                    ),
                    RobotHeadMotifStep(label="return_to_neutral", values={}, hold_ms=0),
                ],
            ),
            "playful_peek_right": RobotHeadMotif(
                name="playful_peek_right",
                description="Operator-only V8 playful peek-right motif.",
                public=False,
                preset="operator_expressive_v8",
                steps=[
                    RobotHeadMotifStep(
                        label="structural_set",
                        values={"head_turn": 0.5},
                        hold_ms=760,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set",
                        values={
                            "head_turn": 0.5,
                            "eye_yaw": 0.48,
                            "right_lids": -0.78,
                            "left_brow": 0.22,
                            "right_brow": 0.4,
                        },
                        hold_ms=700,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release",
                        values={"head_turn": 0.32, "eye_yaw": 0.22, "right_lids": 0.14},
                        hold_ms=420,
                    ),
                    RobotHeadMotifStep(label="return_to_neutral", values={}, hold_ms=0),
                ],
            ),
            "playful_peek_left": RobotHeadMotif(
                name="playful_peek_left",
                description="Operator-only V8 playful peek-left motif.",
                public=False,
                preset="operator_expressive_v8",
                steps=[
                    RobotHeadMotifStep(
                        label="structural_set",
                        values={"head_turn": -0.5},
                        hold_ms=760,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set",
                        values={
                            "head_turn": -0.5,
                            "eye_yaw": -0.48,
                            "left_lids": -0.78,
                            "left_brow": 0.4,
                            "right_brow": 0.22,
                        },
                        hold_ms=700,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release",
                        values={"head_turn": -0.32, "eye_yaw": -0.22, "left_lids": 0.14},
                        hold_ms=420,
                    ),
                    RobotHeadMotifStep(label="return_to_neutral", values={}, hold_ms=0),
                ],
            ),
            "bright_reengage": RobotHeadMotif(
                name="bright_reengage",
                description="Operator-only V8 bright reengage motif.",
                public=False,
                preset="operator_expressive_v8",
                steps=[
                    RobotHeadMotifStep(
                        label="structural_set",
                        values={"head_turn": -0.46},
                        hold_ms=760,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set",
                        values={
                            "head_turn": -0.46,
                            "eye_yaw": -0.42,
                            "eye_pitch": 0.12,
                            "left_lids": 0.24,
                            "right_lids": 0.24,
                            "left_brow": 0.42,
                            "right_brow": 0.42,
                        },
                        hold_ms=700,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release",
                        values={
                            "head_turn": -0.24,
                            "eye_yaw": -0.16,
                            "left_lids": 0.14,
                            "right_lids": 0.14,
                            "left_brow": 0.18,
                            "right_brow": 0.18,
                        },
                        hold_ms=420,
                    ),
                    RobotHeadMotifStep(label="return_to_neutral", values={}, hold_ms=0),
                ],
            ),
            "doubtful_side_glance": RobotHeadMotif(
                name="doubtful_side_glance",
                description="Operator-only V8 doubtful side-glance motif.",
                public=False,
                preset="operator_expressive_v8",
                steps=[
                    RobotHeadMotifStep(
                        label="structural_set",
                        values={"head_turn": -0.44},
                        hold_ms=760,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_set",
                        values={
                            "head_turn": -0.44,
                            "eye_yaw": -0.52,
                            "left_brow": 0.12,
                            "right_brow": 0.42,
                            "left_lids": 0.08,
                            "right_lids": 0.02,
                        },
                        hold_ms=700,
                    ),
                    RobotHeadMotifStep(
                        label="expressive_release",
                        values={"head_turn": -0.22, "eye_yaw": -0.18},
                        hold_ms=420,
                    ),
                    RobotHeadMotifStep(label="return_to_neutral", values={}, hold_ms=0),
                ],
            ),
        },
    )


def load_robot_head_capability_catalog(
    path: Optional[str | Path] = None,
) -> RobotHeadCapabilityCatalog:
    """Load a robot-head capability catalog from disk or the built-in default."""
    if path in (None, ""):
        return build_default_robot_head_catalog()

    catalog_path = Path(path).expanduser()
    payload = catalog_path.read_text(encoding="utf-8")
    return RobotHeadCapabilityCatalog.model_validate_json(payload)
