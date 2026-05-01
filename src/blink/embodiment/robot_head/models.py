"""Pydantic models for Blink's robot head embodiment layer."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RobotHeadDriverMode(str, Enum):
    """Supported robot-head driver modes."""

    NONE = "none"
    MOCK = "mock"
    PREVIEW = "preview"
    SIMULATION = "simulation"
    LIVE = "live"


class RobotHeadUnitCategory(str, Enum):
    """Semantic categories for planner-facing robot-head units."""

    STRUCTURAL = "structural"
    EXPRESSIVE = "expressive"


class RobotHeadUnitProfile(BaseModel):
    """Semantic motion profile for one planner-facing unit."""

    model_config = ConfigDict(frozen=True)

    name: str
    category: RobotHeadUnitCategory
    normalized_min: float = -1.0
    normalized_max: float = 1.0
    preview_only: bool = False
    notes: Optional[str] = None


class RobotHeadPersistentState(BaseModel):
    """Persistent eye-area state held by the embodiment controller."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    values: dict[str, float] = Field(default_factory=dict)
    preset: str = "conversation_safe"
    public: bool = True
    preview_only: bool = False


class RobotHeadMotifStep(BaseModel):
    """One discrete step within a robot-head motif."""

    model_config = ConfigDict(frozen=True)

    label: str
    values: dict[str, float] = Field(default_factory=dict)
    hold_ms: int = 0


class RobotHeadMotif(BaseModel):
    """Short structural or expressive motion sequence."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    steps: list[RobotHeadMotifStep] = Field(default_factory=list)
    preset: str = "conversation_safe"
    public: bool = True
    preview_only: bool = False


class RobotHeadCommand(BaseModel):
    """Controller command submitted to the embodiment queue."""

    command_type: Literal["set_state", "run_motif", "return_neutral", "status"]
    state: Optional[str] = None
    motif: Optional[str] = None
    source: str = "tool"
    reason: Optional[str] = None


class RobotHeadDriverStatus(BaseModel):
    """Structured status reported by a robot-head driver."""

    mode: str
    available: bool
    armed: bool = False
    owner: Optional[str] = None
    degraded: bool = False
    preview_fallback: bool = False
    warnings: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class RobotHeadExecutionStep(BaseModel):
    """Validated execution step sent to a driver."""

    label: str
    values: dict[str, float] = Field(default_factory=dict)
    hold_ms: int = 0


class RobotHeadExecutionPlan(BaseModel):
    """Validated execution plan produced by the controller."""

    command: RobotHeadCommand
    resolved_name: str
    preset: str
    steps: list[RobotHeadExecutionStep] = Field(default_factory=list)
    preview_only: bool = False
    warnings: list[str] = Field(default_factory=list)


class RobotHeadExecutionResult(BaseModel):
    """Result returned by a driver or controller action."""

    accepted: bool
    command_type: str
    resolved_name: Optional[str] = None
    driver: str
    preview_only: bool = False
    preset: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    status: Optional[RobotHeadDriverStatus] = None
    trace_path: Optional[str] = None
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
