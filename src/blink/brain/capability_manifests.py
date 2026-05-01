"""Provider-light capability manifest rendering helpers."""

from __future__ import annotations

from blink.brain.capabilities import CapabilityRegistry
from blink.transcriptions.language import Language


def _public_tool_descriptions(language: Language) -> dict[str, str]:
    """Return public embodied-tool descriptions in stable order."""
    if language.value.lower().startswith(("zh", "cmn")):
        return {
            "robot_head_blink": "让机器人头眨眼一次。",
            "robot_head_wink_left": "让机器人头左眼眨一下。",
            "robot_head_wink_right": "让机器人头右眼眨一下。",
            "robot_head_look_left": "让机器人头明显地向左看一下。",
            "robot_head_look_right": "让机器人头明显地向右看一下。",
            "robot_head_return_neutral": "让机器人头回到中位和中性状态。",
            "robot_head_status": "获取机器人头当前状态。",
        }
    return {
        "robot_head_blink": "Blink once.",
        "robot_head_wink_left": "Wink the left eye once.",
        "robot_head_wink_right": "Wink the right eye once.",
        "robot_head_look_left": "Look left once.",
        "robot_head_look_right": "Look right once.",
        "robot_head_return_neutral": "Return the robot head to neutral.",
        "robot_head_status": "Report the current robot-head status.",
    }


def render_capability_manifest(
    *,
    language: Language | None,
    registry: CapabilityRegistry | None,
    fallback_text: str = "",
) -> str:
    """Render the prompt-facing capability manifest from registry metadata."""
    if registry is None:
        return fallback_text

    localized = _public_tool_descriptions(language or Language.EN)
    lines: list[str] = []
    for definition in registry.public_definitions():
        tool_name = definition.tool_name or definition.capability_id
        description = localized.get(tool_name) or definition.tool_description or definition.description
        lines.append(f"- {tool_name}: {description}")
    rendered = "\n".join(lines).strip()
    return rendered or fallback_text


def render_internal_capability_manifest(
    *,
    language: Language | None,
    registry: CapabilityRegistry | None,
) -> str:
    """Render the planning-facing internal capability summary."""
    if registry is None:
        return ""
    resolved_language = language or Language.EN
    is_zh = resolved_language.value.lower().startswith(("zh", "cmn"))
    lines: list[str] = []
    for definition in registry.internal_definitions():
        policy = definition.initiative_policy
        if policy is None:
            continue
        goal_families = ", ".join(policy.allowed_goal_families) or "none"
        initiative_classes = ", ".join(policy.allowed_initiative_classes) or "goal_only"
        operator_visible = ("是" if policy.operator_visible else "否") if is_zh else (
            "yes" if policy.operator_visible else "no"
        )
        if is_zh:
            lines.append(
                f"- {definition.capability_id} [{definition.family}]: "
                f"families={goal_families}, initiatives={initiative_classes}, operator_visible={operator_visible}"
            )
        else:
            lines.append(
                f"- {definition.capability_id} [{definition.family}]: "
                f"families={goal_families}, initiatives={initiative_classes}, operator_visible={operator_visible}"
            )
    return "\n".join(lines).strip()
