from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog


def test_robot_head_catalog_exposes_expected_public_entries():
    catalog = build_default_robot_head_catalog()

    assert catalog.public_state_names() == [
        "confused",
        "focused_soft",
        "friendly",
        "listen_attentively",
        "neutral",
        "safe_idle",
        "thinking",
    ]
    assert catalog.public_motif_names() == [
        "acknowledge",
        "blink",
        "look_left",
        "look_right",
        "wink_left",
        "wink_right",
    ]


def test_robot_head_catalog_clamps_values_and_preserves_operator_only_units():
    catalog = build_default_robot_head_catalog()

    values, warnings, preview_only = catalog.validate_values(
        {
            "head_turn": 1.4,
            "neck_tilt": 1.2,
            "left_lids": -0.9,
        }
    )

    assert values["head_turn"] == 1.0
    assert values["neck_tilt"] == 0.9
    assert values["left_lids"] == -0.85
    assert preview_only is False
    assert len(warnings) == 3


def test_robot_head_persistent_states_stay_in_eye_area_only():
    catalog = build_default_robot_head_catalog()

    for state in catalog.persistent_states.values():
        for unit_name in state.values:
            assert catalog.units[unit_name].category.value == "expressive"


def test_robot_head_preview_only_motif_is_not_public():
    catalog = build_default_robot_head_catalog()

    motif = catalog.get_motif("curious_tilt")
    assert motif.preview_only is True
    assert motif.public is False


def test_robot_head_policy_private_motifs_stay_internal():
    catalog = build_default_robot_head_catalog()

    assert catalog.get_motif("listen_engage").public is False
    assert catalog.get_motif("thinking_shift").public is False
