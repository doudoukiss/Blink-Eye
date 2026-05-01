import argparse
import json

from blink.cli.local_robot_head_show import main, resolve_config


def test_robot_head_show_config_reads_environment(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_DRIVER", "live")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_PORT", "/dev/cu.fake-robot-head")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_BAUD", "921600")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_HARDWARE_PROFILE_PATH", "/tmp/hardware.json")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_ARM", "1")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_ARM_TTL_SECONDS", "420")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_ALLOW_SENSITIVE_MOTION", "1")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_SHOW_REPORT_DIR", "/tmp/show-reports")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_PREVIEW_TRACE_DIR", "/tmp/show-preview")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_SIM_SCENARIO", "/tmp/show-sim.json")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_SIM_REALTIME", "1")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_SIM_TRACE_DIR", "/tmp/show-sim-traces")

    config = resolve_config(
        argparse.Namespace(
            show="v8",
            list_shows=False,
            robot_head_driver=None,
            robot_head_catalog_path=None,
            robot_head_port=None,
            robot_head_baud=None,
            robot_head_hardware_profile_path=None,
            robot_head_live_arm=False,
            robot_head_arm_ttl_seconds=None,
            robot_head_sim_scenario=None,
            robot_head_sim_realtime=False,
            allow_sensitive_motion=False,
            report_dir=None,
            preview_trace_dir=None,
            robot_head_sim_trace_dir=None,
            verbose=False,
        )
    )

    assert config.show_name == "v8"
    assert config.robot_head_driver == "live"
    assert config.robot_head_port == "/dev/cu.fake-robot-head"
    assert config.robot_head_baud == 921600
    assert config.robot_head_hardware_profile_path == "/tmp/hardware.json"
    assert config.robot_head_live_arm is True
    assert config.robot_head_arm_ttl_seconds == 420
    assert config.allow_sensitive_motion is True
    assert str(config.report_dir) == "/tmp/show-reports"
    assert str(config.preview_trace_dir) == "/tmp/show-preview"
    assert str(config.robot_head_sim_scenario_path) == "/tmp/show-sim.json"
    assert config.robot_head_sim_realtime is True
    assert str(config.simulation_trace_dir) == "/tmp/show-sim-traces"


def test_robot_head_show_config_defaults_to_simulation():
    config = resolve_config(
        argparse.Namespace(
            show="v3",
            list_shows=False,
            robot_head_driver=None,
            robot_head_catalog_path=None,
            robot_head_port=None,
            robot_head_baud=None,
            robot_head_hardware_profile_path=None,
            robot_head_live_arm=False,
            robot_head_arm_ttl_seconds=None,
            robot_head_sim_scenario=None,
            robot_head_sim_realtime=False,
            allow_sensitive_motion=False,
            report_dir=None,
            preview_trace_dir=None,
            robot_head_sim_trace_dir=None,
            verbose=False,
        )
    )

    assert config.robot_head_driver == "simulation"


def test_robot_head_show_main_lists_shows(capsys):
    exit_code = main(["--list-shows"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "investor_head_motion_v3" in captured.out
    assert "investor_expressive_motion_v8" in captured.out


def test_robot_head_show_main_runs_with_simulation(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "v3",
            "--report-dir",
            str(tmp_path / "reports"),
            "--robot-head-sim-trace-dir",
            str(tmp_path / "simulation"),
        ]
    )

    captured = capsys.readouterr()
    report_files = list((tmp_path / "reports").glob("*.json"))
    trace_files = list((tmp_path / "simulation").glob("*.json"))

    assert exit_code == 0
    assert "driver=simulation" in captured.out
    assert len(report_files) == 1
    assert len(trace_files) >= 1

    report_payload = json.loads(report_files[0].read_text())
    assert report_payload["driver_mode"] == "simulation"
