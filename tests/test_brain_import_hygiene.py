import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_lower_brain_collection_stays_provider_light():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    pythonpath = [str(repo_root / "src")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class _BlockedProviderFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "blink.adapters.services.open_ai_adapter":
                    raise ModuleNotFoundError(f"blocked import: {fullname}")
                if fullname == "openai" or fullname.startswith("openai."):
                    raise ModuleNotFoundError(f"blocked import: {fullname}")
                if fullname == "pyloudnorm" or fullname.startswith("pyloudnorm."):
                    raise ModuleNotFoundError(f"blocked import: {fullname}")
                return None

        sys.meta_path.insert(0, _BlockedProviderFinder())

        import blink.brain.context_surfaces  # noqa: F401
        import blink.brain.context.compiler  # noqa: F401
        import blink.brain.evals.continuity_metrics  # noqa: F401
        import blink.brain.evals.replay_cases  # noqa: F401
        import blink.brain.executive  # noqa: F401
        import blink.brain.memory_layers  # noqa: F401
        import blink.brain.memory_v2  # noqa: F401
        import blink.brain.memory_v2.skills  # noqa: F401
        import blink.brain.procedural_planning  # noqa: F401
        import blink.brain.procedural_qa_report  # noqa: F401
        import blink.brain.procedural_skill_digest  # noqa: F401
        import blink.brain.procedural_skill_governance_report  # noqa: F401
        import blink.brain.practice_director  # noqa: F401
        import blink.brain.memory_v2.skill_evidence  # noqa: F401
        import blink.brain.memory_v2.skill_promotion  # noqa: F401
        import blink.brain.adapters.cards  # noqa: F401
        import blink.brain.evals.adapter_promotion  # noqa: F401
        import blink.brain.evals.sim_to_real_report  # noqa: F401
        import blink.brain.active_situation_model  # noqa: F401
        import blink.brain.active_situation_model_digest  # noqa: F401
        import blink.brain.private_working_memory  # noqa: F401
        import blink.brain.private_working_memory_digest  # noqa: F401
        import blink.brain.scene_world_state  # noqa: F401
        import blink.brain.scene_world_state_digest  # noqa: F401
        import blink.brain.runtime  # noqa: F401
        import blink.brain.runtime_shell  # noqa: F401
        import blink.brain._executive.planning  # noqa: F401
        import blink.cli.local_brain_audit  # noqa: F401
        import blink.cli.local_brain_shell  # noqa: F401
        import pytest

        raise SystemExit(
            pytest.main(
                [
                    "tests/test_brain_planning.py",
                    "tests/test_brain_context_policy.py",
                    "tests/test_brain_commitments.py",
                    "tests/test_brain_runtime.py",
                    "tests/test_brain_replay.py",
                    "tests/test_brain_continuity_evals.py",
                    "tests/test_brain_audit_reports.py",
                    "tests/test_brain_adapter_promotion.py",
                    "tests/test_brain_practice_director.py",
                    "tests/test_brain_sim_to_real_report.py",
                    "tests/test_brain_layered_memory.py",
                    "tests/test_brain_memory_v2.py",
                    "tests/test_brain_skill_evidence.py",
                    "tests/brain_properties/test_continuity_dossier_properties.py",
                    "tests/brain_properties/test_context_packet_properties.py",
                    "tests/brain_properties/test_replay_projection_properties.py",
                    "tests/brain_properties/test_private_working_memory_properties.py",
                    "tests/brain_properties/test_scene_world_state_properties.py",
                    "tests/brain_properties/test_active_situation_properties.py",
                    "tests/brain_properties/test_active_state_context_packet_properties.py",
                    "tests/brain_properties/test_runtime_reevaluation_alarm_properties.py",
                    "tests/brain_stateful/test_autonomy_wake_state_machine.py",
                    "tests/brain_stateful/test_practice_director_state_machine.py",
                    "tests/brain_stateful/test_planning_procedural_state_machine.py",
                    "tests/brain_stateful/test_presence_director_reevaluation_state_machine.py",
                    "tests/brain_stateful/test_skill_promotion_state_machine.py",
                    "tests/brain_stateful/test_adapter_promotion_state_machine.py",
                    "tests/test_brain_private_working_memory.py",
                    "tests/test_brain_active_situation_model.py",
                    "tests/test_brain_scene_world_state.py",
                    "tests/test_brain_context_packet_proofs.py",
                    "--collect-only",
                    "-q",
                ]
            )
        )
        """
    )
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_memory_package_roots_support_compat_imports():
    from blink.brain.memory_layers import BrainMemoryExporter
    from blink.brain.memory_v2 import BrainReflectionEngine

    assert BrainMemoryExporter.__name__ == "BrainMemoryExporter"
    assert BrainReflectionEngine.__name__ == "BrainReflectionEngine"


def test_blocked_provider_execution_stays_green_for_runtime_backed_replay_cases():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    pythonpath = [str(repo_root / "src")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class _BlockedProviderFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "blink.adapters.services.open_ai_adapter":
                    raise ModuleNotFoundError(f"blocked import: {fullname}")
                if fullname == "openai" or fullname.startswith("openai."):
                    raise ModuleNotFoundError(f"blocked import: {fullname}")
                if fullname == "pyloudnorm" or fullname.startswith("pyloudnorm."):
                    raise ModuleNotFoundError(f"blocked import: {fullname}")
                return None

        sys.meta_path.insert(0, _BlockedProviderFinder())

        from blink.brain.runtime import BrainRuntime  # noqa: F401
        from blink.embodiment.robot_head.policy import EmbodimentPolicyProcessor  # noqa: F401
        import pytest

        raise SystemExit(
            pytest.main(
                [
                    "tests/test_brain_replay.py::test_brain_replay_harness_rebuilds_timer_maintenance_candidate",
                    "tests/test_brain_replay.py::test_brain_replay_harness_rebuilds_phase4_presence_attention_cycle",
                    "-q",
                ]
            )
        )
        """
    )
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
