from __future__ import annotations

import pytest

from blink.brain.adapters import BrainAdapterDescriptor
from blink.brain.adapters.embodied_policy import (
    EmbodiedPolicyExecutionRequest,
    EmbodiedPolicyExecutionStep,
    LocalRobotHeadEmbodiedPolicyAdapter,
)
from blink.brain.adapters.perception import LocalPerceptionAdapter
from blink.brain.adapters.world_model import LocalDeterministicWorldModelAdapter
from blink.brain.identity import base_brain_system_prompt
from blink.brain.perception.detector import PresenceDetectionResult
from blink.brain.perception.enrichment import VisionEnrichmentResult
from blink.brain.runtime import BrainRuntime
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import PreviewDriver
from blink.transcriptions.language import Language


class DummyDetector:
    backend = "stub_presence"
    available = False

    def detect(self, frame):
        return PresenceDetectionResult.unavailable(
            backend=self.backend,
            reason="presence_detector_unavailable",
        )


class DummyEnrichment:
    available = False

    async def enrich(self, frame):
        return VisionEnrichmentResult.unavailable(reason="vision_enrichment_unavailable")


class DummyLLM:
    def register_function(self, function_name, handler):
        return None


def test_brain_adapter_descriptor_reports_supported_capabilities():
    descriptor = BrainAdapterDescriptor(
        backend_id="local_test",
        backend_version="v1",
        capabilities=("detect", "enrich"),
        degraded_mode_id="empty_result",
        default_timeout_ms=250,
    )

    assert descriptor.supports("detect") is True
    assert descriptor.supports("missing") is False
    assert descriptor.degraded_mode_id == "empty_result"
    assert descriptor.default_timeout_ms == 250


@pytest.mark.asyncio
async def test_local_perception_adapter_reports_descriptor_and_degraded_backend_state():
    adapter = LocalPerceptionAdapter(
        detector=DummyDetector(),
        enrichment=DummyEnrichment(),
    )

    detection = adapter.detect_presence(frame=None)  # type: ignore[arg-type]
    enrichment = await adapter.enrich_scene(frame=None)  # type: ignore[arg-type]

    assert adapter.descriptor.backend_id == "local_perception"
    assert adapter.descriptor.default_timeout_ms == 3000
    assert adapter.presence_detection_backend == "stub_presence"
    assert adapter.presence_detection_available is False
    assert adapter.scene_enrichment_available is False
    assert detection.available is False
    assert enrichment.available is False


def test_brain_store_defaults_to_local_world_model_adapter(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    try:
        assert isinstance(store.world_model_adapter, LocalDeterministicWorldModelAdapter)
        assert store.world_model_adapter.descriptor.backend_id == "local_world_model"
        assert store.world_model_adapter.descriptor.default_timeout_ms == 250
    finally:
        store.close()


@pytest.mark.asyncio
async def test_local_robot_head_policy_adapter_reports_backend_and_status(tmp_path):
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=PreviewDriver(trace_dir=tmp_path),
    )
    adapter = LocalRobotHeadEmbodiedPolicyAdapter(controller=controller)
    try:
        status = await adapter.status()
        result = await adapter.execute_action(
            EmbodiedPolicyExecutionRequest(
                action_id="cmd_report_status",
                source="test",
                controller_plan=(EmbodiedPolicyExecutionStep(command_type="status"),),
            )
        )

        assert adapter.descriptor.backend_id == "local_robot_head_policy"
        assert adapter.descriptor.default_timeout_ms == 5000
        assert adapter.execution_backend == "preview"
        assert status.accepted is True
        assert result.command_type == "status"
        assert result.driver == "preview"
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_brain_runtime_wires_local_default_adapters_explicitly(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="adapter-runtime")
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=PreviewDriver(trace_dir=tmp_path / "runtime-preview"),
    )
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.EN),
        language=Language.EN,
        runtime_kind="browser",
        session_resolver=lambda: session_ids,
        llm=DummyLLM(),
        robot_head_controller=controller,
        brain_db_path=tmp_path / "runtime.db",
    )
    try:
        assert isinstance(runtime.store.world_model_adapter, LocalDeterministicWorldModelAdapter)
        assert runtime.store.world_model_adapter.descriptor.backend_id == "local_world_model"
        assert runtime.action_engine is not None
        assert isinstance(runtime.action_engine.policy_adapter, LocalRobotHeadEmbodiedPolicyAdapter)
        assert runtime.action_engine.policy_adapter.descriptor.backend_id == "local_robot_head_policy"
    finally:
        await controller.close()
        runtime.close()
