from blink.cli.local_voice import LocalVoiceConfig, build_local_voice_runtime
from blink.processors.aggregators.llm_context import NOT_GIVEN
from blink.processors.frame_processor import FrameProcessor
from blink.transcriptions.language import Language


class DummyTransport:
    def __init__(self):
        self._input = FrameProcessor(name="robot-head-test-input")
        self._output = FrameProcessor(name="robot-head-test-output")

    def input(self):
        return self._input

    def output(self):
        return self._output


class DummyLLM:
    def __init__(self):
        self.registered_functions = {}

    def register_function(self, function_name, handler):
        self.registered_functions[function_name] = handler


class DummyLLMProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(name="robot-head-test-llm")
        self.registered_functions = {}

    def register_function(self, function_name, handler):
        self.registered_functions[function_name] = handler


def _base_config(robot_head_driver: str) -> LocalVoiceConfig:
    return LocalVoiceConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Voice prompt",
        language=Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="zf_xiaobei",
        robot_head_driver=robot_head_driver,
    )


def test_local_voice_runtime_keeps_robot_head_disabled_by_default():
    task, context = build_local_voice_runtime(
        _base_config("none"),
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=DummyLLMProcessor(),
        tts=FrameProcessor(name="tts"),
    )

    assert context.tools is not NOT_GIVEN
    assert [tool.name for tool in context.tools.standard_tools] == [
        "brain_remember_profile",
        "brain_remember_preference",
        "brain_remember_task",
            "brain_forget_memory",
            "brain_apply_memory_governance",
            "brain_complete_task",
            "brain_list_visible_memories",
            "brain_explain_memory_continuity",
        ]
    assert task is not None


def test_local_voice_runtime_registers_robot_head_when_preview_enabled():
    llm = DummyLLMProcessor()
    task, context = build_local_voice_runtime(
        _base_config("preview"),
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=FrameProcessor(name="tts"),
    )

    assert context.tools is not NOT_GIVEN
    assert sorted(llm.registered_functions) == [
        "brain_apply_memory_governance",
        "brain_complete_task",
        "brain_explain_memory_continuity",
        "brain_forget_memory",
        "brain_list_visible_memories",
        "brain_remember_preference",
        "brain_remember_profile",
        "brain_remember_task",
        "robot_head_blink",
        "robot_head_look_left",
        "robot_head_look_right",
        "robot_head_return_neutral",
        "robot_head_status",
        "robot_head_wink_left",
        "robot_head_wink_right",
    ]
    assert task is not None


def test_local_voice_runtime_registers_robot_head_when_simulation_enabled():
    llm = DummyLLMProcessor()
    task, context = build_local_voice_runtime(
        _base_config("simulation"),
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=FrameProcessor(name="tts"),
    )

    assert context.tools is not NOT_GIVEN
    assert "brain_remember_profile" in llm.registered_functions
    assert "robot_head_status" in llm.registered_functions
    assert task is not None


def test_local_voice_runtime_registers_robot_head_when_live_enabled():
    llm = DummyLLMProcessor()
    config = _base_config("live")
    config.robot_head_port = "/dev/cu.fake-robot-head"

    task, context = build_local_voice_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=FrameProcessor(name="tts"),
    )

    assert context.tools is not NOT_GIVEN
    assert "brain_remember_profile" in llm.registered_functions
    assert "robot_head_status" in llm.registered_functions
    assert task is not None


def test_local_voice_runtime_operator_mode_keeps_raw_robot_head_tools():
    llm = DummyLLMProcessor()
    config = _base_config("preview")
    config.robot_head_operator_mode = True

    task, context = build_local_voice_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=FrameProcessor(name="tts"),
    )

    assert context.tools is not NOT_GIVEN
    assert sorted(llm.registered_functions) == [
        "brain_apply_memory_governance",
        "brain_complete_task",
        "brain_explain_memory_continuity",
        "brain_forget_memory",
        "brain_list_visible_memories",
        "brain_remember_preference",
        "brain_remember_profile",
        "brain_remember_task",
        "robot_head_return_neutral",
        "robot_head_run_motif",
        "robot_head_set_state",
        "robot_head_status",
    ]
    assert task is not None
