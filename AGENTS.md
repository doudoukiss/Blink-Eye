# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Blink is the canonical product identity for this repository. The implementation installs as `blink-ai` and lives under the `blink` import namespace.

The underlying framework is still the same real-time voice and multimodal conversational agent architecture: audio/video, AI services, transports, and conversation pipelines built around frames.

## Local Development Assumptions

Treat this checkout as a local-first development environment on a MacBook Pro.

- Default Python version: `3.12`
- Default environment manager: `uv`
- Default local inference stack: `ollama`
- Default local model: `qwen3.5:4b`
- Default local language: `zh`
- Default browser Chinese voice path: MeloTTS via `local-http-wav`
- Default browser camera mode: enabled
- Default first-run workflow: `text` profile, not browser/WebRTC and not native local audio

The practical goal is to make the framework easy to run and inspect locally before adding optional transports or provider integrations.

The authoritative record of the Chinese-local adaptation effort is
[`docs/chinese-conversation-adaptation.md`](docs/chinese-conversation-adaptation.md).

## Common Commands

```bash
# Bootstrap local profiles
./scripts/bootstrap-blink-mac.sh --profile text
./scripts/bootstrap-blink-mac.sh --profile voice
./scripts/bootstrap-blink-mac.sh --profile browser
./scripts/bootstrap-blink-mac.sh --profile full
./scripts/bootstrap-blink-mac.sh --profile voice --with-piper

# Run the default local chat flow
./scripts/run-blink-chat.sh
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-chat.sh

# Run the native local voice flow
./scripts/run-blink-voice.sh
./scripts/run-local-voice-en.sh
./scripts/build-macos-camera-helper.sh
./scripts/run-local-voice-macos-camera-en.sh
./scripts/run-blink-voice.sh --tts-backend piper

# Run the recommended MeloTTS -> local-http-wav sidecar
./scripts/bootstrap-melotts-reference.sh
./scripts/run-melotts-reference-server.sh
./scripts/run-local-browser-melo.sh

# Run the English-only browser Kokoro voice path
./scripts/run-local-browser-kokoro-en.sh

# Generate reproducible TTS comparison WAVs
./scripts/eval-local-tts.sh
./scripts/eval-local-tts.sh --backend kokoro --all-kokoro-zh-voices
./scripts/eval-local-tts.sh --backend local-http-wav --language zh

# Run the optional CosyVoice -> local-http-wav sidecar
./scripts/run-local-cosyvoice-adapter.sh
./scripts/bootstrap-cosyvoice-reference.sh
./scripts/run-cosyvoice-reference-server.sh
./scripts/run-local-browser-cosyvoice.sh
./scripts/run-local-voice-cosyvoice.sh

# Run the browser/WebRTC local voice flow
./scripts/run-blink-browser.sh
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-browser.sh
./scripts/run-blink-browser.sh --tts-backend kokoro

# Diagnose the selected local profile
./scripts/doctor-blink-mac.sh --profile text

# One-shot local smoke test
uv run --python 3.12 blink-local-chat --once "Explain Blink in one sentence."

# Setup development environment for the canonical brain-core proof lane
uv sync --python 3.12 --group dev

# Setup development environment with the full recommended local MBP stack
uv sync --python 3.12 --group dev \
  --extra runner \
  --extra webrtc \
  --extra local \
  --extra mlx-whisper \
  --extra kokoro

# Install pre-commit hooks
uv run pre-commit install

# Run all tests
uv run pytest

# Run the canonical brain-core proof lane
./scripts/test-brain-core.sh

# Run one canonical proof sub-lane
./scripts/test-brain-core.sh --lane fast
./scripts/test-brain-core.sh --lane proof
./scripts/test-brain-core.sh --lane fuzz-smoke

# Run a single test file
uv run pytest tests/test_name.py

# Run a specific test
uv run pytest tests/test_name.py::test_function_name

# Preview changelog
uv run towncrier build --draft --version Unreleased

# Lint and format check
uv run ruff check
uv run ruff format --check

# Optional native Atheris lane on libFuzzer-capable machines
./scripts/test-brain-core.sh --lane atheris

# Optional local vision workflows
uv sync --python 3.12 --group dev --extra moondream

# Preview/apply the built-in Witty Sophisticated memory/personality seed
blink-memory-persona-ingest --preset witty-sophisticated --dry-run --report preview.json
blink-memory-persona-ingest --preset witty-sophisticated --apply --approved-report preview.json

# Update dependencies (after editing pyproject.toml)
uv lock && uv sync
```

## Repository Purpose

This repository is the framework itself, not a single deployable app.

When making changes, optimize for:

- clear frame flow
- composable processors
- local reproducibility
- optional integrations rather than mandatory ones

The new local terminal workflow is intentionally the baseline. Voice, WebRTC, telephony, and external providers are second-step workflows that should remain optional.

The repo also now has first-class local native voice and browser/WebRTC voice paths. Those are still optional from a dependency perspective, but they are part of the supported local development surface and should stay aligned with the docs.

For the local product path, assume Simplified Chinese by default. English remains a supported fallback, but not the default.
Chinese voice interaction quality is the main product target. The zero-setup local path is useful for bootstrap and debugging, but it is not the final quality bar.
Speech stays fixed per session for the local voice/browser flows. Do not assume in-session TTS language switching.

## Architecture

### Frame-Based Pipeline Processing

All data flows as **Frame** objects through a pipeline of **FrameProcessors**:

```
[Processor1] → [Processor2] → ... → [ProcessorN]
```

**Key components:**

- **Frames** (`src/blink/frames/frames.py`): Data units (audio, text, video) and control signals. Flow DOWNSTREAM (input→output) or UPSTREAM (acknowledgments/errors).

- **FrameProcessor** (`src/blink/processors/frame_processor.py`): Base processing unit. Each processor receives frames, processes them, and pushes results downstream.

- **Pipeline** (`src/blink/pipeline/pipeline.py`): Chains processors together.

- **ParallelPipeline** (`src/blink/pipeline/parallel_pipeline.py`): Runs multiple pipelines in parallel.

- **Transports** (`src/blink/transports/`): Transports are frame processors used for external I/O layer (Daily WebRTC, LiveKit WebRTC, WebSocket, Local). Abstract interface via `BaseTransport`, `BaseInputTransport` and `BaseOutputTransport`.

- **Pipeline Task (`src/blink/pipeline/task.py`)**: Runs and manages a pipeline. Pipeline tasks send the first frame, `StartFrame`, to the pipeline in order for processors to know they can start processing and pushing frames. Pipeline tasks internally create a pipeline with two additional processors, a source processor before the user-defined pipeline and a sink processor at the end. Those are used for multiple things: error handling, pipeline task level events, heartbeat monitoring, etc.

- **Pipeline Runner (`src/blink/pipeline/runner.py`)**: High-level entry point for executing pipeline tasks. Handles signal management (SIGINT/SIGTERM) for graceful shutdown and optional garbage collection. Run a single pipeline task with `await runner.run(task)` or multiple concurrently with `await asyncio.gather(runner.run(task1), runner.run(task2))`.

- **Services** (`src/blink/services/`): 60+ AI provider integrations (STT, TTS, LLM, etc.). Extend base classes: `AIService`, `LLMService`, `STTService`, `TTSService`, `VisionService`.

- **Serializers** (`src/blink/serializers/`): Convert frames to/from wire formats for WebSocket transports. `FrameSerializer` base class defines `serialize()` and `deserialize()`. Telephony serializers (Twilio, Plivo, Vonage, Telnyx, Exotel, Genesys) handle provider-specific protocols and audio encoding (e.g., μ-law).

- **RTVI** (`src/blink/processors/frameworks/rtvi.py`): Real-Time Voice Interface protocol bridging clients and the pipeline. `RTVIProcessor` handles incoming client messages (text input, audio, function call results). `RTVIObserver` converts pipeline frames to outgoing messages: user/bot speaking events, transcriptions, LLM/TTS lifecycle, function calls, metrics, and audio levels.

- **Observers** (`src/blink/observers/`): Monitor frame flow without modifying the pipeline. Passed to `PipelineTask` via the `observers` parameter. Implement `on_process_frame()` and `on_push_frame()` callbacks.

### Important Patterns

- **Context Aggregation**: `LLMContext` accumulates messages for LLM calls; `UserResponse` aggregates user input

- **Turn Management**: Turn management is done through `LLMUserAggregator` and
  `LLMAssistantAggregator`, created with `LLMContextAggregatorPair`

- **User turn strategies**: Detection of when the user starts and stops speaking is done via user turn start/stop strategies. They push `UserStartedSpeakingFrame` and `UserStoppedSpeakingFrame` respectively.

- **Interruptions**: Interruptions are usually triggered by a user turn start strategy (e.g. `VADUserTurnStartStrategy`) but they can be triggered by other processors as well, in which case the user turn start strategies don't need to. An `InterruptionFrame` carries an optional `asyncio.Event` that is set when the frame reaches the pipeline sink. If a processor stops an `InterruptionFrame` from propagating downstream (i.e., doesn't push it), it **must** call `frame.complete()` to avoid stalling `push_interruption_task_frame_and_wait()` callers.

- **Uninterruptible Frames**: These are frames that will not be removed from internal queues even if there's an interruption. For example, `EndFrame` and `StopFrame`.

- **Events**: Most runtime classes inherit from `BaseObject`. `BaseObject` has support for events. Events can run in the background in an async task (default) or synchronously (`sync=True`) if we want immediate action. Synchronous event handlers need to execute fast.

- **Async Task Management**: Always use `self.create_task(coroutine, name)` instead of raw `asyncio.create_task()`. The `TaskManager` automatically tracks tasks and cleans them up on processor shutdown. Use `await self.cancel_task(task, timeout)` for cancellation.

- **Error Handling**: Use `await self.push_error(msg, exception, fatal)` to push errors upstream. Services should use `fatal=False` (the default) so application code can handle errors and take action (e.g. switch to another service).

### Key Directories

| Directory                  | Purpose                                            |
| -------------------------- | -------------------------------------------------- |
| `src/blink/frames/`      | Frame definitions (100+ types)                     |
| `src/blink/processors/`  | FrameProcessor base + aggregators, filters, audio  |
| `src/blink/pipeline/`    | Pipeline orchestration                             |
| `src/blink/services/`    | AI service integrations (60+ providers)            |
| `src/blink/transports/`  | Transport layer (Daily, LiveKit, WebSocket, Local) |
| `src/blink/serializers/` | Frame serialization for WebSocket protocols        |
| `src/blink/observers/`   | Pipeline observers for monitoring frame flow       |
| `src/blink/audio/`       | VAD, filters, mixers, turn detection, DTMF         |
| `src/blink/turns/`       | User turn management                               |

## Code Style

- **Docstrings**: Google-style. Classes describe purpose; `__init__` has `Args:` section; dataclasses use `Parameters:` section.
- **Linting**: Ruff (line length 100). Pre-commit hooks enforce formatting.
- **Type hints**: Required for complex async code.
- **Dataclass vs Pydantic**: Use `@dataclass` for frames and internal pipeline data (high-frequency, no validation needed). Use Pydantic `BaseModel` for configuration, parameters, metrics, and external API data (benefits from validation and serialization). Specifically:
  - `@dataclass`: Frame types, context aggregator pairs, internal data containers
  - `BaseModel`: Service `InputParams`, transport/VAD/turn params, metrics data, API request/response models, serializer params

### Docstring Example

```python
class MyService(LLMService):
    """Description of what the service does.

    More detailed description.

    Event handlers available:

    - on_connected: Called when we are connected

    Example::

        @service.event_handler("on_connected")
        async def on_connected(service, frame):
            ...
    """

    def __init__(self, param1: str, **kwargs):
        """Initialize the service.

        Args:
            param1: Description of param1.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
```

## Service Implementation

When adding a new service:

1. Extend the appropriate base class (`STTService`, `TTSService`, `LLMService`, etc.)
2. Implement required abstract methods
3. Handle necessary frames
4. By default, all frames should be pushed in the direction they came
5. Push `ErrorFrame` on failures
6. Add metrics tracking via `MetricsData` if relevant
7. Follow the pattern of existing services in `src/blink/services/`

## Testing

Test utilities live in `src/blink/tests/utils.py`. Use `run_test()` to send frames through a pipeline and assert expected output frames in each direction. Use `SleepFrame(sleep=N)` to add delays between frames.

For brain-core changes, prefer the dedicated proof lanes before broader runtime
or media verification:

- `./scripts/test-brain-core.sh` is the canonical entrypoint
- `./scripts/test-brain-core.sh --lane fast` for import hygiene, replay, packet proofs, continuity evals, planning, and active-state unit coverage
- `./scripts/test-brain-core.sh --lane proof` for `tests/brain_properties/` and `tests/brain_stateful/`
- `./scripts/test-brain-core.sh --lane fuzz-smoke` for deterministic harness smoke
- `./scripts/test-brain-core.sh --lane atheris` for optional native coverage-guided runs on libFuzzer-capable machines

For embodied/perception changes, use the sibling proof entrypoint:

- `./scripts/test-embodied-core.sh` is the canonical embodied entrypoint
- `./scripts/test-embodied-core.sh --lane fast` for perception broker, robot-head policy/runtime, embodied shell-contract slices, and capability/action coverage
- `./scripts/test-embodied-core.sh --lane proof` for scene-world and multimodal-autobiography properties plus embodied-adjacent stateful coverage

## Local Workflow Conventions

- Start with `./scripts/run-blink-chat.sh` when you need a quick sanity check.
- Prefer the profile-based bootstrap scripts instead of ad hoc extra installation.
- Prefer text-only local verification before pulling in audio, browser, or telephony infrastructure.
- Use `./scripts/doctor-blink-mac.sh --profile ...` before assuming a local failure is inside the framework.
- Only add extras when the task requires them.
- Keep Chinese-local constraints in `system_instruction`, not `developer` messages, because the Ollama adapter ignores the `developer` role.
- Treat checked-in runtime profiles in `config/local_runtime_profiles.json` as
  the product-behavior defaults for local browser/native paths. Treat `.env` as
  machine/secrets/local override state, not as the source of truth for persona,
  prompt, language contract, camera policy, or barge-in defaults.
- Treat `./scripts/run-local-voice-en.sh` as the stable English-only Kokoro
  path without camera, Moondream, MeloTTS, or browser/WebRTC. It defaults to
  protected playback because native PyAudio has no browser/WebRTC echo
  cancellation; use `--allow-barge-in` only with headphones or another echo-safe
  setup.
- Treat `./scripts/run-local-voice-macos-camera-en.sh` as the separate
  macOS-only English Kokoro camera path. `BlinkCameraHelper.app` owns camera
  permission, Moondream runs only on explicit `fetch_user_image` calls, and it
  also defaults to protected playback. The retired Terminal/OpenCV native camera
  route must not be revived.
- Treat `./scripts/run-local-browser-melo.sh` and
  `./scripts/run-local-browser-kokoro-en.sh` as equal primary browser/WebRTC
  product paths. `browser-zh-melo` is the Chinese MeloTTS + camera path;
  `browser-en-kokoro` is the English Kokoro + camera/Moondream path with
  browser media handling.
- Treat `./scripts/run-local-browser-kokoro-en.sh` as the preferred
  English-only browser voice path when you want Kokoro without MeloTTS. It keeps
  Browser/WebRTC media handling and starts with language `en`, TTS `kokoro`,
  browser vision/Moondream available by default, continuous perception off,
  protected playback on, and no Melo sidecar. It also ignores inherited prompt
  overrides so old Chinese `.env` prompt settings do not leak into the English
  lane.
- Camera/vision plus MeloTTS for the Chinese product path stays on the browser
  path; camera/vision plus Kokoro for English daily UX also stays on the browser
  path.
- Treat Mandarin voice quality as a TTS concern. The Chinese browser path uses
  MeloTTS behind `local-http-wav`; the English browser path uses Kokoro. XTTS
  and CosyVoice remain optional secondary paths.
- Treat `./scripts/run-local-browser-melo.sh` as preserving the Chinese browser
  camera default. If a launcher change causes that wrapper to come up without
  local vision unless the user adds extra flags, treat that as a regression.
- Treat `./scripts/run-local-browser-kokoro-en.sh` the same way for English:
  it should come up with browser vision/Moondream available unless the user
  explicitly passes `--no-vision` or sets `BLINK_LOCAL_BROWSER_VISION=0`.
- The browser actor runtime is now a public product surface. Preserve
  `/api/runtime/actor-state`, `/api/runtime/actor-events`, bounded actor traces,
  browser actor surface v2, and the bilingual actor bench/release gate whenever
  changing listening, TTS, camera, interruption, memory, persona, or browser UI.
- Treat the browser and sidecar wrapper scripts as owning the child processes
  they start. If a launcher change causes stale `resource_tracker` helpers or
  abandoned sidecars to accumulate after repeated `Ctrl+C` stop/start cycles,
  treat that as a regression.
- Treat `artifacts/runtime_logs/latest-browser-melo.log` as the first local
  run log to inspect after browser Melo failures. Pair it with
  `/api/runtime/voice-metrics` so microphone delivery, STT waiting, camera
  state, and TTS sidecar health are debugged separately without exposing raw
  transcript text or audio.
- For the repeated browser "first answer works, second answer fails or sounds
  wrong" regression, read `docs/debugging.md` and
  `docs/chinese-conversation-adaptation.md` before changing code. The known
  chain was camera frame-rate pressure, too many tiny Melo chunks, then an
  overcorrection to overlarge chunks. Keep `local-http-wav` on moderate bounded
  chunks with safe request metrics, and do not re-enable browser media mutation
  or automatic WebRTC renegotiation as a speech fix.
- Keep native local audio and browser/WebRTC voice on protected playback by
  default. Use `--allow-barge-in` or `BLINK_LOCAL_ALLOW_BARGE_IN=1` only when
  the setup is echo-safe enough not to self-interrupt.
- Use `BLINK_LOCAL_TTS_VOICE_ZH` and `BLINK_LOCAL_TTS_VOICE_EN` for normal local voice routing. Treat `BLINK_LOCAL_TTS_VOICE` as a generic fallback only.
- Treat `local-http-wav` as the stable seam for higher-quality speech engines. External reference checkouts such as CosyVoice are design material, not packaged runtime code, and the MeloTTS reference server should stay outside `src/blink` so the main package dependency graph stays clean.
- The included CosyVoice adapter is a sidecar bridge to the `local-http-wav` seam, not a core framework dependency. Keep it available, but do not make it the main Chinese-quality recommendation.
- Keep documentation aligned with the actual local run path in [LOCAL_DEVELOPMENT.md](LOCAL_DEVELOPMENT.md), [LOCAL_CAPABILITY_MATRIX.md](LOCAL_CAPABILITY_MATRIX.md), [tutorial.md](tutorial.md), and [docs/chinese-conversation-adaptation.md](docs/chinese-conversation-adaptation.md).
- Treat memory/personality enrichment as preview-first. Use typed memory,
  relationship/teaching memory, and behavior-control surfaces; do not mutate
  hidden prompts, checked-in persona prose, model/provider settings, or hardware
  policy. The browser workbench Witty Sophisticated seed must remain
  public-safe, confirmation-gated, and idempotent.

## Codex Planning Persistence

Meaningful implementation roadmaps and key decision records should be captured in the public documentation that owns the relevant feature, rather than living only in chat.
