import asyncio
import os
import re
from dataclasses import dataclass

import numpy as np
import pytest

from tests._optional import BROWSER_E2E, OPTIONAL_RUNTIME, require_optional_modules

if os.getenv("BLINK_RUN_BROWSER_E2E") != "1":
    pytest.skip(
        "Browser E2E smoke tests are opt-in. Set BLINK_RUN_BROWSER_E2E=1 to run them.",
        allow_module_level=True,
    )

pytestmark = [OPTIONAL_RUNTIME, BROWSER_E2E]
require_optional_modules("fastapi", "aiortc", "playwright.async_api", "uvicorn")

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from blink.cli.local_browser import LocalBrowserConfig, create_app
from blink.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, UserImageRawFrame
from blink.pipeline.pipeline import Pipeline
from blink.pipeline.task import PipelineParams, PipelineTask
from blink.processors.aggregators.llm_context import LLMContext
from blink.processors.frame_processor import FrameDirection, FrameProcessor
from blink.transcriptions.language import Language
from blink.web.server_startup import start_uvicorn_server


@dataclass
class BrowserSmokeSignals:
    audio_seen: asyncio.Event
    camera_seen: asyncio.Event
    response_sent: asyncio.Event


def _smoke_tone_pcm(
    *,
    sample_rate: int = 24_000,
    duration_secs: float = 0.25,
    frequency_hz: float = 440.0,
) -> bytes:
    sample_count = int(sample_rate * duration_secs)
    timeline = np.linspace(0, duration_secs, sample_count, endpoint=False)
    tone = 0.15 * np.sin(2 * np.pi * frequency_hz * timeline)
    return np.clip(tone * np.iinfo(np.int16).max, -32768, 32767).astype(np.int16).tobytes()


class BrowserSmokeProbe(FrameProcessor):
    def __init__(self, signals: BrowserSmokeSignals):
        super().__init__(name="browser-smoke-probe")
        self._signals = signals
        self._response_audio = _smoke_tone_pcm()
        self._response_sent = False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._signals.audio_seen.set()
        elif isinstance(frame, UserImageRawFrame) and frame.transport_source == "camera":
            self._signals.camera_seen.set()
        else:
            await self.push_frame(frame, direction)
            return

        if (
            not self._response_sent
            and self._signals.audio_seen.is_set()
            and self._signals.camera_seen.is_set()
        ):
            self._response_sent = True
            self._signals.response_sent.set()
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=self._response_audio,
                    sample_rate=24_000,
                    num_channels=1,
                ),
                FrameDirection.DOWNSTREAM,
            )


def _build_smoke_runtime(signals: BrowserSmokeSignals):
    def _runtime_builder(config, *, transport, idle_timeout_secs=None, **kwargs):
        del config, kwargs
        context = LLMContext()
        task = PipelineTask(
            Pipeline(
                [
                    transport.input(),
                    BrowserSmokeProbe(signals),
                    transport.output(),
                ]
            ),
            params=PipelineParams(enable_metrics=False, enable_usage_metrics=False),
            idle_timeout_secs=idle_timeout_secs,
        )
        return task, context

    return _runtime_builder


async def _click_first_matching_button(page, patterns: list[str]) -> str | None:
    buttons = page.locator("button")
    count = await buttons.count()
    for index in range(count):
        button = buttons.nth(index)
        parts = [
            (await button.inner_text()).strip(),
            ((await button.get_attribute("aria-label")) or "").strip(),
            ((await button.get_attribute("title")) or "").strip(),
        ]
        label = " ".join(part for part in parts if part)
        if not label:
            continue
        if any(re.search(pattern, label, re.IGNORECASE) for pattern in patterns):
            await button.click()
            return label
    return None


async def _count_enabled_action_buttons(page) -> int:
    return await page.locator("button").evaluate_all(
        """els => els.filter(el => {
            const text = (el.innerText || "").trim();
            const aria = el.getAttribute("aria-label") || "";
            const role = el.getAttribute("role") || "";
            return (
                !el.disabled &&
                role !== "tab" &&
                !["Toggle theme", "Connect", "Conversation", "CONVERSATION", "Metrics", "METRICS"].includes(text) &&
                aria !== "Copy to clipboard"
            );
        }).length"""
    )


@pytest.mark.asyncio
async def test_browser_client_smoke_voice_and_camera(unused_tcp_port):
    signals = BrowserSmokeSignals(
        audio_seen=asyncio.Event(),
        camera_seen=asyncio.Event(),
        response_sent=asyncio.Event(),
    )
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="smoke-model",
        system_prompt="Smoke test prompt",
        language=Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="zf_xiaobei",
        tts_base_url=None,
        host="127.0.0.1",
        port=unused_tcp_port,
        vision_enabled=True,
    )
    app, uvicorn = create_app(config, runtime_builder=_build_smoke_runtime(signals))
    server = uvicorn.Server(
        uvicorn.Config(app, host=config.host, port=config.port, log_level="warning")
    )
    serve_task = await start_uvicorn_server(
        server,
        host=config.host,
        port=config.port,
        ready_path="/client/",
    )

    try:
        async with async_playwright() as playwright:
            try:
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--use-fake-ui-for-media-stream",
                        "--use-fake-device-for-media-stream",
                        "--autoplay-policy=no-user-gesture-required",
                    ],
                )
            except PlaywrightError as exc:
                if "Executable doesn't exist" in str(exc):
                    pytest.skip("Chromium is not installed. Run `uv run python -m playwright install chromium`.")
                raise

            try:
                context = await browser.new_context()
                origin = f"http://{config.host}:{config.port}"
                await context.grant_permissions(["camera", "microphone"], origin=origin)
                page = await context.new_page()
                console_errors: list[str] = []
                page.on(
                    "console",
                    lambda message: console_errors.append(message.text)
                    if message.type == "error"
                    else None,
                )
                await page.goto(f"{origin}/client/")
                await page.wait_for_function(
                    "document.title === 'Blink Voice' && document.querySelector('#root')?.childElementCount > 0"
                )
                await page.wait_for_function(
                    """
                    () => Array.from(document.querySelectorAll("button")).some(
                        button => (button.innerText || "").trim() === "Connect"
                    )
                    """,
                    timeout=15_000,
                )
                enabled_before_connect = await _count_enabled_action_buttons(page)

                assert await _click_first_matching_button(
                    page, [r"connect", r"start", r"join", r"talk"]
                ) == "Connect"

                await page.wait_for_function(
                    """
                    () => {
                        const text = document.body.innerText || "";
                        return text.includes("Client") && text.includes("INITIALIZED");
                    }
                    """,
                    timeout=15_000,
                )
                await page.wait_for_function(
                    """
                    () => {
                        const text = document.body.innerText || "";
                        return text.includes("Track started: audio");
                    }
                    """,
                    timeout=15_000,
                )

                enabled_after_connect = await _count_enabled_action_buttons(page)

                assert await page.title() == "Blink Voice"
                assert enabled_after_connect > enabled_before_connect
                assert console_errors == []
            finally:
                await browser.close()
    finally:
        server.should_exit = True
        await asyncio.wait_for(serve_task, timeout=5.0)
