# Manual QA Scorecard

Use this after regenerating WAVs with `./scripts/eval-local-tts.sh`.

Score each item from 1 to 5.

| Case | What to listen for | Score | Notes |
| --- | --- | --- | --- |
| `zh_01_intro` | baseline Mandarin clarity |  |  |
| `zh_02_datetime` | dates, time, and numeric naturalness |  |  |
| `zh_03_versions` | version, acronym, and technical shorthand handling |  |  |
| `zh_04_symbols` | path, URL, and symbol handling |  |  |
| `zh_05_latency` | latency units and alphanumeric ID handling |  |  |
| `mixed_01_diagnostic` | Chinese-English mixing for technical terms |  |  |
| custom technical answer | response stays short and oral |  |  |
| English fallback | English path sounds distinct from Chinese path |  |  |
| Kokoro vs Melo | whether the sidecar materially improves intelligibility |  |  |

## Suggested smoke-test prompts

Use these in voice or browser mode:

1. `请用两句话解释 Blink 的 frame 和 pipeline 有什么关系。`
2. `请口语化说明为什么本地语音输出会受到 TTS 的影响。`
3. `请解释 v2.3 版本在今天下午三点二十分完成测试。`
4. `请先用中文说，再补一句英文总结。`

## What to flag immediately

- raw URL reading
- literal markdown reading
- reading long file paths verbatim
- numbers or versions spoken in a way that becomes unintelligible
- English fallback still sounding like it is using the Chinese path

## Hybrid OpenAI Demo QA Checklist

Use this checklist before investor-facing hybrid demos. This is separate from
the WAV scorecard above because it validates the end-to-end demo lane rather
than offline TTS samples.

| Case | How to run | Pass criteria | Score | Notes |
| --- | --- | --- | --- | --- |
| LLM-only text smoke | `./scripts/smoke-hybrid-openai-demo.sh` | One concise answer returns from `openai-responses`; output is bounded and not markdown-heavy. |  | This is not STT/TTS/WebRTC proof. |
| Native voice smoke | `./scripts/run-blink-voice-openai.sh` | STT is local, answer starts directly, speech is short and understandable, TTS remains local. |  | Use the default protected-playback behavior. |
| Browser/WebRTC smoke | `./scripts/run-blink-browser-openai.sh` then open `/client/` | Browser connects, mic works, first answer is spoken, operator workbench is usable. |  | Allow mic and camera. |
| Browser Input / STT health | Open the workbench Voice section during browser voice use | Mic state moves from waiting/receiving/stalled truthfully, STT state reaches transcribed after speech, STT wait age appears only while waiting, and no raw transcript text is displayed. |  | Use this to separate WebRTC mic stalls from STT issues. |
| Browser performance state | Open `/api/runtime/performance-state` and `/api/runtime/performance-events` during `browser-zh-melo` or `browser-en-kokoro` | State shows the active profile, protected playback, mic/camera state, TTS label, and ordered public-safe events without raw transcript, prompt, audio, image, or memory payloads. |  | `browser-zh-melo` should report `local-http-wav/MeloTTS`; `browser-en-kokoro` should report `kokoro/English`. |
| Browser second-turn voice | Ask two short Chinese questions back to back in `/client/` | Both turns produce STT completions and audible answers; the second answer does not fade, distort, or stop early. |  | Inspect `artifacts/runtime_logs/latest-browser-melo.log` for bounded Melo `chars`/`bytes`/`audio_ms`/`request_ms` lines if this fails. |
| Browser model selector | Open `/client/`, choose a local model, reconnect, then inspect `/api/runtime/stack` | The selected available local profile is accepted for the new session and stack shows the chosen provider/model without secrets. |  | Remote OpenAI choices should stay locked unless explicitly enabled for demo/development. |
| Melo sidecar health | `curl http://127.0.0.1:8001/healthz` and `curl http://127.0.0.1:8001/voices` | Sidecar responds and expected speakers are visible. |  | Melo port is API-only. |
| Provider stack verification | `curl http://127.0.0.1:7860/api/runtime/stack` | Shows `openai-responses`, expected model, `demo_mode=true`, local STT, local TTS, and intended vision state. |  | Must not expose secrets or prompts. |
| Camera non-regression | Ask `请描述一下你现在看到的画面` in browser mode | Blink uses the camera tool or gives a bounded stale/uncertain-frame answer. |  | It should not deny camera access when browser permission is granted. |
| Missing credential failure | Run an OpenAI wrapper with `OPENAI_API_KEY` unset | Wrapper exits early with a clear missing-key message. |  | Do not capture or paste keys in notes. |
| Connectivity failure | Use a deliberately invalid OpenAI base URL for one test | Failure is bounded and points at the LLM provider path, not local STT/TTS. |  | Restore env immediately after test. |

## Hybrid Demo Immediate Stop Conditions

- wrapper or stack output exposes `OPENAI_API_KEY`, authorization headers,
  prompts, raw request payloads, source refs, event ids, DB paths, or memory
  internals
- `/api/runtime/stack` does not show the intended provider/model before demo
- browser camera prompt says camera is unavailable after permission is granted
- workbench Voice Input / STT health shows stalled/no-audio while the browser is
  expected to be recording
- voice answers are long, markdown-like, code-heavy, repeatedly cut off, or the
  second browser reply sounds faded/distorted after a clean first reply
