# Blink Examples

This directory contains both local-first examples and provider-specific examples. Do not treat the whole directory as equally runnable on a fresh MacBook checkout.

## Start With The Local Suite

For the practical local MBP workflow, start with [`LOCAL_DEVELOPMENT.md`](../LOCAL_DEVELOPMENT.md) and then run the curated local examples in [`local/`](./local/).

Recommended bootstrap commands:

```bash
./scripts/bootstrap-blink-mac.sh --profile text
./scripts/bootstrap-blink-mac.sh --profile voice
./scripts/bootstrap-blink-mac.sh --profile browser
```

If `.env` does not exist yet, the bootstrap script creates it from [`env.local.example`](../env.local.example).

## Curated Local Examples

These examples are the repo’s first-class local development path:

### [`local/01-ollama-terminal-chat.py`](./local/01-ollama-terminal-chat.py)

Text-only terminal chat with Ollama. No audio or browser setup required. The default reply mode is Simplified Chinese.

```bash
uv run python examples/local/01-ollama-terminal-chat.py
```

### [`local/02-local-native-voice.py`](./local/02-local-native-voice.py)

Native microphone and speaker voice loop on macOS using local STT, Ollama, and local TTS. The default spoken Chinese path uses Kokoro so it runs directly on Apple Silicon. XTTS and `local-http-wav` remain optional upgrades.

```bash
uv run python examples/local/02-local-native-voice.py
```

### [`local/03-local-browser-voice.py`](./local/03-local-browser-voice.py)

Browser/WebRTC voice flow using the same local STT/LLM/TTS stack.

```bash
uv run python examples/local/03-local-browser-voice.py
```

Then open `http://127.0.0.1:7860/client/`.

### [`local/04-local-transcription.py`](./local/04-local-transcription.py)

Microphone transcription loop using the configured local STT backend.

```bash
uv run python examples/local/04-local-transcription.py
```

### [`local/05-local-vision-moondream.py`](./local/05-local-vision-moondream.py)

Optional local image description with Moondream.

```bash
uv run python examples/local/05-local-vision-moondream.py
```

## Important Setup Notes

- Native local audio requires `portaudio`.
- The browser/WebRTC flow requires the `runner` and `webrtc` extras.
- Many provider-specific examples still require `.env` keys from [`env.example`](../env.example).
- Use [`LOCAL_CAPABILITY_MATRIX.md`](../LOCAL_CAPABILITY_MATRIX.md) when deciding whether an example should be treated as local, local-with-setup, or external.

## Chinese Local Workflow

For the Chinese-default local product path, use the maintained repo docs:

- [`docs/chinese-conversation-adaptation.md`](../docs/chinese-conversation-adaptation.md)
- [`LOCAL_DEVELOPMENT.md`](../LOCAL_DEVELOPMENT.md)
- [`LOCAL_CAPABILITY_MATRIX.md`](../LOCAL_CAPABILITY_MATRIX.md)

## What The Other Directories Mean

### [`getting-started/`](./getting-started/)

Progressive framework examples. Some are lightweight, some assume provider credentials.

### [`voice/`](./voice/)

Voice pipelines focused on hosted STT, TTS, and LLM providers. These are not the default local workflow.

### [`transcription/`](./transcription/)

Speech-to-text examples across both local and hosted backends.

### [`vision/`](./vision/)

Image-description and multimodal examples. Some are local, many are provider-specific.

### [`transports/`](./transports/)

Transport-focused examples such as Daily, LiveKit, and SmallWebRTC.

### Other directories

The remaining directories are feature demos, provider demos, or integration examples. They are useful once you know which transport or provider you want, but they should not be treated as the local getting-started path.

## External Workflows

Some examples still depend on external services or infrastructure:

- Daily
- LiveKit
- Twilio
- WhatsApp
- hosted STT/TTS providers
- hosted realtime speech-to-speech APIs
- hosted video avatars

For those, start from [`env.example`](../env.example) and the provider-specific example docs instead of the local bootstrap.
