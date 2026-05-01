# Dogfooding script

Run both paths separately:

```bash
./scripts/run-local-browser-melo.sh
./scripts/run-local-browser-kokoro-en.sh
```

Open:

```text
http://127.0.0.1:7860/client/
```

For each path, test:

1. Connection/profile status.
2. Long user turn with active listening.
3. Camera-on visual question.
4. Protected playback no-self-interrupt.
5. Headphone explicit interruption if echo-safe.
6. Memory write, recall, correction.
7. Five-minute free conversation and enjoyment score.
