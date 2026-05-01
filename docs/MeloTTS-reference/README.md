# MeloTTS Reference Environment

This directory holds the isolated environment used by the repo-owned MeloTTS HTTP-WAV server.

It is part of the Blink local product path, while the underlying runtime remains the shared Blink framework core.

It is only one part of the overall Chinese-conversation adaptation effort. For
the full system record, see
[`../chinese-conversation-adaptation.md`](../chinese-conversation-adaptation.md).

Use these commands from the repository root:

```bash
./scripts/bootstrap-melotts-reference.sh
./scripts/run-melotts-reference-server.sh
```

The bootstrap now does more than create a virtual environment:

- clones a fresh MeloTTS source checkout into `docs/MeloTTS-reference/vendor/MeloTTS`
- applies a repo-owned zh/en Apple Silicon patch layer
- installs the tested runtime from `docs/MeloTTS-reference/runtime-requirements.txt`
- prefetches the Chinese and English assets used by the local Blink sidecar

This isolated environment exists so the main runtime package can keep its own
dependency graph while still recommending MeloTTS as the primary Chinese-quality
path behind `local-http-wav`.

Generated reference artifacts under `docs/MeloTTS-reference/.venv` and
`docs/MeloTTS-reference/vendor` are created on demand and intentionally kept
out of version control.
