# 06 — Safety, privacy, and consent boundaries

## Default privacy rule

Default actor traces must not include:

- raw audio;
- raw image/video frames;
- SDP or ICE details;
- API keys, tokens, Authorization headers;
- full prompts;
- long raw transcripts;
- private memory IDs unless exposed through public-safe references.

## Camera rule

Blink must distinguish:

```text
camera enabled
camera frame available
camera frame fresh
vision tool used this frame
continuous perception enabled
```

It must not imply continuous visual understanding when only on-demand single-frame vision was used.

## Memory rule

Memory must be visible, correctable, suppressible, and forgettable. The UI should expose memory effects without turning private memory into a surveillance surface.

## Persona rule

Blink can have a stable AI presence. It must not pretend to have a human biography, body, personal emotions, or real-world experiences it does not have.

## Future avatar rule

Any realistic human-like avatar, identity reference, or likeness-conditioned video generation must require explicit consent, clear disclosure, and auditability. The current upgrade should use an abstract/non-human presence indicator only.
