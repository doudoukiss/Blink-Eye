# Safety, privacy, and consent

## Default safety posture

- Browser camera and microphone access must remain user-permissioned.
- Moondream/vision use must be visible: camera available is not the same as vision used.
- Persistent traces must not store raw audio, raw images, SDP, ICE candidates, secrets, credentials, hidden prompts, full messages, or unbounded transcripts.
- Debug transcript/image persistence must be opt-in, local-only, and clearly labeled.
- Memory must be inspectable and editable by the user.
- Blink must not claim a fake human biography or pretend to have human life experiences.

## Camera and vision honesty

Allowed:

```text
“I’m looking now.”
“I used the latest camera frame.”
“The frame looks stale, so I may be wrong.”
“Camera is available, but I have not used it for this answer.”
```

Disallowed:

```text
“I can see everything continuously” when only on-demand snapshots are used.
“I noticed earlier…” unless that observation was actually stored with consent.
```

## Avatar-readiness boundary

This upgrade defines an event contract for future avatar or abstract visual surfaces. It does not implement realistic human likeness generation, identity cloning, or face reenactment.
