# Safety, privacy, and consent boundaries

## Default data boundary

Default public state, events, traces, bench artifacts, and preference records must not store:

- raw audio
- raw camera frames/images
- SDP, ICE candidates, device identifiers, or network internals
- credentials, tokens, or secrets
- hidden prompts or full model messages
- raw memory bodies
- unbounded transcript text
- realistic human avatar identity references

Live browser state may show bounded user-visible text for current UX feedback. Persistent traces should store IDs, hashes, counts, labels, reason codes, durations, and bounded summaries.

## Camera boundary

Camera is explicit. Blink must show whether camera is available, whether it is currently looking, whether a frame was actually used, frame age/staleness, and when vision is unavailable. Continuous perception remains off by default.

## Memory boundary

Memory use must be inspectable and correctable. A memory selected for a reply must have a visible behavioral effect or be suppressed. Contradicted memories are not silently reused.

## Persona boundary

Blink may have a stable AI persona. It must not invent a human life history, claim embodied sensations it does not have, or use manipulative intimacy. Person-like means coherent, reactive, and socially legible, not fake-human.

## Avatar boundary

V3 may define an adapter contract for abstract/status/symbolic avatars. It must not implement realistic human likeness generation, identity cloning, face reenactment, lip-sync human video, raw camera passthrough, or hidden-prompt avatar control.
