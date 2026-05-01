# 03 — Safety, Scope, and Deployment Boundaries

## Identity and avatar scope

This pass does not create a realistic human avatar. LPM-style video performance remains future work. The current pass builds the interaction/performance control layer that a future avatar could consume.

## Camera transparency

Blink may use the camera only through the browser session and explicit camera-state surfaces. If the system analyzes a single recent frame, the assistant must not claim continuous video understanding. If the frame is stale or unavailable, it must say so.

## Memory transparency

The browser UI may show short memory-use summaries, selected memories, and behavior effects. It must not expose private raw prompt internals, credentials, unrelated memory entries, or chain-of-thought.

## Barge-in policy

Protected playback is the default. Explicit barge-in is allowed only when configured and should remain visible in the UI. The system must avoid repeating the native PyAudio self-interruption failure.

## Native lane

Native English Kokoro is an isolation lane. It should not be promoted as the primary camera/interruption UX unless the media stack changes materially.
