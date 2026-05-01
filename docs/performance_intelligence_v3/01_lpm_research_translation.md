# LPM research translation for Blink V3

The LPM paper's central claim is that performance is the externalization of intent, emotion, and personality through visual, vocal, and temporal behavior. It frames conversation as a full-duplex performance problem: a character speaks, listens, reacts, emotes, yields, and remains coherent over time.

## What transfers directly to Blink

### 1. Speaking and listening are different control streams

LPM separates speaking audio from listening audio. For Blink, the equivalent is:

```text
assistant speech stream -> TTS chunks, subtitles, speaking state, cancellation
user speech stream      -> active listener, floor control, interruption, semantic state
```

Blink keeps these streams separate all the way through performance planning and
evaluation.

### 2. Listening is active output

The paper emphasizes that listening behavior is half of conversation. Blink should therefore treat listening as an active state, not as silence. Browser UI should show what has been heard, what constraints were detected, whether Blink is still listening, whether it is ready to answer, and whether camera context is relevant.

### 3. Long-horizon stability needs anchors

LPM uses multi-reference identity images. Blink's analogue is a multi-reference persona and memory system:

```text
persona reference for interruption
persona reference for correction
persona reference for technical planning
persona reference for visual grounding
persona reference for uncertainty
persona reference for memory callback
persona reference for disagreement
persona reference for playful but truthful behavior
```

A single giant personality prompt is underspecified. Blink needs reference anchors selected by situation.

### 4. Online interaction is a scheduler problem

LPM's online runtime separates persistent visual state from refreshable condition caches, applies boundary-aligned updates, and limits lookahead. Blink needs the same idea for speech chunks, floor transitions, camera results, memory updates, and interruption handling.

### 5. Evaluation must measure performance, not only answer correctness

LPM-Bench evaluates speaking, listening, conversation, motion, identity, text controllability, and audio-video synchronization. Blink's V3 bench should evaluate state clarity, felt-heard, voice pacing, camera honesty, interruption naturalness, memory usefulness, persona consistency, enjoyment, and not-fake-human.

## What should not transfer yet

Do not train or ship a realistic human video avatar. The current safe step is to make the actor runtime avatar-ready through public-safe control frames and abstract/symbolic surfaces only.
