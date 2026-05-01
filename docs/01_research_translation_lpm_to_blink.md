# 01 — Research translation: LPM ideas mapped to Blink

## Core LPM idea

LPM defines performance as the externalization of intent, emotion, and personality through visual, vocal, and temporal behavior. For Blink, this means personality and memory should not remain hidden prompt/context features. They must appear through timing, turn-taking, listening behavior, speech chunking, camera grounding, and UI feedback.

## Research ideas to steal without overbuilding

### 1. Speak/listen/idle labels become actor events

LPM constructs frame-level speaking, listening, and idle labels. Blink should not label video frames, but it should label runtime moments:

```text
listening_started
user_speech_continuing
transcribing
heard_final
thinking
looking
assistant_subtitle
tts_audio_started
speaking
interruption_candidate
interruption_accepted
waiting
```

These events should drive UI, traces, evaluation, and future avatar adapters.

### 2. Separate speaking and listening control paths

LPM separates speak audio and listen audio. Blink should separate:

```text
assistant speech control -> Melo chunking, subtitles, queue, cancellation
user listening control    -> VAD/STT, active listening, interruption, camera intent, floor state
```

This means two controllers, not one generic loop.

### 3. Multi-reference identity becomes multi-reference persona

LPM stabilizes character identity with global, multi-view, and expression references. Blink should stabilize persona with a **behavior reference bank**:

```text
how Blink handles interruption
how Blink disagrees
how Blink uses memory
how Blink corrects itself
how Blink answers technical roadmap questions
how Blink uses camera carefully
how Blink expresses uncertainty
how Blink stays concise in voice mode
```

A turn should retrieve 1–3 relevant references and compile them into a performance plan.

### 4. Online runtime states become Blink actor states

LPM’s online system progresses through warmup, idle, listening, responding, and back to idle. Blink should expose:

```text
booting
connected
listening
speech_detected
transcribing
heard
thinking
looking
speaking
overlap
interrupted
repairing
waiting
error
```

The UI should show these as first-class product states, not developer logs.

### 5. Boundary-aligned updates and controlled lookahead become interruption safety

LPM applies updates at chunk boundaries and limits generation lookahead. Blink should do the same with Melo speech:

```text
small speech chunks
subtitle before audio
bounded TTS queue
new input applies at chunk boundary
accepted interruption invalidates old generation token
stale audio is dropped
```

### 6. LPM-Bench becomes Blink-Actor-Bench

Blink should evaluate performance, not only response correctness:

```text
state clarity
felt heard
interruption naturalness
speech pacing
camera grounding
memory usefulness
persona consistency
enjoyment
not fake-human
not annoying
```
