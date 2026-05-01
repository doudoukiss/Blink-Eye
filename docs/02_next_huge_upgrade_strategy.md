# 02 — Next huge upgrade strategy

## Product thesis

Blink should become a **real-time conversational actor**, not a browser chatbot with audio.

The current browser/Melo path already solves the hardest immediate media problem better than native PyAudio: browser/WebRTC owns microphone/camera capture and offers echo-cancellation infrastructure. The next bottleneck is not media transport. It is performance coordination.

## Target outcome

After this upgrade, a normal tester should feel:

```text
Blink is present while I talk.
Blink knows whether I am still speaking.
Blink yields when I interrupt.
Blink’s voice is easier to follow.
Blink uses camera honestly and visibly.
Blink remembers project constraints and shows that memory affected the answer.
Blink has a stable style without pretending to be human.
```

## Main architectural decision

Add an explicit **Actor Runtime Layer** between the brain runtime and the browser UI / TTS output.

```text
Browser WebRTC media
  -> VAD/STT/camera frames
  -> Actor Event Ledger
  -> Conversation Floor Controller
  -> Active Listener
  -> Memory/Persona/Performance Compiler
  -> LLM Content Planner
  -> Melo Speech Performance Director
  -> Browser Actor Surface
  -> Blink-Actor-Bench traces
```

## What not to do in this upgrade

- Do not return to native PyAudio as the main UX path.
- Do not build a realistic generated human avatar yet.
- Do not rely on another hidden personality prompt as the main personality fix.
- Do not enable unsafe barge-in on speakers just because browser interruption works in some cases.
- Do not store raw audio, raw images, SDP, tokens, or full prompts in default traces.

## Upgrade principle

Every hidden capability must produce an observable effect:

| Hidden capability | Observable effect |
|---|---|
| VAD/STT | Listening, heard text, turn duration |
| LLM thinking | Thinking state and latency metric |
| Camera | Ready/stale/looking/used-frame status |
| Memory | Used-in-this-reply records and behavior effects |
| Persona | Retrieved behavior references and response shape |
| Speech director | Subtitles, chunking, queue, cancellation |
| Interruption | Accepted/rejected/protected status and stop latency |
