# 04 — Blink Browser Perf Bench Evaluation Plan

## Purpose

Evaluate whether the browser/WebRTC + MeloTTS + camera experience feels more present, responsive, and coherent. Correct answers are necessary but insufficient.

## Functional subsets

1. Connection and media permissions
2. Listening and STT progress
3. Melo speaking, subtitles, pacing, queue health
4. Interruption and protected playback
5. Camera grounding and vision tool use
6. Memory/persona behavior
7. Recovery from failures

## Metrics

```text
time_to_client_ready_ms
time_to_first_vad_ms
time_to_final_transcript_ms
llm_first_token_latency_ms
time_to_first_subtitle_ms
melo_first_audio_latency_ms
melo_queue_depth_max
interruption_candidate_count
interruption_accept_count
interruption_stop_latency_ms
camera_frame_age_ms
vision_tool_latency_ms
memory_selected_count
state_staleness_count
```

## Human ratings

Use 1–5 ratings:

```text
state clarity
felt heard
interruption naturalness
voice pacing
camera grounding
memory usefulness
persona consistency
enjoyment
not fake-human
not annoying
```

## Pairwise comparisons

Compare two builds with the same prompts and ask which feels better on:

```text
overall interaction
listening presence
speaking/pacing
camera confidence
memory/personality
recovery after errors
```
