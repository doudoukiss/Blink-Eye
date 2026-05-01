# Blink Bilingual Actor Bench human rating form

Profile tested:

- [ ] browser-zh-melo
- [ ] browser-en-kokoro

Rate 1–5:

| Dimension | Score | Notes |
|---|---:|---|
| State clarity |  |  |
| Felt heard |  |  |
| Voice pacing |  |  |
| Camera grounding |  |  |
| Memory usefulness |  |  |
| Interruption naturalness |  |  |
| Personality consistency |  |  |
| Enjoyment |  |  |
| Not fake-human |  |  |
| Not annoying |  |  |

Hard failures observed:

- [ ] camera claim without camera use
- [ ] hidden camera use
- [ ] unsafe trace payload
- [ ] self-interruption in protected playback
- [ ] stale TTS after interruption
- [ ] memory contradiction after correction
- [ ] profile-specific regression
