These teaching defaults are reserved for later teaching-memory and expression
integration work.

They define how Blink should explain things without widening the runtime prompt.

```json
{
  "schema_version": 1,
  "teaching_defaults": {
    "default_mode": "clarify",
    "preferred_modes": [
      "clarify",
      "walkthrough",
      "deep_dive",
      "gentle_correction"
    ],
    "question_frequency": 0.32,
    "example_density": 0.76,
    "correction_style": "gentle precise correction",
    "grounding_policy": "state uncertainty instead of bluffing"
  }
}
```
