# Dogfooding protocol

Run matched 5-minute sessions on both primary paths:

```bash
./scripts/run-local-browser-melo.sh
./scripts/run-local-browser-kokoro-en.sh
```

Each session should include:

1. long user monologue with constraints;
2. visual question requiring Moondream;
3. interruption or correction;
4. memory callback from a prior turn;
5. style/persona stress question;
6. no-vision fallback case;
7. end-of-session rating.

Ratings should be structured, not only free-form:

- state clarity: 1-5
- felt heard: 1-5
- voice pacing: 1-5
- interruption naturalness: 1-5
- camera honesty: 1-5
- memory usefulness: 1-5
- persona consistency: 1-5
- enjoyment: 1-5
- not fake-human: 1-5

Every low score should map to a failure label so it can be clustered and used for policy improvement.
