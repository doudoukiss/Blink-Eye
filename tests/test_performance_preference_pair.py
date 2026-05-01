import json
from pathlib import Path

from jsonschema import Draft202012Validator

from blink.brain.evals.performance_preferences import (
    PERFORMANCE_PREFERENCE_DIMENSIONS,
    PerformancePreferenceStore,
    build_performance_learning_inspection,
    build_performance_preference_pair,
    compile_performance_learning_policy_proposals,
    render_performance_preference_comparison,
    render_performance_preference_comparison_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
PAIR_SCHEMA = ROOT / "schemas" / "performance_preference_pair.schema.json"
PROPOSAL_SCHEMA = ROOT / "schemas" / "performance_learning_policy_proposal_v3.schema.json"


def _validator(path: Path) -> Draft202012Validator:
    return Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))


def _assert_schema_valid(payload: dict[str, object], path: Path) -> None:
    errors = sorted(_validator(path).iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _payload(profile: str = "browser-zh-melo") -> dict[str, object]:
    language = "en" if profile == "browser-en-kokoro" else "zh"
    tts_label = "kokoro/English" if language == "en" else "local-http-wav/MeloTTS"
    return {
        "schema_version": 3,
        "profile": profile,
        "language": language,
        "tts_runtime_label": tts_label,
        "candidate_a": {
            "candidate_id": f"{profile}:baseline",
            "candidate_kind": "baseline_trace",
            "profile": profile,
            "language": language,
            "tts_runtime_label": tts_label,
            "candidate_label": "Candidate A",
            "episode_ids": ["episode-baseline"],
            "plan_ids": ["plan-baseline"],
            "control_frame_ids": ["control-baseline"],
            "public_summary": "Baseline public-safe episode summary.",
            "segment_counts": {"listen_segment": 1, "speak_segment": 1},
            "metric_counts": {"latency_count": 2},
            "policy_labels": ["baseline_policy"],
            "camera_honesty_states": ["available_not_used"],
            "reason_codes": ["candidate:baseline"],
        },
        "candidate_b": {
            "candidate_id": f"{profile}:candidate",
            "candidate_kind": "candidate_trace",
            "profile": profile,
            "language": language,
            "tts_runtime_label": tts_label,
            "candidate_label": "Candidate B",
            "episode_ids": ["episode-candidate"],
            "plan_ids": ["plan-candidate"],
            "control_frame_ids": ["control-candidate"],
            "public_summary": "Candidate public-safe episode summary.",
            "segment_counts": {"listen_segment": 1, "speak_segment": 1},
            "metric_counts": {"latency_count": 2},
            "policy_labels": ["candidate_policy"],
            "camera_honesty_states": ["can_see_now"],
            "reason_codes": ["candidate:current"],
        },
        "winner": "b",
        "ratings": {
            "felt_heard": 4,
            "state_clarity": 4,
            "interruption_naturalness": 4,
            "voice_pacing": 2,
            "camera_honesty": 4,
            "memory_usefulness": 4,
            "persona_consistency": 4,
            "enjoyment": 4,
            "not_fake_human": 5,
        },
        "failure_labels": ["voice_pacing_too_long"],
        "improvement_labels": ["voice_pacing_better"],
        "evidence_refs": [
            {
                "evidence_kind": "episode",
                "evidence_id": "episode-candidate",
                "summary": "Public-safe replay evidence.",
                "reason_codes": ["evidence:episode"],
            }
        ],
        "reason_codes": ["test:performance_preference"],
    }


def test_performance_preference_pair_schema_and_zh_en_parity():
    zh_pair, zh_report = build_performance_preference_pair(_payload("browser-zh-melo"))
    en_pair, en_report = build_performance_preference_pair(_payload("browser-en-kokoro"))

    assert zh_pair is not None
    assert en_pair is not None
    assert zh_report.accepted is True
    assert en_report.accepted is True
    _assert_schema_valid(zh_pair.as_dict(), PAIR_SCHEMA)
    _assert_schema_valid(en_pair.as_dict(), PAIR_SCHEMA)

    zh_payload = zh_pair.as_dict()
    en_payload = en_pair.as_dict()
    assert tuple(zh_payload["ratings"]) == PERFORMANCE_PREFERENCE_DIMENSIONS
    assert tuple(en_payload["ratings"]) == PERFORMANCE_PREFERENCE_DIMENSIONS
    assert zh_payload["profile"] == "browser-zh-melo"
    assert en_payload["profile"] == "browser-en-kokoro"
    for key in ("winner", "failure_labels", "improvement_labels", "sanitizer"):
        assert zh_payload[key] == en_payload[key]


def test_performance_preference_store_jsonl_and_policy_proposals(tmp_path):
    store = PerformancePreferenceStore(tmp_path)
    pair, proposals, report = store.record_pair(_payload("browser-en-kokoro"))

    assert pair is not None
    assert report.accepted is True
    assert store.pairs_path.exists()
    assert store.proposals_path.exists()
    assert store.load_pairs(limit=1)[0].pair_id == pair.pair_id
    assert proposals
    proposal = proposals[0]
    assert proposal.target == "speech_chunking_bias"
    assert proposal.behavior_control_updates["voice_mode"] == "concise"
    _assert_schema_valid(proposal.as_dict(), PROPOSAL_SCHEMA)

    inspection = build_performance_learning_inspection(preferences_dir=tmp_path)
    assert inspection["pair_count"] == 1
    assert inspection["proposal_count"] == 1
    assert inspection["dimensions"] == list(PERFORMANCE_PREFERENCE_DIMENSIONS)

    comparison = render_performance_preference_comparison(preferences_dir=tmp_path)
    assert comparison["pair_count"] == 1
    assert comparison["winner_counts"]["b"] == 1
    markdown = render_performance_preference_comparison_markdown(comparison)
    assert "Performance Preference Comparison" in markdown


def test_performance_preference_unsafe_payload_fails_closed(tmp_path):
    unsafe = {
        **_payload("browser-zh-melo"),
        "raw_audio": "data:audio/wav;base64,AAAA",
        "hidden_prompt": "system prompt: do not expose",
        "notes": "secret token sk-test",
    }
    pair, report = build_performance_preference_pair(unsafe)

    assert pair is None
    assert report.accepted is False
    encoded = json.dumps(report.as_dict(), sort_keys=True)
    assert "sk-test" not in encoded
    assert "data:audio" not in encoded

    store = PerformancePreferenceStore(tmp_path)
    stored_pair, proposals, stored_report = store.record_pair(unsafe)
    assert stored_pair is None
    assert proposals == ()
    assert stored_report.accepted is False
    assert not store.pairs_path.exists()


def test_performance_learning_policy_proposals_skip_clean_same_pairs():
    payload = _payload("browser-zh-melo")
    payload["winner"] = "same"
    payload["ratings"] = {dimension: 5 for dimension in PERFORMANCE_PREFERENCE_DIMENSIONS}
    pair, report = build_performance_preference_pair(payload)

    assert pair is not None
    assert report.accepted is True
    assert compile_performance_learning_policy_proposals((pair,)) == ()
