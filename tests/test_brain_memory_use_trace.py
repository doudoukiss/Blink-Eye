import json

from blink.brain.context import BrainContextCompiler, BrainContextTask
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.identity import base_brain_system_prompt, load_default_agent_blocks
from blink.brain.memory_v2 import (
    BrainMemoryUseTrace,
    BrainMemoryUseTraceRef,
    MemoryContinuityTrace,
    build_memory_continuity_trace,
    build_memory_use_trace,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language


def _session(client_id: str = "use-trace"):
    return resolve_brain_session_ids(runtime_kind="browser", client_id=client_id)


def _trace_ref(session_ids, claim_id: str = "claim_abc") -> BrainMemoryUseTraceRef:
    return BrainMemoryUseTraceRef(
        memory_id=f"memory_claim:user:{session_ids.user_id}:{claim_id}",
        display_kind="preference",
        title="coffee",
        section_key="relevant_continuity",
        used_reason="selected_for_relevant_continuity",
        safe_provenance_label="Remembered from your explicit preference.",
        reason_codes=("source:context_selection",),
    )


def test_memory_use_trace_roundtrip_is_stable_and_json_safe():
    session_ids = _session("roundtrip")
    trace = build_memory_use_trace(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
        thread_id=session_ids.thread_id,
        task="reply",
        selected_section_names=("relevant_continuity", "persona_expression"),
        refs=(_trace_ref(session_ids),),
        reason_codes=("test",),
    )

    hydrated = BrainMemoryUseTrace.from_dict(trace.as_dict())

    assert hydrated == trace
    assert hydrated.as_dict() == trace.as_dict()
    assert json.loads(json.dumps(trace.as_dict(), sort_keys=True)) == trace.as_dict()


def test_empty_memory_use_trace_is_deterministic():
    session_ids = _session("empty")
    first = build_memory_use_trace(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
        thread_id=session_ids.thread_id,
        task="reply",
    )
    second = build_memory_use_trace(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
        thread_id=session_ids.thread_id,
        task="reply",
    )

    assert first == second
    assert first.refs == ()
    assert "memory_use_trace_empty" in first.reason_codes
    assert first.summary == "No user-visible memories influenced this packet."


def test_store_appends_and_replays_memory_use_trace(tmp_path):
    store = BrainStore(path=tmp_path / "source.db")
    replayed = BrainStore(path=tmp_path / "replayed.db")
    session_ids = _session("store")
    trace = build_memory_use_trace(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
        thread_id=session_ids.thread_id,
        task="reply",
        selected_section_names=("relevant_continuity",),
        refs=(_trace_ref(session_ids),),
    )

    persisted = store.append_memory_use_trace(
        trace=trace,
        session_id=session_ids.session_id,
        source="test",
        ts="2026-04-23T01:02:03+00:00",
    )
    events = store.brain_events_since(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        after_id=0,
        limit=8,
    )
    for event in events:
        replayed.import_brain_event(event)

    assert persisted.created_at == "2026-04-23T01:02:03+00:00"
    assert (
        store.latest_memory_use_trace(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        )
        == persisted
    )
    assert (
        replayed.latest_memory_use_trace(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        )
        == persisted
    )


def test_store_appends_and_replays_memory_continuity_trace(tmp_path):
    store = BrainStore(path=tmp_path / "source.db")
    replayed = BrainStore(path=tmp_path / "replayed.db")
    session_ids = _session("continuity-store")
    use_trace = build_memory_use_trace(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
        thread_id=session_ids.thread_id,
        task="reply",
        selected_section_names=("relevant_continuity",),
        refs=(_trace_ref(session_ids),),
    )
    trace = build_memory_continuity_trace(
        memory_use_trace=use_trace,
        session_id=session_ids.session_id,
        profile="browser-en-kokoro",
        language="en",
        hidden_counts={"suppressed": 1},
    )

    persisted = store.append_memory_continuity_trace(
        trace=trace,
        session_id=session_ids.session_id,
        source="test",
        ts="2026-04-23T01:02:03+00:00",
    )
    events = store.brain_events_since(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        after_id=0,
        limit=8,
    )
    for event in events:
        replayed.import_brain_event(event)

    assert persisted.created_at == "2026-04-23T01:02:03+00:00"
    assert MemoryContinuityTrace.from_dict(persisted.as_dict()) == persisted
    assert (
        store.latest_memory_continuity_trace(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        )
        == persisted
    )
    assert (
        replayed.latest_memory_continuity_trace(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        )
        == persisted
    )
    encoded = json.dumps(persisted.as_dict(), sort_keys=True).lower()
    for banned in ("raw_json", "system_prompt", "source_event_ids", "audio_bytes", "image_bytes"):
        assert banned not in encoded


def test_compiler_memory_use_trace_sidecar_is_pure_and_scoped(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("compiler")
    store.ensure_default_blocks(load_default_agent_blocks())
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.like",
        subject="coffee",
        value={"value": "coffee"},
        rendered_text="user likes coffee",
        confidence=0.9,
        singleton=False,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    compiler = BrainContextCompiler(
        store=store,
        session_resolver=lambda: session_ids,
        language=Language.EN,
        base_prompt=base_brain_system_prompt(Language.EN),
        context_surface_builder=BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
        ),
    )

    packet = compiler.compile_packet(
        latest_user_text="Please use my coffee preference.",
        task=BrainContextTask.REPLY,
    )
    events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=4,
    )

    assert packet.memory_use_trace is not None
    assert packet.memory_use_trace.user_id == session_ids.user_id
    assert packet.memory_use_trace.created_at == ""
    assert any(
        ref.memory_id.startswith("memory_claim:user:") for ref in packet.memory_use_trace.refs
    )
    assert all("source_event" not in str(ref.as_dict()) for ref in packet.memory_use_trace.refs)
    assert not any(event.event_type == "memory.use.traced" for event in events)


def test_reply_packet_includes_user_profile_memory_and_trace(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("profile-memory")
    store.ensure_default_blocks(load_default_agent_blocks())
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.name",
        subject="阿周",
        value={"value": "阿周"},
        rendered_text="用户名字是 阿周",
        confidence=0.92,
        singleton=True,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    compiler = BrainContextCompiler(
        store=store,
        session_resolver=lambda: session_ids,
        language=Language.ZH,
        base_prompt=base_brain_system_prompt(Language.ZH),
        context_surface_builder=BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.ZH,
        ),
    )

    packet = compiler.compile_packet(
        latest_user_text="你还记得我是谁吗？",
        task=BrainContextTask.REPLY,
    )

    section = packet.selected_context.section("user_profile")
    assert section is not None
    assert "阿周" in section.content
    assert packet.memory_use_trace is not None
    assert "user_profile" in packet.memory_use_trace.selected_section_names
    profile_refs = [
        ref
        for ref in packet.memory_use_trace.refs
        if ref.memory_id.startswith(f"memory_claim:user:{session_ids.user_id}:")
        and "阿周" in ref.title
    ]
    assert profile_refs
    assert profile_refs[0].display_kind == "profile"
    assert profile_refs[0].section_key in {"active_continuity", "user_profile"}
    assert profile_refs[0].used_reason in {
        "selected_for_active_continuity",
        "selected_for_user_profile",
    }


def test_memory_use_trace_serialization_does_not_expose_raw_provenance():
    session_ids = _session("leak-free")
    trace = build_memory_use_trace(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
        thread_id=session_ids.thread_id,
        task="reply",
        refs=(_trace_ref(session_ids),),
        reason_codes=("private_scratchpad_filtered",),
    )
    encoded = json.dumps(trace.as_dict(), sort_keys=True)

    assert "source_refs" not in encoded
    assert "source_event_ids" not in encoded
    assert "db_path" not in encoded
    assert "private scratchpad" not in encoded.lower()
    assert "raw_json" not in encoded
