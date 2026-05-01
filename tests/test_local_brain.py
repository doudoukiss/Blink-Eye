from pathlib import Path

from blink.cli.local_brain import (
    LocalBrainMemoryStore,
    build_local_brain_prompt,
)
from blink.transcriptions.language import Language


def test_local_brain_memory_store_remembers_and_overrides_stable_facts(tmp_path):
    store = LocalBrainMemoryStore(path=tmp_path / "memory.json")

    store.remember_from_text("我叫小周，我喜欢机器人。")
    store.remember_from_text("请叫我阿周。")

    facts = store.facts()
    statements = [fact.statement for fact in facts]

    assert "用户名字是 小周" not in statements
    assert "用户名字是 阿周" in statements
    assert "用户喜欢 机器人" in statements


def test_build_local_brain_prompt_includes_identity_and_memory(tmp_path):
    store = LocalBrainMemoryStore(path=tmp_path / "memory.json")
    store.remember_from_text("我叫星野。")

    prompt = build_local_brain_prompt(
        "请用中文回答。",
        language=Language.ZH,
        robot_head_enabled=True,
        memory_store=store,
    )

    assert "你不是外部代班助手，你就是 Blink" in prompt
    assert "当前连接的机器人头是你的物理身体" in prompt
    assert "用户名字是 星野" in prompt


def test_local_brain_memory_store_uses_default_cache_path_when_unspecified():
    store = LocalBrainMemoryStore()

    assert isinstance(store.path, Path)
    assert store.path.name == "memory.json"
