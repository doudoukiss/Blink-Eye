from blink.cli.local_brain_audit import main as blink_local_brain_audit_main
from blink.cli.local_brain_chat import main as blink_local_brain_chat_main
from blink.cli.local_brain_reflect import main as blink_local_brain_reflect_main
from blink.cli.local_brain_shell import main as blink_local_brain_shell_main
from blink.cli.local_browser import main as blink_local_browser_main
from blink.cli.local_chat import main as blink_local_chat_main
from blink.cli.memory_persona_ingest import main as blink_memory_persona_ingest_main


def test_blink_chat_cli_entrypoint_is_callable():
    assert callable(blink_local_chat_main)


def test_blink_brain_chat_cli_entrypoint_is_callable():
    assert callable(blink_local_brain_chat_main)


def test_blink_brain_audit_cli_entrypoint_is_callable():
    assert callable(blink_local_brain_audit_main)


def test_blink_brain_reflect_cli_entrypoint_is_callable():
    assert callable(blink_local_brain_reflect_main)


def test_blink_brain_shell_cli_entrypoint_is_callable():
    assert callable(blink_local_brain_shell_main)


def test_blink_browser_cli_entrypoint_is_callable():
    assert callable(blink_local_browser_main)


def test_blink_memory_persona_ingest_cli_entrypoint_is_callable():
    assert callable(blink_memory_persona_ingest_main)
