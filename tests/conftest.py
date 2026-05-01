import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "brain_fast",
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
settings.register_profile(
    "brain_stateful",
    max_examples=8,
    stateful_step_count=6,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
settings.load_profile("brain_fast")


@pytest.fixture(autouse=True)
def isolate_local_brain_storage(monkeypatch, tmp_path):
    """Keep Blink brain persistence isolated per test."""
    monkeypatch.setenv("BLINK_LOCAL_BRAIN_DB_PATH", str(tmp_path / "brain.db"))
    monkeypatch.setenv("BLINK_LOCAL_BRAIN_MEMORY_PATH", str(tmp_path / "legacy-memory.json"))
