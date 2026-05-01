import blink
from blink import version as blink_version
from blink._version import __version__
from blink.project_identity import (
    PROJECT_IDENTITY,
    cache_dir,
    cli_command,
    import_banner_env_name,
    install_requirement,
    local_env_name,
    pip_install_command,
)


def test_blink_version_uses_checked_in_version():
    assert blink_version() == __version__


def test_project_identity_exposes_blink_canonical_fields():
    assert PROJECT_IDENTITY.display_name == "Blink"
    assert PROJECT_IDENTITY.distribution_name == "blink-ai"
    assert PROJECT_IDENTITY.import_namespace == "blink"
    assert PROJECT_IDENTITY.cli_prefix == "blink-local"
    assert PROJECT_IDENTITY.env_prefix == "BLINK"
    assert PROJECT_IDENTITY.homepage_url == "https://github.com/blink-ai/Blink"
    assert PROJECT_IDENTITY.documentation_url == "https://github.com/blink-ai/Blink/tree/main/docs"
    assert PROJECT_IDENTITY.source_url == "https://github.com/blink-ai/Blink"
    assert PROJECT_IDENTITY.issues_url == "https://github.com/blink-ai/Blink/issues"
    assert PROJECT_IDENTITY.changelog_url == "https://github.com/blink-ai/Blink/blob/main/CHANGELOG.md"


def test_project_identity_install_helpers_expose_blink_names():
    assert install_requirement() == "blink-ai"
    assert install_requirement("runner") == "blink-ai[runner]"
    assert pip_install_command("runner") == "pip install blink-ai[runner]"
    assert cli_command("chat") == "blink-local-chat"
    assert local_env_name("LANGUAGE") == "BLINK_LOCAL_LANGUAGE"
    assert import_banner_env_name() == "BLINK_IMPORT_BANNER"


def test_cache_dir_prefers_blink_path(monkeypatch, tmp_path):
    monkeypatch.setattr("blink.project_identity.Path.home", lambda: tmp_path)

    assert cache_dir("piper") == tmp_path / ".cache" / "blink" / "piper"


def test_cache_dir_is_unchanged_by_unrelated_old_paths(monkeypatch, tmp_path):
    monkeypatch.setattr("blink.project_identity.Path.home", lambda: tmp_path)
    old_cache = tmp_path / ".cache" / "old-runtime" / "piper"
    old_cache.mkdir(parents=True)

    canonical_cache = tmp_path / ".cache" / "blink" / "piper"

    assert cache_dir("piper") == canonical_cache
    assert old_cache.exists()
