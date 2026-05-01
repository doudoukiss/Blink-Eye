import pytest

from blink.web import smallwebrtc_ui


def test_smallwebrtc_ui_dependency_guard_raises_clean_import_error(monkeypatch):
    monkeypatch.setattr(smallwebrtc_ui, "_SMALLWEBRTC_UI_IMPORT_ERROR", ModuleNotFoundError("fastapi"))

    with pytest.raises(ImportError, match=r"blink-ai\[runner\]"):
        smallwebrtc_ui.require_smallwebrtc_ui()


def test_mount_smallwebrtc_ui_uses_stable_client_path(monkeypatch, tmp_path):
    mounted = {}

    class DummyApp:
        def mount(self, path, frontend):
            mounted["path"] = path
            mounted["frontend"] = frontend

    class DummyStaticFiles:
        def __init__(self, *, directory, html):
            self.directory = directory
            self.html = html

    monkeypatch.setattr(smallwebrtc_ui, "_SMALLWEBRTC_UI_IMPORT_ERROR", None)
    monkeypatch.setattr(smallwebrtc_ui, "StaticFiles", DummyStaticFiles)
    monkeypatch.setattr(smallwebrtc_ui, "SMALLWEBRTC_UI_DIST_DIR", tmp_path)
    tmp_path.joinpath("index.html").write_text("<html></html>")

    smallwebrtc_ui.mount_smallwebrtc_ui(DummyApp())

    assert mounted["path"] == "/client"
    assert mounted["frontend"].directory == tmp_path
    assert mounted["frontend"].html is True


@pytest.mark.asyncio
async def test_smallwebrtc_static_files_disable_caching(monkeypatch, tmp_path):
    class DummyResponse:
        def __init__(self):
            self.headers = {}

    class DummyStaticFiles:
        def __init__(self, *, directory, html):
            self.directory = directory
            self.html = html

        async def get_response(self, path, scope):
            return DummyResponse()

    monkeypatch.setattr(smallwebrtc_ui, "_SMALLWEBRTC_UI_IMPORT_ERROR", None)
    monkeypatch.setattr(smallwebrtc_ui, "StaticFiles", DummyStaticFiles)
    monkeypatch.setattr(smallwebrtc_ui, "SMALLWEBRTC_UI_DIST_DIR", tmp_path)
    tmp_path.joinpath("index.html").write_text("<html></html>")

    static_files = smallwebrtc_ui.create_smallwebrtc_static_files()
    response = await static_files.get_response("index.html", {})

    assert response.headers["Cache-Control"] == "no-store, max-age=0"


def test_smallwebrtc_ui_dependency_guard_raises_clean_error_when_bundle_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(smallwebrtc_ui, "_SMALLWEBRTC_UI_IMPORT_ERROR", None)
    monkeypatch.setattr(smallwebrtc_ui, "SMALLWEBRTC_UI_DIST_DIR", tmp_path)
    monkeypatch.setattr(smallwebrtc_ui, "SMALLWEBRTC_UI_SOURCE_DIR", tmp_path / "missing-source")

    with pytest.raises(ImportError, match=r"browser UI bundle is missing"):
        smallwebrtc_ui.require_smallwebrtc_ui()


@pytest.mark.asyncio
async def test_create_smallwebrtc_root_redirect_points_at_client(monkeypatch):
    class DummyRedirectResponse:
        def __init__(self, *, url):
            self.url = url

    monkeypatch.setattr(smallwebrtc_ui, "_SMALLWEBRTC_UI_IMPORT_ERROR", None)
    monkeypatch.setattr(smallwebrtc_ui, "RedirectResponse", DummyRedirectResponse)

    handler = smallwebrtc_ui.create_smallwebrtc_root_redirect()
    response = await handler()

    assert response.url == "/client/"


def test_blink_browser_index_references_media_autoplay_helper():
    index_html = (smallwebrtc_ui.SMALLWEBRTC_UI_SOURCE_DIR / "index.html").read_text(
        encoding="utf-8"
    )

    assert "blink-media-autoplay.js" in index_html


def test_blink_browser_media_autoplay_helper_exists():
    helper_script = smallwebrtc_ui.SMALLWEBRTC_UI_SOURCE_DIR / "assets" / "blink-media-autoplay.js"

    assert helper_script.is_file()


def test_blink_expression_panel_shows_active_listening_without_new_media_capture():
    panel_script = (
        smallwebrtc_ui.SMALLWEBRTC_UI_SOURCE_DIR / "assets" / "blink-expression-panel.js"
    ).read_text(encoding="utf-8")

    assert "active_listening" in panel_script
    assert "/api/runtime/actor-state" in panel_script
    assert "/api/runtime/actor-events" in panel_script
    assert "actor_surface_v2_enabled" in panel_script
    assert "renderActorSurface" in panel_script
    assert "placePanel(panel)" in panel_script
    assert 'position: "relative"' in panel_script
    assert "if (state.collapsed)" in panel_script
    assert "if (!state.collapsed) {\n        refresh();" in panel_script
    assert "fullRefreshMinMs" in panel_script
    assert "refreshState.fullInFlight" in panel_script
    assert "refreshState.performanceInFlight" in panel_script
    assert "actorMediaText" in panel_script
    assert 'setTextLine(performanceBody, "media", actorMediaText(actorObject, performance));' in panel_script
    assert "profileBadge" in panel_script
    assert "Debug timeline" in panel_script
    assert "Used memory/persona" in panel_script
    assert "activeListeningStatusText" in panel_script
    assert "partialTranscriptText" in panel_script
    assert "lastPartialTranscript" in panel_script
    assert "no partials from STT" in panel_script
    assert "camera_presence" in panel_script
    assert "cameraPresenceStatusText" in panel_script
    assert "current_answer_used_vision" in panel_script
    assert "grounding_mode" in panel_script
    assert "getUserMedia" not in panel_script
    assert "createOffer" not in panel_script


def test_blink_operator_workbench_shows_memory_persona_performance():
    panel_script = (
        smallwebrtc_ui.SMALLWEBRTC_UI_SOURCE_DIR / "assets" / "blink-operator-workbench.js"
    ).read_text(encoding="utf-8")

    assert "memory_persona_performance" in panel_script
    assert "collapsed: true" in panel_script
    assert 'position: "relative"' in panel_script
    assert "if (state.collapsed && options.force !== true)" in panel_script
    assert "refresh({ force: true, includeEvidence: true, includeStatic: true });" in panel_script
    assert "used_in_current_reply" in panel_script
    assert "Used in this reply" in panel_script
    assert "Behavior effect" in panel_script
    assert "renderUsedMemoryRef" in panel_script
    assert "renderPersonaReference" in panel_script
    assert "getUserMedia" not in panel_script
    assert "createOffer" not in panel_script


def test_blink_browser_source_workspace_exists():
    source_index = smallwebrtc_ui.SMALLWEBRTC_UI_SOURCE_DIR / "index.html"
    package_json = smallwebrtc_ui.SMALLWEBRTC_UI_SOURCE_DIR.parent / "package.json"
    build_script = smallwebrtc_ui.SMALLWEBRTC_UI_SOURCE_DIR.parent / "build.mjs"

    assert source_index.is_file()
    assert package_json.is_file()
    assert build_script.is_file()


def test_blink_browser_manifest_is_source_and_blink_branded():
    manifest_path = smallwebrtc_ui.SMALLWEBRTC_UI_SOURCE_DIR / "manifest.webmanifest"
    manifest_text = manifest_path.read_text(encoding="utf-8")

    assert manifest_path.is_file()
    assert '"name": "Blink Voice"' in manifest_text
    assert '"short_name": "Blink"' in manifest_text


def test_blink_browser_index_references_manifest_and_app_name():
    index_html = (smallwebrtc_ui.SMALLWEBRTC_UI_SOURCE_DIR / "index.html").read_text(
        encoding="utf-8"
    )

    assert "manifest.webmanifest" in index_html
    assert 'application-name" content="Blink Voice"' in index_html
