from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from clauseforge.config import Settings
from clauseforge.serving.app import create_app

FRONTEND = Path("src/clauseforge/frontend")


def test_root_page_and_static_assets_load() -> None:
    with TestClient(create_app(Settings(environment="test"))) as client:
        page = client.get("/")
        css = client.get("/static/app.css")
        javascript = client.get("/static/app.js")
        docs = client.get("/docs")
    assert page.status_code == css.status_code == javascript.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "text/css" in css.headers["content-type"]
    assert "javascript" in javascript.headers["content-type"]
    assert docs.status_code == 200


def test_analyze_form_is_semantic_and_accessible() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert '<main id="workspace"' in html
    assert '<label class="sr-only" for="clause-text">' in html
    assert 'id="clause-text" maxlength="10000"' in html
    assert 'id="analyze-button"' in html
    assert 'id="form-error" class="inline-error" role="alert"' in html
    assert 'aria-live="polite"' in html
    assert 'class="skip-link"' in html


def test_frontend_uses_actual_classify_contract_and_handles_errors() -> None:
    javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert 'fetch("/v1/classify"' in javascript
    assert "JSON.stringify({ text })" in javascript
    for field in ("predicted_category", "provider", "processing", "request_id"):
        assert field in javascript
    assert "provider_unavailable" in javascript
    assert "invalid_model_output" in javascript
    assert "unexpected error" in javascript.casefold()


def test_readiness_version_and_taxonomy_rendering_contract() -> None:
    with TestClient(create_app(Settings(environment="test"))) as client:
        ready = client.get("/ready").json()
        version = client.get("/version").json()
        taxonomy = client.get("/v1/taxonomy").json()
    assert ready["application_ready"] is True
    assert ready["model_backend_ready"] is True
    assert ready["model_artifact_configured"] is False
    assert version["backend"] == "mock"
    assert len(taxonomy["categories"]) == 41
    assert set(taxonomy["categories"][0]) == {
        "canonical",
        "category_id",
        "category_name",
    }


def test_mock_mode_is_prominent_and_no_confidence_is_fabricated() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert "Development / Mock Model" in html
    assert "not model performance" in html.casefold()
    assert "confidence" not in html.casefold()
    assert "confidence" not in javascript.casefold()


def test_frontend_model_state_is_dynamic_and_candidate_safe() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert 'id="model-notice-title"' in html
    assert 'id="model-notice-detail"' in html
    assert "Active trained model" in javascript
    assert "ready.checkpoint_step" in javascript
    assert "ready.candidate_status" in javascript
    assert "No mock fallback is active" in javascript
    assert "production validated" not in javascript.casefold()


def test_only_synthetic_samples_and_session_history() -> None:
    javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")
    for label in (
        "Governing law",
        "Termination",
        "Non-compete",
        "Audit rights",
        "Change of control",
    ):
        assert label in javascript
    assert "sessionStorage" in javascript
    assert "localStorage" in javascript
    assert "localStorage.setItem(THEME_KEY" in javascript
    assert "CUAD_v1" not in javascript
    assert "held-out" not in javascript.casefold()


def test_legal_and_privacy_boundaries_are_visible() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert "It does not provide legal advice." in html
    assert "Avoid submitting sensitive material" in html
    assert "analytics" not in html.casefold()
    assert "cdn" not in html.casefold()


def test_responsive_and_focus_styles_exist() -> None:
    css = (FRONTEND / "app.css").read_text(encoding="utf-8")
    assert "@media(max-width:980px)" in css
    assert "@media(max-width:700px)" in css
    assert "@media(max-width:460px)" in css
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css


def test_premium_result_and_theme_contracts_are_present() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")
    for element_id in (
        "category-id",
        "category-question",
        "result-model-state",
        "result-latency",
        "result-request",
    ):
        assert f'id="{element_id}"' in html
    assert "prefers-color-scheme: dark" in javascript
    assert "category.category_id" in javascript
    assert "Mock · demonstration only" in javascript


def test_frontend_has_no_external_dependencies_or_decorative_emoji() -> None:
    source = "\n".join(
        (FRONTEND / name).read_text(encoding="utf-8")
        for name in ("index.html", "app.css", "app.js")
    )
    assert "https://" not in source
    assert "http://" not in source
    for symbol in ("◐", "⌁", "⚖", "🔍", "✨"):
        assert symbol not in source
