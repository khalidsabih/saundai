from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_safe_ai_upgrade_keeps_rc1_transport_and_docker_contract_exact() -> None:
    # This UI upgrade is deliberately presentation-only. These hashes are the
    # exact RC1 baseline that was known to upload reliably for the user.
    api = (ROOT / "frontend" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert 'credentials: "include"' in api
    assert '"/api/tracks/upload"' in api
    assert '"/api/tracks/upload-intents"' in api
    assert "EventSource" in api
    next_config = (ROOT / "frontend" / "next.config.ts").read_text(encoding="utf-8")
    assert 'output: "standalone"' in next_config
    assert 'source: "/api/:path*"' in next_config
    assert 'destination: `${internalApi}/api/:path*`' in next_config
    assert 'X-Content-Type-Options' in next_config
    assert _sha256("compose.yaml") == "762df2e22a1c21cf095ed040b92bba11357ca7c3d3c9d05a9ad785cf166bd92a"


def test_safe_ai_upgrade_has_visible_listen_analyse_plan_play_live_story() -> None:
    ai = (FRONTEND / "components/AIExperience.tsx").read_text(encoding="utf-8")
    create = (FRONTEND / "components/SimpleExperience.tsx").read_text(encoding="utf-8")
    finished = (FRONTEND / "components/FinishedSetPlayer.tsx").read_text(encoding="utf-8")
    live = (FRONTEND / "components/LivePlayer.tsx").read_text(encoding="utf-8")
    css = (FRONTEND / "app/globals.css").read_text(encoding="utf-8")

    for label in ("Listening", "Analysing", "Planning", "Playing", "Live"):
        assert label in ai
    assert "SAUND is analysing" in create
    assert "SAUND is auditioning the handoffs" in create
    assert "SAUND finished the performance" in finished
    assert "SAUND is live" in live
    assert ".ai-experience__rail" in css
    assert "prefers-reduced-motion: reduce" in css


def test_create_keeps_analysis_visible_without_blocking_direction() -> None:
    create = (FRONTEND / "components/SimpleExperience.tsx").read_text(encoding="utf-8")
    assert "canProceedToDirection" in create
    assert 'disabled={!canProceedToDirection}' in create
    assert "analysing in the background" in create
    assert "selectedPaths.length >= 2" in create
    assignment = create.split("const canProceedToDirection", 1)[1].split(";", 1)[0]
    assert "readiness" not in assignment


def test_visible_ai_component_has_no_network_or_backend_dependency() -> None:
    ai = (FRONTEND / "components/AIExperience.tsx").read_text(encoding="utf-8")
    assert "@/lib/api" not in ai
    assert "fetch(" not in ai
    assert "EventSource" not in ai
    assert "WebSocket" not in ai
