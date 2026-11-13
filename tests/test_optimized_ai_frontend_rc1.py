from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_optimized_frontend_keeps_known_good_rc1_transport_exact() -> None:
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


def test_create_analysis_is_background_work_not_a_navigation_gate() -> None:
    create = (FRONTEND / "components/SimpleExperience.tsx").read_text(encoding="utf-8")
    assert "const canStartSession = selectedPaths.length >= 2" in create
    assignment = create.split("const canStartSession", 1)[1].split(";", 1)[0]
    assert "selectedReady" not in assignment
    assert "preparingSelectedCount" not in assignment
    assert "failedSelectedCount === 0" in assignment
    assert "<CrateSelector" in create
    assert "<SetIntentPanel" in create
    assert "<BuildProgress" in create


def test_studio_prioritizes_timeline_and_collapses_setup_after_planning() -> None:
    studio = (FRONTEND / "components/ProWorkspace.tsx").read_text(encoding="utf-8")
    css = (FRONTEND / "app/globals.css").read_text(encoding="utf-8")
    assert "setupExpanded" in studio
    assert "setSetupExpanded(false)" in studio
    assert "Edit music & direction" in studio
    assert "studio-intro--optimized" in studio
    assert ".studio-intro--optimized" in css
    assert ".studio-setup-toggle" in css


def test_transition_timeline_is_direct_audition_surface() -> None:
    studio = (FRONTEND / "components/ProWorkspace.tsx").read_text(encoding="utf-8")
    plan = (FRONTEND / "components/PlanView.tsx").read_text(encoding="utf-8")
    inspector = (FRONTEND / "components/TransitionInspector.tsx").read_text(encoding="utf-8")
    css = (FRONTEND / "app/globals.css").read_text(encoding="utf-8")
    assert "activateTransition" in studio
    assert "queuedPreviewIndex" in studio
    assert "previewAudioRef" in studio
    assert "Playing the rendered handoff you selected." in studio
    assert 'className="studio-preview-dock"' in studio
    assert "playingTransition" in plan
    assert "preparingTransition" in plan
    assert "Listen" in plan and "Playing" in plan
    assert 'id="transition-detail"' in inspector
    assert "Why this transition?" in studio
    assert ".studio-preview-dock" in css
    assert ".studio-transition-detail" in css


def test_deep_performance_evidence_is_progressively_disclosed() -> None:
    plan = (FRONTEND / "components/PlanView.tsx").read_text(encoding="utf-8")
    assert '<details className="performance-evidence">' in plan
    assert "Performance evidence" in plan
    assert 'aria-label="Performance proof"' in plan
