from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_optimized_frontend_keeps_known_good_rc1_transport_exact() -> None:
    assert _sha256("frontend/lib/api.ts") == "b5255b01733771d460cb4d82bde744df021ae4d6c72b526a85344ccefc7f20bd"
    assert _sha256("frontend/next.config.ts") == "6e2f3d1a80e62e07e8757df97e11d5e450e8f78f705a6d7d995c6b9b28db556d"
    assert _sha256("compose.yaml") == "762df2e22a1c21cf095ed040b92bba11357ca7c3d3c9d05a9ad785cf166bd92a"


def test_create_analysis_is_background_work_not_a_navigation_gate() -> None:
    create = (FRONTEND / "components/SimpleExperience.tsx").read_text(encoding="utf-8")
    assert "const canProceedToDirection = selectedPaths.length >= 2" in create
    assert "analysing in the background" in create
    assert "You can keep shaping the set while that work continues." in create
    assert "Keep moving while SAUND analyses." in create
    assert "analysing first" not in create


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
