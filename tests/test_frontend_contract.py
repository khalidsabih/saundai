from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def test_frontend_has_expected_source_files() -> None:
    expected = [
        "app/page.tsx",
        "app/layout.tsx",
        "app/globals.css",
        "components/SimpleExperience.tsx",
        "components/AccountDialog.tsx",
        "components/HeroSoundtrack.tsx",
        "components/FinishedSetPlayer.tsx",
        "components/LivePlayer.tsx",
        "components/ProWorkspace.tsx",
        "components/TransitionInspector.tsx",
        "components/AudioPreview.tsx",
        "lib/api.ts",
        "lib/session.tsx",
        "lib/types.ts",
        "public/saund-mark.svg",
        "public/media/saund-hero-desktop.mp4",
        "public/media/saund-hero-mobile.mp4",
        "public/media/saund-hero-poster.jpg",
        "public/media/saund-hero-poster-mobile.jpg",
        "public/media/saund-hero-intro.mp3",
        "public/media/saund-hero-bed.mp3",
    ]
    for relative in expected:
        path = FRONTEND / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 0



def test_opening_gate_uses_cinematic_background_without_controls() -> None:
    simple = (FRONTEND / "components/SimpleExperience.tsx").read_text(encoding="utf-8")
    css = (FRONTEND / "app/globals.css").read_text(encoding="utf-8")
    assert 'className="saund-intro__video"' in simple
    assert 'src="/media/saund-hero-desktop.mp4"' in simple
    assert 'src="/media/saund-hero-mobile.mp4"' in simple
    assert 'poster="/media/saund-hero-poster.jpg"' in simple
    assert "autoPlay" in simple and "muted" in simple and "loop" in simple and "playsInline" in simple
    assert " controls" not in simple
    assert "prefers-reduced-motion: reduce" in css
    assert '.saund-intro__video' in css




def test_opening_gate_has_mutable_soundtrack_and_click_handoff() -> None:
    simple = (FRONTEND / "components/SimpleExperience.tsx").read_text(encoding="utf-8")
    soundtrack = (FRONTEND / "components/HeroSoundtrack.tsx").read_text(encoding="utf-8")
    css = (FRONTEND / "app/globals.css").read_text(encoding="utf-8")
    assert "HeroSoundtrack" in simple
    assert 'src="/media/saund-hero-intro.mp3"' in soundtrack
    assert 'src="/media/saund-hero-bed.mp3"' in soundtrack
    assert "NATURAL_CROSSFADE_SECONDS" in soundtrack
    assert "Sound on" in soundtrack
    assert "Mute" in soundtrack
    assert "loop" in soundtrack
    assert "setSoundtrackActive(false)" in simple
    enter_block = simple.split("function enterCreate()", 1)[1].split("async function ingest", 1)[0]
    assert "soundtrackRef.current?.start()" not in enter_block
    assert "setSoundtrackActive(false)" in enter_block
    assert 'className={`saund-intro autonomous-intro ${introExiting ? "is-exiting" : ""}`}' in simple
    assert 'Start a session' in simple
    assert 'disabled={!introReady || introExiting}' in simple
    assert 'introProgressRef.current < 1' in simple
    assert 'ref={introProgressBarRef}' in simple
    assert 'onAnimationEnd={() => setIntroReady(true)}' not in simple
    assert '{introReady ? "Enter the system" : "Opening"}' in simple
    assert 'saund-intro-locked' in simple
    assert 'height: 100dvh' in css
    assert 'overflow: hidden' in css
    assert '@keyframes saund-word-left' in css
    assert '@keyframes saund-word-right' in css
    assert '@keyframes saund-tag-up' in css
    assert 'phase={!soundtrackActive ? "off" : (entered || introExiting ? "workspace" : "intro")}' in simple

def test_frontend_dependencies_are_exactly_pinned() -> None:
    package = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
    assert package["name"] == "saund-ai-console"
    for group in ("dependencies", "devDependencies"):
        for name, version in package[group].items():
            assert not version.startswith(("^", "~", ">", "<", "*")), (name, version)


def test_frontend_uses_no_store_and_same_origin_api() -> None:
    api_source = (FRONTEND / "lib/api.ts").read_text(encoding="utf-8")
    config = (FRONTEND / "next.config.ts").read_text(encoding="utf-8")
    assert 'cache: "no-store"' in api_source
    assert "readPlanFromJob" in api_source
    assert "EventSource" in api_source
    assert 'NEXT_PUBLIC_SAUND_API_URL ?? ""' in api_source
    assert "localhost:8000" not in api_source
    assert 'SAUND_INTERNAL_API_URL ?? "http://api:8000"' in config
    assert 'source: "/api/:path*"' in config


def test_frontend_defaults_to_create_upload_and_playback() -> None:
    page = (FRONTEND / "app/page.tsx").read_text(encoding="utf-8")
    simple = (FRONTEND / "components/SimpleExperience.tsx").read_text(encoding="utf-8")
    intent = (FRONTEND / "components/set/SetIntentPanel.tsx").read_text(encoding="utf-8")
    finished = (FRONTEND / "components/FinishedSetPlayer.tsx").read_text(encoding="utf-8")
    api = (FRONTEND / "lib/api.ts").read_text(encoding="utf-8")
    assert 'useState<"create" | "studio">("create")' in page
    assert "createAutomaticSet" in simple
    assert "Build my set" in intent
    assert "Tell SAUND how the night should feel." in simple
    assert "Perform Live" in finished
    assert "/api/sets/automatic" in api
    assert "/api/auth/me" in api
    assert "/api/auth/login" in api
    assert "/api/auth/register" in api
    assert "/api/auth/logout" in api
    assert "/api/account/preferences" in api
    assert "/api/live/stream" in api
    assert "/api/feedback" in api


def test_create_allows_guest_setup_but_gates_build_on_account() -> None:
    simple = (FRONTEND / "components/SimpleExperience.tsx").read_text(encoding="utf-8")
    crate = (FRONTEND / "components/music/CrateSelector.tsx").read_text(encoding="utf-8")
    intent = (FRONTEND / "components/set/SetIntentPanel.tsx").read_text(encoding="utf-8")
    account_dialog = (FRONTEND / "components/AccountDialog.tsx").read_text(encoding="utf-8")
    session = (FRONTEND / "lib/session.tsx").read_text(encoding="utf-8")
    assert 'className={`session-dropzone ${dragging ? "is-dragging" : ""}`}' in crate
    assert 'if (!account)' in simple
    assert 'setAccountDialogReason("build")' in simple
    assert "Build my set" in intent
    assert 'Log in to start.' in account_dialog
    assert 'Your tracks and direction stay exactly as they are.' in account_dialog
    assert 'getAuthSession' in session
    assert 'signIn' in session and 'createAccount' in session and 'signOut' in session


def test_create_and_studio_share_session_and_brain_contract() -> None:
    session = (FRONTEND / "lib/session.tsx").read_text(encoding="utf-8")
    simple = (FRONTEND / "components/SimpleExperience.tsx").read_text(encoding="utf-8")
    studio = (FRONTEND / "components/ProWorkspace.tsx").read_text(encoding="utf-8")
    assert 'const key = "saund-listener-id"' in session
    assert 'const legacyKey = "djai-listener-id"' in session
    assert "useSaundSession" in simple
    assert "useSaundSession" in studio
    assert "createAutomaticSet" in simple
    assert "automatic_delivery: false" in studio
    assert "Current Studio state" in studio
    assert "Move brain" in studio


def test_frontend_exposes_real_autonomous_dj_performance_state() -> None:
    studio = (FRONTEND / "components/ProWorkspace.tsx").read_text(encoding="utf-8")
    plan = (FRONTEND / "components/PlanView.tsx").read_text(encoding="utf-8")
    inspector = (FRONTEND / "components/TransitionInspector.tsx").read_text(encoding="utf-8")
    live = (FRONTEND / "components/LivePlayer.tsx").read_text(encoding="utf-8")
    lab = (FRONTEND / "components/TrainingLab.tsx").read_text(encoding="utf-8")
    create = (FRONTEND / "components/SimpleExperience.tsx").read_text(encoding="utf-8")

    assert "Current Studio state" in studio
    assert "Performance timeline" in plan
    assert "SAUND is performing the handoff" in inspector
    assert "Tempo handoff" in inspector
    assert "Musical ownership" in inspector
    assert "tempo_release" in live
    assert "Deck B owns tempo" in live
    assert "WHAT SAUND KNOWS" in lab
    assert "WHAT SAUND IS LEARNING" in lab
    assert "WHAT CAN AFFECT LIVE" in lab
    assert "DJ Move Brain" in lab
    assert "Complete DJ move" in inspector
    assert "Move brain" in studio
    assert "dj_move_brain" in (FRONTEND / "lib/types.ts").read_text(encoding="utf-8")
    assert "saund.ai.intro.seen.v1" in create
    assert "rehearses the hard moments" in create


def test_release_ui_hides_engine_dashboard_details() -> None:
    studio = (FRONTEND / "components/ProWorkspace.tsx").read_text(encoding="utf-8")
    inspector = (FRONTEND / "components/TransitionInspector.tsx").read_text(encoding="utf-8")
    plan = (FRONTEND / "components/PlanView.tsx").read_text(encoding="utf-8")
    assert "hero-signal" not in studio
    assert "override" not in studio.lower()
    assert "Compatibility" not in inspector
    assert "Metric" not in inspector
    assert "EnergyCurve" not in plan
    assert "Set movement" in plan
    assert "Rendered before selection" in inspector


def test_release_ui_has_accessible_selection_and_reorder_controls() -> None:
    crate = (FRONTEND / "components/music/CrateSelector.tsx").read_text(encoding="utf-8")
    library = (FRONTEND / "components/TrackLibrary.tsx").read_text(encoding="utf-8")
    builder = (FRONTEND / "components/SetBuilder.tsx").read_text(encoding="utf-8")
    assert "aria-pressed={selected}" in crate
    assert "aria-pressed={active}" in library
    assert "Move ${track.name} up" in builder
    assert "Move ${track.name} down" in builder


def test_feedback_and_live_playback_handle_failures() -> None:
    feedback = (FRONTEND / "components/FeedbackButtons.tsx").read_text(encoding="utf-8")
    live = (FRONTEND / "components/LivePlayer.tsx").read_text(encoding="utf-8")
    assert "Couldn’t save your feedback" in feedback
    assert "setError" in feedback
    assert "manifestError" in live
    assert "onError" in live


def test_mobile_studio_navigation_is_four_items() -> None:
    studio = (FRONTEND / "components/ProWorkspace.tsx").read_text(encoding="utf-8")
    css = (FRONTEND / "app/globals.css").read_text(encoding="utf-8")
    assert studio.count('href: "#') == 4
    assert "repeat(4, minmax(0, 1fr))" in css


def test_docker_frontend_requires_reproducible_lockfile() -> None:
    dockerfile = (FRONTEND / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY package.json package-lock.json ./" in dockerfile
    assert "npm ci" in dockerfile
    assert "npm install" not in dockerfile


def test_training_lab_exposes_provenance_aware_dataset_builder_and_shadow_training() -> None:
    lab = (FRONTEND / "components/TrainingLab.tsx").read_text(encoding="utf-8")
    api = (FRONTEND / "lib/api.ts").read_text(encoding="utf-8")
    assert "BOOTSTRAP DATA / EXTERNAL SOURCES" in lab
    assert "Freeze bootstrap corpus" in lab
    assert "Train bootstrap shadows" in lab
    assert "New taste heads start at zero authority" in lab
    assert "Hard safety" in lab
    assert "Persistent bootstrap benchmark" in lab
    assert "top-3" in lab
    assert "latest_benchmark" in (FRONTEND / "lib/types.ts").read_text(encoding="utf-8")
    assert "/api/dataset-builder/status" in api
    assert "/api/dataset-builder/build" in api
    assert "/api/dataset-builder/train" in api


def test_account_foundation_two_exposes_private_project_history_and_studio_gate() -> None:
    api = (FRONTEND / "lib/api.ts").read_text(encoding="utf-8")
    account = (FRONTEND / "components/AccountDialog.tsx").read_text(encoding="utf-8")
    studio = (FRONTEND / "components/ProWorkspace.tsx").read_text(encoding="utf-8")
    css = (FRONTEND / "app/globals.css").read_text(encoding="utf-8")
    assert 'credentials: "include"' in api
    assert 'withCredentials: true' in api
    assert '/api/account/projects' in api
    assert 'Recent sessions' in account
    assert 'getAccountProjects' in account
    assert 'requestProtectedAction' in studio
    assert 'setAccountDialogReason("build")' in studio
    assert '.account-dialog__projects' in css


def test_performance_plan_surfaces_release_evidence_strip() -> None:
    plan = (FRONTEND / "components/PlanView.tsx").read_text(encoding="utf-8")
    css = (FRONTEND / "app/globals.css").read_text(encoding="utf-8")
    assert 'aria-label="Performance proof"' in plan
    assert "AI Headphones" in plan
    assert "search explored" in plan
    assert "Crate Memory" in plan
    assert "release_benchmark" in plan
    assert ".performance-proof" in css


def test_release_candidate_status_is_visible_but_non_blocking_in_studio() -> None:
    studio = (FRONTEND / "components/ProWorkspace.tsx").read_text(encoding="utf-8")
    api = (FRONTEND / "lib/api.ts").read_text(encoding="utf-8")
    types = (FRONTEND / "lib/types.ts").read_text(encoding="utf-8")
    assert "getReleaseReadiness" in studio
    assert 'getReleaseReadiness().catch(() => null)' in studio
    assert 'release.authority.planner_mode === "move_authoritative"' in studio
    assert "Executive route" in studio
    assert "Shadow route" in studio
    assert '"/api/release/status"' in api
    assert "ReleaseReadiness" in types
