from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("zip") is None or shutil.which("tar") is None,
    reason="release packaging requires zip and tar",
)
def test_package_release_replaces_existing_archive(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    archive = tmp_path / "release.zip"

    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("SAUND_AI/stale-entry.txt", "must not survive")

    environment = {**__import__("os").environ, "SAUND_SKIP_RELEASE_GATE": "1"}
    subprocess.run(
        [str(project_root / "scripts" / "package-release.sh"), str(archive)],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    with zipfile.ZipFile(archive) as packaged:
        names = set(packaged.namelist())

    assert "SAUND_AI/stale-entry.txt" not in names
    assert "SAUND_AI/VERSION" in names
    assert "SAUND_AI/src/djai/measurement.py" in names
    assert "SAUND_AI/src/djai/performance/engine.py" in names
    assert "SAUND_AI/src/djai/rehearsal_cache.py" in names
    assert "SAUND_AI/src/djai/brain/director.py" in names
    assert "SAUND_AI/DJAI_6.0.0_RELEASE_NOTES.md" in names
    assert "SAUND_AI/DJAI_6.0.0_VERIFICATION.md" in names
    assert "SAUND_AI/SAUND_AI_1.0.0_RELEASE_NOTES.md" in names
    assert "SAUND_AI/SAUND_AI_1.0.0_PRODUCTION.md" in names
    assert "SAUND_AI/datasets/registry.json" in names
    assert "SAUND_AI/docs/SAUND_HERO_AUDIO.md" in names
    assert "SAUND_AI/frontend/public/media/saund-hero-intro.mp3" in names
    assert "SAUND_AI/frontend/public/media/saund-hero-bed.mp3" in names
    assert "SAUND_AI/config/model-registry.json" in names
    assert "SAUND_AI/src/djai/training/dataset_builder.py" in names
    assert "SAUND_AI/src/djai/training/source_catalog.py" in names
    assert "SAUND_AI/datasets/source-catalog.json" in names
    assert "SAUND_AI/SAUND_AI_6.0.0_RC10_DATASET_BUILDER.md" in names
    assert "SAUND_AI/config/scoring-model.json" not in names
    assert "SAUND_AI/.env" not in names
    assert "SAUND_AI/accounts/accounts.sqlite3" not in names
    assert not any(name.endswith((".orig", ".rej")) for name in names)
    assert not any(name.startswith("SAUND_AI/datasets/downloads/") for name in names)
    assert not any(name.startswith("SAUND_AI/datasets/external/") for name in names)
    assert not any(name.startswith("SAUND_AI/datasets/bootstrap-corpora/") for name in names)
    assert not any(name.startswith("SAUND_AI/config/bootstrap-models/") for name in names)
    assert not any(name.startswith("SAUND_AI/config/neural/") and name not in {"SAUND_AI/config/neural/", "SAUND_AI/config/neural/.gitkeep"} for name in names)
    assert not any(".egg-info/" in name for name in names)
    assert not any("__pycache__/" in name for name in names)
    assert not any(".pytest_cache/" in name for name in names)


def test_release_gate_requires_hero_intro_rights_confirmation() -> None:
    project_root = Path(__file__).resolve().parents[1]
    gate = (project_root / "scripts" / "release-gate.sh").read_text(encoding="utf-8")
    assert "SAUND_HERO_INTRO_RIGHTS_CONFIRMED" in gate
    assert "hero intro distribution rights must be confirmed" in gate
    assert "dataset-audit --lane production" in gate
