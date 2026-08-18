#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from djai.intelligence.governance import audit_model_governance  # noqa: E402
from djai.training.provenance import audit_dataset_registry  # noqa: E402


def main() -> int:
    product = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    frontend = json.loads((ROOT / "frontend/package.json").read_text(encoding="utf-8"))["version"]
    blockers: list[str] = []
    if not (product == pyproject == frontend):
        blockers.append(f"version mismatch: VERSION={product}, python={pyproject}, frontend={frontend}")

    dataset_path = ROOT / "datasets" / "registry.json"
    catalog_path = ROOT / "datasets" / "source-catalog.json"
    if not dataset_path.exists():
        blockers.append("datasets/registry.json is missing")
        data = {"ready": False, "blockers": [{"reason": "missing_registry"}]}
    else:
        data = audit_dataset_registry(dataset_path, lane="production")
        if not data.get("ready", False):
            blockers.append("production dataset provenance audit is blocked")
    if not catalog_path.exists():
        blockers.append("datasets/source-catalog.json is missing")

    governance = audit_model_governance(ROOT / "config" / "model-registry.json")
    if not governance.ready:
        blockers.extend(f"model governance: {item}" for item in governance.blockers)

    report = {
        "product": "SAUND.AI",
        "version": product,
        "ready": not blockers,
        "blockers": blockers,
        "dataset_provenance": data,
        "model_governance": governance.to_dict(),
        "policy": {
            "learned_models_default_deny": True,
            "hard_audio_safety_non_overridable": True,
            "external_training_data_default_deny": True,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
