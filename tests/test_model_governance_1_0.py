import json
from pathlib import Path

from djai.intelligence.governance import audit_model_governance


def test_shadow_only_registry_is_governance_ready(tmp_path: Path):
    registry = tmp_path / "model-registry.json"
    registry.write_text(json.dumps({"schema_version":"5.0","models":[{"id":"x","task":"x","provider":"x","model":"x","version":"1","authority":"shadow"}]}))
    report = audit_model_governance(registry)
    assert report.ready
    assert report.production_models == 0


def test_unapproved_production_model_is_blocked(tmp_path: Path):
    checkpoint = tmp_path / "head.json"
    checkpoint.write_text("{}")
    registry = tmp_path / "model-registry.json"
    registry.write_text(json.dumps({"schema_version":"5.0","models":[{
        "id":"x","task":"x","provider":"x","model":"x","version":"1","authority":"production",
        "checkpoint":str(checkpoint),"production_approved":False
    }]}))
    report = audit_model_governance(registry)
    assert not report.ready
    assert report.blockers
