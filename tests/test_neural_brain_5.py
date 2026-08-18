from __future__ import annotations
import json
from pathlib import Path
import pytest
from conftest import analysis
from djai.brain.transition_router import route_transition
from djai.intelligence.neural_ears import predict_beats
from djai.intelligence.neural_policy import rank_duration_values,rendered_judge_prediction
from djai.intelligence.model_registry import load_model_registry
from djai.judge.professional import judge_transition
from djai.models import CompatibilityBreakdown,TransitionMeasurement,TransitionPlan
from djai.perception.world_model import build_track_world_model
from djai.training.neural_heads import train_neural_head,promote_model
from djai.version import DJAI_VERSION,CACHE_SCHEMA_VERSION,PLAN_SCHEMA_VERSION
from djai.candidate_renderer import _capture_render_embeddings

def _head(path,model_id,*,labels=None,feature="energy_delta",weights=None,bias=None):
 p={"schema_version":"1.0","model_id":model_id,"task":model_id,"feature_names":[feature],"feature_mean":{feature:0.0},"feature_scale":{feature:1.0},"metrics":{"validation_accuracy":.8,"validation_examples":60}}
 if labels is not None:p.update(kind="multiclass",labels=labels,weights=weights or [[0.0] for _ in labels],bias=bias or [0.0 for _ in labels])
 else:p.update(kind="scalar",weights=weights or [0.0],bias=0.0 if bias is None else bias)
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(p))
def _registry(tmp,cp,model_id,authority):
 p=tmp/"registry.json";p.write_text(json.dumps({"schema_version":"5.0","models":[{"id":model_id,"task":model_id,"provider":"test","model":"test","version":"1","authority":authority,"weight":.35,"minimum_confidence":.5,"checkpoint":str(cp),"production_approved":authority=="production"}]}));return p

def test_versions():assert (DJAI_VERSION,CACHE_SCHEMA_VERSION,PLAN_SCHEMA_VERSION)==("2.0.0-rc.1","5.0.0","8.0.0")
def test_neural_beats_never_download_implicitly(tmp_path,monkeypatch):
 monkeypatch.setenv("DJAI_MODEL_ALLOW_DOWNLOAD","0");monkeypatch.setenv("DJAI_NEURAL_BEATS","auto");monkeypatch.setenv("DJAI_BEAT_THIS_READY_MARKER",str(tmp_path/"missing"));r=predict_beats(tmp_path/"x.wav",fallback_beats=(0,.5,1),fallback_downbeats=(0,));assert not r.available and not r.used and "bootstrap" in r.reason
def test_shadow_head_cannot_reroute(tmp_path,monkeypatch):
 a=analysis(tmp_path/"a.wav",bpm=124);b=analysis(tmp_path/"b.wav",bpm=125);cp=tmp_path/"f.json";_head(cp,"transition_family_policy",labels=["direct_blend","bridge_preferred"],bias=[-6,6]);monkeypatch.setenv("DJAI_MODEL_REGISTRY",str(_registry(tmp_path,cp,"transition_family_policy","shadow")));r=route_transition(a,b);assert r.family=="direct_blend" and r.neural_policy["authority"]==0
def test_advisory_head_can_reroute(tmp_path,monkeypatch):
 a=analysis(tmp_path/"a.wav",bpm=124);b=analysis(tmp_path/"b.wav",bpm=125);cp=tmp_path/"f.json";_head(cp,"transition_family_policy",labels=["direct_blend","bridge_preferred"],bias=[-6,6]);monkeypatch.setenv("DJAI_MODEL_REGISTRY",str(_registry(tmp_path,cp,"transition_family_policy","advisory")));r=route_transition(a,b);assert r.family=="bridge_preferred" and r.neural_policy["changed_family"]
def test_duration_head_can_reorder(tmp_path,monkeypatch):
 a=analysis(tmp_path/"a.wav",bpm=124);b=analysis(tmp_path/"b.wav",bpm=124);cp=tmp_path/"d.json";_head(cp,"duration_policy",feature="duration_bars",weights=[-.5],bias=10);monkeypatch.setenv("DJAI_MODEL_REGISTRY",str(_registry(tmp_path,cp,"duration_policy","advisory")));order,meta=rank_duration_values(a,b,route_transition(a,b),(32,16,8,4));assert order!=(32,16,8,4) and meta["authority"]>0
def _plan(a,b):
 return TransitionPlan(track_a=a.path,track_b=b.path,exit_time_a=200,entry_time_b=20,transition_bars=16,transition_seconds=30,target_bpm=124,time_stretch_rate_b=1,time_stretch_applied=False,technique="equal_power_blend",score=.8,compatibility=CompatibilityBreakdown(tempo=.9,harmonic=.9,energy=.9,texture=.9,phrase=.9,sections=.9,overall=.9,tempo_difference_percent=0,energy_delta=.05,beat_grid=.9,vocal_overlap=.9,bass_conflict=.9),cue_confidence=.9,intent_score=.8,technique_score=.8)
def test_neural_judge_cannot_override_hard_failure(tmp_path,monkeypatch):
 a=analysis(tmp_path/"a.wav",bpm=124);b=analysis(tmp_path/"b.wav",bpm=124);cp=tmp_path/"j.json";_head(cp,"rendered_dj_judge",feature="render_overall",weights=[8],bias=8);monkeypatch.setenv("DJAI_MODEL_REGISTRY",str(_registry(tmp_path,cp,"rendered_dj_judge","advisory")));p=_plan(a,b);p.measurement=TransitionMeasurement(version="x",sample_rate=22050,duration_seconds=30,overall=.95,musical_overall=.9,hard_failures=["clipping"]);r=judge_transition(p,track_a=a,track_b=b,rejection_path=tmp_path/"none",minimum_score=0);assert r.neural_rendered_authority>0 and not r.feasible and not r.accepted
def test_rendered_judge_consumes_rendered_audio_embedding(tmp_path,monkeypatch):
 a=analysis(tmp_path/"a.wav",bpm=124);b=analysis(tmp_path/"b.wav",bpm=124);cp=tmp_path/"j-embed.json";_head(cp,"rendered_dj_judge",feature="render_embedding_00",weights=[8],bias=0);monkeypatch.setenv("DJAI_MODEL_REGISTRY",str(_registry(tmp_path,cp,"rendered_dj_judge","shadow")));p=_plan(a,b);p.measurement=TransitionMeasurement(version="x",sample_rate=22050,duration_seconds=30,overall=.8,musical_overall=.8);p.metadata={**p.metadata,"rendered_transition_embedding":[1.0]+[0.0]*23};prediction=rendered_judge_prediction(a,b,p);assert prediction.available and prediction.score>.99 and prediction.authority==0

def test_taste_lab_auto_capture_keeps_explicit_disable_override(monkeypatch):
 monkeypatch.setenv("DJAI_CAPTURE_RENDER_EMBEDDINGS","auto");assert _capture_render_embeddings({"mode":"taste_lab"}) is True
 monkeypatch.setenv("DJAI_CAPTURE_RENDER_EMBEDDINGS","0");assert _capture_render_embeddings({"mode":"taste_lab"}) is False

def test_world_model_contract(tmp_path):
 t=analysis(tmp_path/"a.wav",bpm=124);w=build_track_world_model(t);assert w.version=="6.0" and w.path==t.path
def test_training_filters_unreviewed(tmp_path):
 data=tmp_path/"d.jsonl";rows=[{"id":str(i),"rights_cleared":True,"features":{"energy_delta":-1 if i<12 else 1},"label":"direct_blend" if i<12 else "bridge_preferred"} for i in range(24)];rows.append({"id":"bad","rights_cleared":False,"features":{"energy_delta":100},"label":"bridge_preferred"});data.write_text("\n".join(json.dumps(x) for x in rows)+"\n");out=tmp_path/"h.json";train_neural_head("transition_family",data,out,iterations=200);assert json.loads(out.read_text())["metrics"]["examples"]==24
def test_production_promotion_requires_rights(tmp_path):
 cp=tmp_path/"n.json";_head(cp,"next_track_policy",weights=[1],bias=0);reg=_registry(tmp_path,cp,"next_track_policy","shadow");
 with pytest.raises(ValueError):promote_model("next_track_policy",authority="production",registry_path=reg,rights_reviewed=False)
 with pytest.raises(ValueError):promote_model("next_track_policy",authority="production",registry_path=reg,rights_reviewed=True)
 promote_model("next_track_policy",authority="production",registry_path=reg,rights_reviewed=True,governance_approved=True,arena_report_payload={"deployment_ready":True});m=load_model_registry(reg).get("next_track_policy");assert m and m.production_approved


def test_measurement_finalization_marks_only_final_set_signatures(tmp_path):
    import json
    from djai.measurement_store import finalize_measurement_run, measurement_log_offset

    log = tmp_path / "measurements.jsonl"
    old = {
        "schema": "djai-transition-measurement-v1",
        "record_id": "old",
        "accepted": True,
        "rank": 1,
        "measured_score": 0.8,
        "execution_signature": "previous-run",
        "context": {"run": "old"},
    }
    log.write_text(json.dumps(old) + "\n", encoding="utf-8")
    start = measurement_log_offset(log)
    current = [
        {
            "schema": "djai-transition-measurement-v1",
            "record_id": "a",
            "accepted": True,
            "rank": 1,
            "measured_score": 0.91,
            "execution_signature": "superseded-edge",
            "context": {"run": "current"},
        },
        {
            "schema": "djai-transition-measurement-v1",
            "record_id": "b",
            "accepted": True,
            "rank": 1,
            "measured_score": 0.94,
            "execution_signature": "final-edge-1",
            "context": {"run": "current"},
        },
        {
            "schema": "djai-transition-measurement-v1",
            "record_id": "c",
            "accepted": False,
            "rank": 2,
            "measured_score": 0.88,
            "execution_signature": "losing-rehearsal",
            "context": {"run": "current"},
        },
    ]
    with log.open("a", encoding="utf-8") as handle:
        for row in current:
            handle.write(json.dumps(row) + "\n")

    finalize_measurement_run(
        log,
        start_offset=start,
        accepted_signatures={"final-edge-1"},
    )
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert rows[0] == old
    assert [row["accepted"] for row in rows[1:]] == [False, True, False]
    assert rows[1]["context"]["superseded_by_set_replan"] is True
    assert rows[2]["context"]["set_plan_final_acceptance"] is True
