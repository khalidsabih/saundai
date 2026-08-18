from __future__ import annotations

from dataclasses import replace

from conftest import analysis
from djai.brain import dj_move
from djai.models import BarFeature
from djai.music.timeline import TIMELINE_VERSION, build_musical_timeline
from djai.perception.world_model import build_track_world_model
from djai.planner import plan_transition_alternatives
from djai.set_models import SetPlan
from djai.version import MUSICAL_TIMELINE_SCHEMA_VERSION


def _bar(index: int, *, section: str, energy: float, vocal: float = 0.12) -> BarFeature:
    strong = index in {0, 4, 8, 12, 16, 20}
    return BarFeature(
        bar_index=index,
        start_seconds=index * 2.0,
        end_seconds=(index + 1) * 2.0,
        section_label=section,  # type: ignore[arg-type]
        energy=energy,
        rhythm=0.76 if section == "drop" else 0.58,
        brightness=0.52,
        percussive_activity=0.68,
        vocal_probability=vocal,
        bass_activity=0.58,
        harmonic_tension=0.72 if section == "build" else 0.42,
        boundary_strength=0.86 if strong else 0.20,
        phrase_strength=0.88 if strong else 0.28,
        beat_grid_confidence=0.94,
        local_bpm=120.0,
    )


def _timeline_track(tmp_path, name: str = "timeline.wav"):
    track = analysis(tmp_path / name, bpm=120.0, energy=0.62)
    track.bar_features = []
    for index in range(24):
        if index < 4:
            section, energy = "intro", 0.34
        elif index < 12:
            section, energy = "build", 0.46 + 0.04 * (index - 4)
        elif index < 20:
            section, energy = "drop", 0.91
        else:
            section, energy = "groove", 0.61
        track.bar_features.append(_bar(index, section=section, energy=energy, vocal=0.22 if section == "drop" else 0.10))
    track.duration_seconds = 48.0
    track.beats_seconds = [index * 0.5 for index in range(96)]
    track.downbeats_seconds = [index * 2.0 for index in range(24)]
    track.entry_candidates = [replace(track.entry_candidates[0], time_seconds=8.0, bar_index=4, section_label="build")]
    track.exit_candidates = [replace(track.exit_candidates[0], time_seconds=40.0, bar_index=20, section_label="groove")]
    return track


def test_timeline_is_the_authoritative_beat_bar_phrase_section_model(tmp_path) -> None:
    track = _timeline_track(tmp_path)
    timeline = build_musical_timeline(track)
    assert timeline.version == TIMELINE_VERSION == MUSICAL_TIMELINE_SCHEMA_VERSION
    assert len(timeline.beats) == 96
    assert len(timeline.bars) == 24
    assert timeline.phrases
    assert timeline.sections
    assert any(item.kind == "pending_drop_payoff" and item.payoff_bar == 12 for item in timeline.commitments)
    assert timeline.phrase_for_bar(12).role == "drop"
    assert timeline.phrase_for_time(24.5).phrase_id == timeline.phrase_for_bar(12).phrase_id
    assert timeline.section_for_bar(12).label == "drop"


def test_legacy_move_brain_and_world_model_share_the_same_phrases(tmp_path) -> None:
    track = _timeline_track(tmp_path)
    canonical = build_musical_timeline(track)
    legacy_move = dj_move.build_musical_timeline(track)
    world = build_track_world_model(track, phrase_bars=8)
    assert legacy_move is canonical
    assert world.structural_summary["canonical_timeline"] is True
    assert world.structural_summary["timeline_fingerprint"] == canonical.fingerprint
    assert [(item.start_bar, item.end_bar, item.role) for item in world.phrases] == [
        (item.start_bar, item.end_bar, item.role) for item in canonical.phrases
    ]


def test_transition_plan_records_canonical_phrase_identity(tmp_path) -> None:
    a = _timeline_track(tmp_path, "a.wav")
    b = _timeline_track(tmp_path, "b.wav")
    plan = plan_transition_alternatives(a, b, force=True, tempo_policy="safe")[0]
    assert plan.metadata["musical_timeline_version"] == TIMELINE_VERSION
    assert str(plan.metadata["exit_phrase_id"]).startswith("phrase_")
    assert str(plan.metadata["entry_phrase_id"]).startswith("phrase_")
    assert plan.metadata["exit_phrase_role"] == build_musical_timeline(a).phrase_for_bar(
        int(plan.metadata["exit_bar_index"])
    ).role
    assert plan.metadata["deck_performance"]["musical_timeline_version"] == TIMELINE_VERSION


def test_set_plan_exposes_canonical_timeline_to_studio_payload(tmp_path) -> None:
    track = _timeline_track(tmp_path)
    plan = SetPlan(
        plan_id="timeline-test",
        tracks=[track],
        order=[0],
        transitions=[],
        baseline_order=[0],
        baseline_score=1.0,
        optimized_score=1.0,
        estimated_duration_seconds=track.duration_seconds,
        source="test",
    )
    payload = plan.to_dict(include_analysis=False)
    timeline = payload["tracks"][0]["musical_timeline"]
    assert timeline["version"] == TIMELINE_VERSION
    assert timeline["fingerprint"] == build_musical_timeline(track).fingerprint
    assert timeline["phrases"]
    assert "song_role" in timeline["phrases"][0]
    assert "song_form_confidence" in timeline["phrases"][0]
    assert timeline["sections"]
    assert "beats" not in timeline and "bars" not in timeline


def test_song_form_recovers_first_verse_from_coarse_intro_labels(tmp_path) -> None:
    track = analysis(tmp_path / "song-form.wav", bpm=120.0, energy=0.62)
    bars = []
    for index in range(20):
        strong = index in {0, 4, 8, 12, 16}
        if index < 4:
            section, energy, vocal = "intro", 0.28, 0.05
        elif index < 8:
            # Deliberately keep the coarse analyzer label as intro.  The
            # canonical song-form layer should notice the vocal/body onset.
            section, energy, vocal = "intro", 0.56, 0.66
        elif index < 12:
            section, energy, vocal = "groove", 0.78, 0.72
        elif index < 16:
            section, energy, vocal = "groove", 0.58, 0.18
        else:
            section, energy, vocal = "outro", 0.42, 0.10
        bars.append(
            BarFeature(
                bar_index=index,
                start_seconds=index * 2.0,
                end_seconds=(index + 1) * 2.0,
                section_label=section,  # type: ignore[arg-type]
                energy=energy,
                rhythm=0.72,
                brightness=0.52,
                percussive_activity=0.68,
                vocal_probability=vocal,
                bass_activity=0.56,
                harmonic_tension=0.44,
                boundary_strength=0.88 if strong else 0.20,
                phrase_strength=0.90 if strong else 0.28,
                beat_grid_confidence=0.95,
                local_bpm=120.0,
            )
        )
    track.bar_features = bars
    track.duration_seconds = 40.0
    timeline = build_musical_timeline(track)

    assert timeline.phrase_for_bar(0).song_role == "intro"
    first_body = timeline.phrase_for_bar(4)
    assert first_body is not None
    assert first_body.song_role == "verse"
    assert first_body.song_role_index == 1
    assert any(
        item.kind == "first_body_payoff"
        and item.target_bar == first_body.start_bar
        and item.protect_until_bar == first_body.end_bar
        for item in timeline.commitments
    )
