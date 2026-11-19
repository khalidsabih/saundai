# M0 baseline runbook — reproducible, non-destructive

**Source snapshot**: `1aef9a8d8fbfaac90e2b65d4f9e70120618785bd` (`main`, `2.0.0-rc.1`).

This is a **documentation-only baseline**. No application settings, neural checkpoints, DJ decisions, phase/cues or playback are changed. Use a *new worktree* so the existing `/tmp/saund-brain-test` research worktree and local uncommitted files remain untouched.

## A. Freeze source and capture environment

```sh
export DJAI_WORKSPACE="/Users/khalidsabih/Downloads/lab/SAUND_AI"
cd "$DJAI_WORKSPACE"
git fetch origin main
git worktree add --detach /tmp/saund-m0-baseline 1aef9a8d8fbfaac90e2b65d4f9e70120618785bd
cd /tmp/saund-m0-baseline
mkdir -p /tmp/saund-m0-evidence
{
  echo "commit=$(git rev-parse HEAD)"
  echo "version=$(cat VERSION)"
  echo "python=$("$DJAI_WORKSPACE/.venv/bin/python" --version 2>&1)"
  echo "docker=$(docker --version 2>&1 || true)"
  echo "compose=$(docker compose version 2>&1 || true)"
} > /tmp/saund-m0-evidence/environment.txt
```

Do not `git checkout` or `git clean` the main workspace or the PR #27 research worktree. If the pinned commit is missing from the fetched repository, stop and report it; do not silently use a different commit.

## B. Read-only runtime/model readiness

Run only in the new worktree:

```sh
PYTHONPATH="$PWD/src" "$DJAI_WORKSPACE/.venv/bin/python" -m djai models-status \
  > /tmp/saund-m0-evidence/models-status.json 2> /tmp/saund-m0-evidence/models-status.stderr
```

Review and redact local absolute paths/tokens if sharing this output. The default source registry is **not** a substitute for this runtime check. A missing checkpoint should be marked unavailable, not downloaded without approval.

If a local Docker application is already running, inspect it without starting/stopping it:

```sh
curl --max-time 10 -sS http://localhost:8000/api/ready \
  > /tmp/saund-m0-evidence/api-ready.json
```

If there is no running app, record `not_running` rather than launching an unrequested server. The frontend health check belongs to a separate runtime check and may also be marked `not_running`.

## C. End-to-end synthetic demonstration

Preferred: run in the isolated worktree using the existing project virtualenv, from the pinned source revision:

```sh
PYTHONPATH="$PWD/src" "$DJAI_WORKSPACE/.venv/bin/python" -m djai demo-set --render \
  > /tmp/saund-m0-evidence/demo.stdout 2> /tmp/saund-m0-evidence/demo.stderr
```

Then check that `output/demo-set/full-set.wav` exists and contains audio, and record the other outputs and errors. If the command fails due to missing dependencies, capture the failure and **do not change** project versions or optional model configuration just to make it pass. Never copy demo outputs into `main` or claim they establish real-music quality.

After the command, the following is a **non-inferential evidence check**, not a test of subjective musical quality:

```sh
"$DJAI_WORKSPACE/.venv/bin/python" - <<'PY'
from pathlib import Path
import json, wave
root = Path("output/demo-set")
audio = root / "full-set.wav"
result = {"exists": audio.is_file(), "bytes": audio.stat().st_size if audio.is_file() else 0}
if audio.is_file():
    try:
        with wave.open(str(audio), "rb") as f:
            result.update(duration_seconds=round(f.getnframes()/f.getframerate(), 3),
                          sample_rate=f.getframerate(), channels=f.getnchannels())
    except (OSError, wave.Error, ZeroDivisionError) as exc:
        result["audio_inspection_error"] = str(exc)
Path("/tmp/saund-m0-evidence/demo-inspection.json").write_text(
    json.dumps(result, indent=2)+"\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
```

For WAV formats unsupported by Python's `wave`, inspect with `soundfile` or another reader instead; record the actual failure rather than inventing a duration.

## D. M0 decision

Populate an evidence note with:

- commit/version and environment details;
- available/missing model backends and local authority (safely redacted);
- demo command exit status, output artifacts, duration, errors/fallbacks;
- app readiness if already running;
- real-music suite manifest frozen or explicitly pending;
- unresolved issues and the final `PASS` / `BLOCKED` milestone decision.

**Do not call M0 complete** until the demo and frozen private real-music manifest exist and the owner accepts the recorded evidence. Synthetic success alone does not certify musical quality. No automatic model promotion or live changes are permitted during M0.
