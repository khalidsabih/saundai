#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from djai.release_candidate import certification_manifest, release_candidate_status  # noqa: E402


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or certify the SAUND.AI 2.0 release candidate.")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Print operational or strict release-candidate readiness")
    status.add_argument("--strict", action="store_true")
    status.add_argument("--suite", default=None)

    certify = sub.add_parser("certify", help="Write a release certification manifest; fails unless every strict gate passes")
    certify.add_argument("--suite", default=None)
    certify.add_argument("--output", default="dist/release-certification.json")

    args = parser.parse_args()
    if args.command == "status":
        payload = release_candidate_status(ROOT, strict=args.strict, benchmark_suite=args.suite)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.strict:
            return 0 if payload.get("release_certifiable") else 2
        return 0 if payload.get("operational_ready") else 2

    payload = certification_manifest(ROOT, benchmark_suite=args.suite)
    if not payload.get("certified"):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    destination = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    _write_atomic(destination, payload)
    print(json.dumps({"certified": True, "output": str(destination), "certification_id": payload["certification_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
