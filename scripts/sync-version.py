#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def update_text(path: Path, pattern: str, replacement: str) -> bool:
    source = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not locate version field in {path.relative_to(ROOT)}")
    if source == updated:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def desired_changes() -> list[tuple[Path, str]]:
    return [
        (ROOT / "pyproject.toml", "python project"),
        (ROOT / "src" / "djai" / "version.py", "runtime"),
        (ROOT / "frontend" / "package.json", "frontend package"),
        (ROOT / "frontend" / "package-lock.json", "frontend lock"),
        (ROOT / "frontend" / "public" / "build-info.json", "frontend build info"),
    ]


def sync() -> None:
    update_text(ROOT / "pyproject.toml", r'^(version\s*=\s*")[^"]+("\s*)$', rf'\g<1>{VERSION}\2')
    update_text(ROOT / "src" / "djai" / "version.py", r'^(DJAI_VERSION\s*=\s*")[^"]+("\s*)$', rf'\g<1>{VERSION}\2')

    package_path = ROOT / "frontend" / "package.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["version"] = VERSION
    package_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")

    lock_path = ROOT / "frontend" / "package-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["version"] = VERSION
    if isinstance(lock.get("packages"), dict) and isinstance(lock["packages"].get(""), dict):
        lock["packages"][""]["version"] = VERSION
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    info_path = ROOT / "frontend" / "public" / "build-info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        info = {}
    info.update({
        "product": "SAUND.AI",
        "version": VERSION,
        "build_commit": None,
        "build_time_utc": None,
        "source": "development-placeholder",
    })
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def check() -> list[str]:
    errors: list[str] = []
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    runtime = (ROOT / "src" / "djai" / "version.py").read_text(encoding="utf-8")
    package = json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8"))
    info = json.loads((ROOT / "frontend" / "public" / "build-info.json").read_text(encoding="utf-8"))
    values = {
        "VERSION": VERSION,
        "pyproject": re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE).group(1),
        "runtime": re.search(r'^DJAI_VERSION\s*=\s*"([^"]+)"', runtime, re.MULTILINE).group(1),
        "frontend": str(package.get("version")),
        "frontend_lock": str(lock.get("version")),
        "frontend_lock_root": str(lock.get("packages", {}).get("", {}).get("version")),
        "frontend_build_info": str(info.get("version")),
    }
    for label, value in values.items():
        if value != VERSION:
            errors.append(f"{label}={value} (expected {VERSION})")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep VERSION authoritative across SAUND.AI build metadata.")
    parser.add_argument("--check", action="store_true", help="Fail instead of editing when a version copy is stale.")
    args = parser.parse_args()
    if not args.check:
        sync()
    errors = check()
    if errors:
        for error in errors:
            print(f"VERSION MISMATCH: {error}")
        return 1
    print(f"SAUND.AI version metadata is synchronized at {VERSION}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
