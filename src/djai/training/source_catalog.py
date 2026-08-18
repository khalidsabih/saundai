from __future__ import annotations

import hashlib
import json
import shutil
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SourceLane = Literal["research", "production"]
AssetKind = Literal["metadata", "annotations", "audio", "archive"]

CATALOG_SCHEMA_VERSION = "1.0"
_USER_AGENT = "SAUND.AI-Dataset-Builder/1.0"


@dataclass(frozen=True, slots=True)
class DatasetAsset:
    name: str
    url: str | None
    kind: AssetKind
    checksum: str | None = None
    checksum_algorithm: str | None = None
    size_note: str = ""
    manual: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DatasetSource:
    source_id: str
    name: str
    lane: SourceLane
    purpose: tuple[str, ...]
    homepage: str
    importer: str
    contains_audio: bool
    rights_status: str
    default_asset: str | None = None
    assets: tuple[DatasetAsset, ...] = field(default_factory=tuple)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["id"] = payload.pop("source_id")
        return payload


# External sources intentionally default to the research lane.  A source being
# public or Creative-Commons-described is not, by itself, a production rights
# review for SAUND.AI.  Production promotion remains default-deny.
_SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        source_id="edm-cue",
        name="EDM-CUE / M-DJCUE",
        lane="research",
        purpose=("expert cue points", "phrase alignment", "cue prior"),
        homepage="https://ismir2024program.ismir.net/poster_19.html",
        importer="edm-cue",
        contains_audio=False,
        rights_status="annotations/model are public; review the source release before production training",
        default_asset=None,
        assets=(
            DatasetAsset(
                name="annotations",
                url=None,
                kind="annotations",
                manual=True,
                notes="Download the public annotation export from the authors, then import the local CSV/JSON/JSONL file.",
            ),
        ),
        notes="The paper reports 21k+ expert cue annotations for nearly 5k tracks. SAUND.AI never resolves linked commercial track audio automatically.",
    ),
    DatasetSource(
        source_id="unmixdb",
        name="UnmixDB",
        lane="research",
        purpose=("DJ mix ground truth", "cue regions", "tempo/speed", "technical transition structure"),
        homepage="https://zenodo.org/records/1422385",
        importer="unmixdb",
        contains_audio=True,
        rights_status="source description states CC-licensed redistributable dance tracks; still requires SAUND.AI production rights review",
        default_asset="README.txt",
        assets=(
            DatasetAsset(
                name="README.txt",
                url=None,
                kind="metadata",
                notes="Resolved dynamically through the Zenodo record API.",
            ),
            DatasetAsset(
                name="mixotic-set044-excerpts.zip",
                url=None,
                kind="audio",
                checksum="eadf9fb3b6ed7dad72db72acc111c66a",
                checksum_algorithm="md5",
                size_note="~890 MB",
                notes="Large audio archive; requires --allow-audio-download.",
            ),
        ),
        notes="The importer accepts an extracted UnmixDB directory and normalizes tab-separated start/end/label ground truth.",
    ),
    DatasetSource(
        source_id="fma",
        name="Free Music Archive (FMA)",
        lane="research",
        purpose=("rights-aware music catalog", "genre diversity", "audio bootstrap pool"),
        homepage="https://github.com/mdeff/fma",
        importer="fma",
        contains_audio=True,
        rights_status="metadata is CC BY 4.0; each audio track keeps the artist-selected license and must be filtered/reviewed before production use",
        default_asset="fma_metadata.zip",
        assets=(
            DatasetAsset(
                name="fma_metadata.zip",
                url="https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
                kind="metadata",
                checksum="f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
                checksum_algorithm="sha1",
                size_note="~342 MiB",
            ),
            DatasetAsset(
                name="fma_small.zip",
                url="https://os.unil.cloud.switch.ch/fma/fma_small.zip",
                kind="audio",
                checksum="ade154f733639d52e35e32f5593efe5be76c6d70",
                checksum_algorithm="sha1",
                size_note="~7.2 GiB",
                notes="8,000 30-second tracks; requires --allow-audio-download.",
            ),
        ),
        notes="SAUND.AI imports metadata first and preserves per-track license fields. It does not mark the whole audio corpus production-cleared.",
    ),
    DatasetSource(
        source_id="listenbrainz",
        name="ListenBrainz public listens",
        lane="research",
        purpose=("next-track prior", "sequence statistics", "taste pretraining"),
        homepage="https://listenbrainz.org/data/",
        importer="listenbrainz",
        contains_audio=False,
        rights_status="public listen data is published CC0; SAUND.AI still requires privacy/product review before production personalization training",
        default_asset=None,
        assets=(
            DatasetAsset(
                name="listenbrainz-listens-dump.tar.zst",
                url=None,
                kind="archive",
                manual=True,
                size_note="very large",
                notes="Use an official current mirror, extract the dump, then point the importer at the extracted listens directory.",
            ),
        ),
        notes="The normalizer hashes user identifiers and stores adjacent listening edges, not raw usernames.",
    ),
    DatasetSource(
        source_id="medley2k",
        name="Medley2K",
        lane="research",
        purpose=("human transition structure", "transition timing", "structural pretraining"),
        homepage="https://arxiv.org/abs/2008.11159",
        importer="medley2k",
        contains_audio=False,
        rights_status="paper makes the dataset available; license/distribution terms must be reviewed before production use",
        default_asset=None,
        assets=(
            DatasetAsset(
                name="annotations",
                url=None,
                kind="annotations",
                manual=True,
                notes="Import the downloaded transition metadata/JSON/CSV export. MIDI/audio acquisition is intentionally not automated.",
            ),
        ),
        notes="Used as a structural prior only; medley transitions are not treated as proof of professional DJ taste.",
    ),
)


def source_catalog() -> dict[str, Any]:
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "policy": {
            "external_default_lane": "research",
            "production_default_deny": True,
            "audio_download_requires_explicit_opt_in": True,
        },
        "sources": [source.to_dict() for source in _SOURCES],
    }


def get_source(source_id: str) -> DatasetSource:
    normalized = str(source_id).strip().lower()
    for source in _SOURCES:
        if source.source_id == normalized:
            return source
    raise ValueError(f"Unknown dataset source: {source_id}")


def write_source_catalog(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(source_catalog(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return destination


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _zenodo_asset_url(record_id: str, asset_name: str) -> tuple[str, str | None, str | None]:
    request = urllib.request.Request(
        f"https://zenodo.org/api/records/{record_id}",
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS host
        payload = json.load(response)
    for item in payload.get("files", []):
        if str(item.get("key")) != asset_name:
            continue
        links = item.get("links", {}) if isinstance(item.get("links"), dict) else {}
        url = str(links.get("self") or links.get("content") or "")
        checksum = str(item.get("checksum") or "")
        if checksum and ":" in checksum:
            algorithm, value = checksum.split(":", 1)
            return url, algorithm.lower(), value.lower()
        return url, None, None
    raise ValueError(f"Asset {asset_name!r} was not found in Zenodo record {record_id}.")


def fetch_source_asset(
    source_id: str,
    *,
    asset_name: str | None = None,
    output_dir: str | Path = "datasets/downloads",
    allow_audio_download: bool = False,
    overwrite: bool = False,
) -> Path:
    source = get_source(source_id)
    selected_name = asset_name or source.default_asset
    if not selected_name:
        raise ValueError(f"{source.name} requires manual acquisition; no default downloadable asset is configured.")
    asset = next((item for item in source.assets if item.name == selected_name), None)
    if asset is None:
        raise ValueError(f"Unknown asset {selected_name!r} for {source.name}.")
    if asset.manual:
        raise ValueError(f"{source.name}/{asset.name} requires manual acquisition: {asset.notes}")
    if asset.kind == "audio" and not allow_audio_download:
        raise ValueError("Audio downloads are opt-in. Re-run with --allow-audio-download after reviewing source rights and storage requirements.")

    url = asset.url
    checksum_algorithm = asset.checksum_algorithm
    checksum = asset.checksum
    if source.source_id == "unmixdb":
        url, resolved_algorithm, resolved_checksum = _zenodo_asset_url("1422385", asset.name)
        checksum_algorithm = checksum_algorithm or resolved_algorithm
        checksum = checksum or resolved_checksum
    if not url:
        raise ValueError(f"No direct URL is configured for {source.name}/{asset.name}.")

    root = Path(output_dir).expanduser().resolve() / source.source_id
    root.mkdir(parents=True, exist_ok=True)
    destination = root / asset.name
    if destination.exists() and not overwrite:
        if checksum and checksum_algorithm:
            actual = _hash_file(destination, checksum_algorithm)
            if actual.lower() == checksum.lower():
                return destination
        raise FileExistsError(f"{destination} already exists; use --overwrite to replace it.")

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:  # noqa: S310 - catalog HTTPS URLs only
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    except (urllib.error.URLError, TimeoutError, OSError):
        temporary.unlink(missing_ok=True)
        raise

    if checksum and checksum_algorithm:
        actual = _hash_file(temporary, checksum_algorithm)
        if actual.lower() != checksum.lower():
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {asset.name}: expected {checksum_algorithm}:{checksum}, got {actual}."
            )
    temporary.replace(destination)
    return destination
