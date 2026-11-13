"""
Offline audit of a Hugging Face SentenceTransformer snapshot.

Hashes cached model files without downloading anything.
This is an inventory, not yet a runtime enforcement lock.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re

from datetime import datetime, timezone
from pathlib import Path


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def fingerprint_snapshot(snapshot: Path) -> dict:
    """Fingerprint files by relative path and content."""

    snapshot = Path(snapshot).resolve()

    if not snapshot.is_dir():
        raise ValueError(
            "Snapshot directory does not exist"
        )

    files = []

    for path in sorted(snapshot.rglob("*")):
        # A Hugging Face snapshot normally contains
        # symlinks to cached blobs. Hash their contents.
        if path.is_symlink() and not path.is_file():
            raise ValueError(
                "Broken or unsupported snapshot symlink"
            )

        if not path.is_file():
            continue

        relative = path.relative_to(
            snapshot
        ).as_posix()

        files.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    if not files:
        raise ValueError(
            "No model files found in snapshot"
        )

    encoded = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    names = {item["path"] for item in files}

    weight_files = [
        name
        for name in sorted(names)
        if name.endswith(
            (".safetensors", ".bin")
        )
        and (
            "model" in name.lower()
            or "pytorch" in name.lower()
        )
    ]

    return {
        "snapshot_sha256": hashlib.sha256(
            encoded
        ).hexdigest(),
        "asset_file_count": len(files),
        "total_asset_bytes": sum(
            file["size_bytes"]
            for file in files
        ),
        "modules_json_present": (
            "modules.json" in names
        ),
        "weight_files": weight_files,
        "files": files,
    }


def resolve_cached_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    custom_cache = os.getenv(
        "SENTENCE_TRANSFORMERS_HOME"
    )

    # local_files_only=True guarantees no
    # network download is attempted.
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_files_only=True,
        cache_dir=custom_cache or None,
    )

    return Path(path)


def audit(output_dir: Path) -> dict:
    snapshot = resolve_cached_snapshot()

    fingerprint = fingerprint_snapshot(
        snapshot
    )

    report = {
        "schema_version": 1,
        "captured_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_id": MODEL_ID,
        "cached_snapshot_path": str(
            snapshot.resolve()
        ),
        "cached_revision": snapshot.name,
        "revision_looks_like_git_sha": bool(
            re.fullmatch(
                r"[0-9a-f]{40}",
                snapshot.name,
            )
        ),
        "sentence_transformers_version": (
            importlib.metadata.version(
                "sentence-transformers"
            )
        ),
        "cache_resolution": "local_only",
        "model_downloaded": False,
        "runtime_revision_enforced": False,
        "offline_model_load_verified": False,
        "preflight_fingerprint_enforced": False,
        **fingerprint,
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    report_path = (
        output_dir / "sentence_model_audit.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    report = audit(args.output_dir)

    print("=" * 60)
    print("SENTENCETRANSFORMER ASSET AUDIT")
    print("=" * 60)

    for key in (
        "model_id",
        "cached_revision",
        "revision_looks_like_git_sha",
        "asset_file_count",
        "total_asset_bytes",
        "snapshot_sha256",
        "modules_json_present",
        "weight_files",
        "runtime_revision_enforced",
        "offline_model_load_verified",
        "preflight_fingerprint_enforced",
    ):
        print(f"{key}: {report[key]}")

    print(
        "Report:",
        args.output_dir
        / "sentence_model_audit.json",
    )


if __name__ == "__main__":
    main()
