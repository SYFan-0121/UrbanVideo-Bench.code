"""Merge manually transferred MP4s without overwriting dataset files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", default=r"D:\VLM\syf\incoming-local-videos")
    parser.add_argument("--destination", default=r"D:\VLM\syf\UrbanVideo-Bench.code\dataset\videos")
    parser.add_argument(
        "--manifest",
        default=r"D:\VLM\syf\UrbanVideo-Bench.code\result\supplemental_video_manifest.csv",
    )
    args = parser.parse_args()

    incoming = Path(args.incoming).resolve()
    destination = Path(args.destination).resolve()
    manifest_path = Path(args.manifest).resolve()
    if not incoming.is_dir():
        raise SystemExit(f"Incoming folder does not exist: {incoming}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    conflicts = 0
    for source in sorted(incoming.glob("*.mp4")):
        target = destination / source.name
        source_hash = sha256(source)
        if target.exists():
            target_hash = sha256(target)
            if source_hash == target_hash:
                status = "already_present_identical"
            else:
                status = "conflict_not_overwritten"
                conflicts += 1
        else:
            shutil.copy2(source, target)
            if target.stat().st_size != source.stat().st_size or sha256(target) != source_hash:
                target.unlink(missing_ok=True)
                status = "copy_verification_failed"
                conflicts += 1
            else:
                status = "copied_and_verified"
        rows.append({"video_id": source.name, "bytes": source.stat().st_size, "sha256": source_hash, "status": status})
        print(f"{status}: {source.name}")

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "bytes", "sha256", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifest={manifest_path}")
    print(f"merged_files={sum(row['status'] == 'copied_and_verified' for row in rows)}")
    print(f"conflicts={conflicts}")
    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
