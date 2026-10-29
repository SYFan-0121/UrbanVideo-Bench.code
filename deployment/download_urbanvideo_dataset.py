"""下载 Hugging Face 指定版本的数据集并记录下载清单。

The dataset is intentionally kept outside Git.  snapshot_download resumes
interrupted transfers and materializes the files under the requested local
directory, so both experiment folders can use dataset/MCQ.parquet and
dataset/videos/*.mp4 relative paths.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "EmbodiedCity/UrbanVideo-Bench"
ALLOW_PATTERNS = [
    "MCQ.parquet",
    "videos/*.mp4",
    "README.md",
    "evaluations.jpg",
    ".gitattributes",
]


def summarize(dataset_dir: Path) -> tuple[int, int]:
    videos_dir = dataset_dir / "videos"
    videos = sorted(videos_dir.glob("*.mp4")) if videos_dir.exists() else []
    total_bytes = sum(path.stat().st_size for path in videos)
    mcq = dataset_dir / "MCQ.parquet"
    print(f"dataset_dir={dataset_dir}")
    print(f"MCQ.parquet={'present' if mcq.is_file() else 'MISSING'}")
    print(f"video_count={len(videos)}")
    print(f"video_bytes={total_bytes}")
    print(f"video_gib={total_bytes / (1024 ** 3):.2f}")
    return len(videos), total_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset", help="Target dataset directory")
    parser.add_argument("--revision", default="main", help="Hub revision to download")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing an optional Hugging Face token",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get(args.token_env) or None
    repo_info = HfApi().dataset_info(repo_id=REPO_ID, revision=args.revision, token=token)
    resolved_revision = repo_info.sha
    print(f"Downloading {REPO_ID}@{resolved_revision} into {data_dir}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=resolved_revision,
        local_dir=str(data_dir),
        allow_patterns=ALLOW_PATTERNS,
        token=token,
        max_workers=max(1, args.max_workers),
    )
    video_count, video_bytes = summarize(data_dir)
    if not (data_dir / "MCQ.parquet").is_file() or video_count == 0:
        raise SystemExit("Dataset download is incomplete: MCQ.parquet or videos are missing")
    manifest = {
        "repo_id": REPO_ID,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_count": video_count,
        "video_bytes": video_bytes,
    }
    manifest_path = data_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
