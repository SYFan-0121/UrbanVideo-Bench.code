"""Verify the local dataset against MCQ.parquet and report missing videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--expected-videos", type=int, default=0)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return non-zero when any question-referenced video is missing",
    )
    args = parser.parse_args()

    root = Path(args.data_dir).resolve()
    mcq_path = root / "MCQ.parquet"
    videos_dir = root / "videos"
    if not mcq_path.is_file():
        raise SystemExit(f"Missing {mcq_path}")
    if not videos_dir.is_dir():
        raise SystemExit(f"Missing {videos_dir}")

    questions = pd.read_parquet(mcq_path)
    if "video_id" not in questions.columns:
        raise SystemExit("MCQ.parquet has no video_id column")
    available = {path.name for path in videos_dir.glob("*.mp4")}
    question_videos = {str(value) for value in questions["video_id"].dropna()}
    missing = sorted(question_videos - available)
    extra = sorted(available - question_videos)
    total_bytes = sum(path.stat().st_size for path in videos_dir.glob("*.mp4"))
    present_question_mask = questions["video_id"].astype(str).isin(available)

    print(f"questions={len(questions)}")
    print(f"unique_question_videos={len(question_videos)}")
    print(f"available_videos={len(available)}")
    print(f"video_gib={total_bytes / (1024 ** 3):.2f}")
    print(f"questions_with_video={int(present_question_mask.sum())}")
    print(f"question_coverage={present_question_mask.mean():.4%}")
    print(f"missing_question_videos={len(missing)}")
    print(f"extra_videos_not_referenced={len(extra)}")
    if missing:
        missing_path = root / "missing_video_ids.txt"
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"missing_video_list={missing_path}")
        print("First missing videos:")
        for name in missing[:20]:
            print(f"  {name}")
    if args.expected_videos and len(available) != args.expected_videos:
        print(f"WARNING: expected {args.expected_videos} videos, found {len(available)}")
    return 1 if missing and args.require_complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
