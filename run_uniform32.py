"""全视频均匀 32 帧、单次回答基准；独立于 run.py 的 Agentic 流程。

复用 latest_model_32frame_benchmark_test.py 的逐题推理、CSV 保存与续跑结构。
只解码目标帧，复用项目 Prompt、A–G 答案提取和 VLM 接口，不加载 Agentic 检索。
"""

import argparse
import base64
from datetime import datetime
import json
import os
from pathlib import Path
import re
import time

import cv2
import pandas as pd
import requests

from confidence import extract_option
from prompts import build_prompt
from vlm_api import create_vlm_client, vlm_chat_completion


N_FRAMES = 32
STRATEGY = "uniform_32_no_vote"


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "default"


def add_timestamp(frame, index, total, fps):
    """沿用当前实验的水印样式，保持与已完成均匀基准的图像输入一致。"""
    height, width = frame.shape[:2]
    text = f"Frame {index + 1}/{total} | t={index / fps if fps > 0 else 0:.1f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = cv2.getTextSize(text, font, 0.6, 2)[0]
    x, y = width - size[0] - 15, height - 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - size[1] - 8), (x + size[0] + 8, y + 8), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.putText(frame, text, (x, y), font, 0.6, (255, 255, 255), 2)
    return frame


def sample_uniform32(video_path):
    """包含首尾帧的等间隔采样；解码不足 32 帧时记录失败，不缩减预算。"""
    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < N_FRAMES:
            raise ValueError(f"Requires {N_FRAMES} frames; video has {total}: {video_path}")
        fps = video.get(cv2.CAP_PROP_FPS)
        step = (total - 1) / (N_FRAMES - 1)
        indices = [int(i * step) for i in range(N_FRAMES)]
        frames = []
        for index in indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = video.read()
            if not success:
                raise ValueError(f"Cannot decode frame {index}: {video_path}")
            success, buffer = cv2.imencode(".jpg", add_timestamp(frame, index, total, fps))
            if not success:
                raise ValueError(f"Cannot encode frame {index}: {video_path}")
            frames.append(base64.b64encode(buffer).decode("utf-8"))
        return frames, {
            "strategy": STRATEGY, "target_frames": N_FRAMES, "actual_frames": len(frames),
            "window": [0.0, 1.0], "start_ratio": 0.0, "end_ratio": 1.0,
            "selected_indices": indices, "total_frames": total,
        }
    finally:
        video.release()


def answer_once(client, model, category, question, frames):
    content = [{"type": "text", "text": build_prompt(category, question)}]
    content.extend({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}} for frame in frames)
    response = vlm_chat_completion(client, model, messages=[{"role": "user", "content": content}], timeout=120)
    raw = response.choices[0].message.content
    return extract_option(raw), raw


def completed_mask(results):
    """按有效答案跳过已完成题目，不用最后一行推断进度，避免遗漏失败题。"""
    return results["Output"].fillna("").map(extract_option).notna()


def load_results(questions, result_path, resume, run_id):
    """在结果入口统一核对题库和实验策略，拒绝混入其他实验。"""
    if not resume:
        if result_path.exists():
            raise FileExistsError(f"Result already exists; use --resume or another RUN_ID: {result_path}")
        results = questions.copy()
    else:
        results = pd.read_csv(result_path, index_col=0)
        keys = ["video_id", "question_category", "question", "answer"]
        if not results.index.equals(questions.index) or not results[keys].equals(questions[keys]):
            raise ValueError("Resume dataset differs from the original questions")
        if "RetrievalStrategy" not in results or "RunID" not in results:
            raise ValueError("Resume file does not identify a uniform32 experiment")
        if not results["RetrievalStrategy"].dropna().eq(STRATEGY).all() or not results["RunID"].dropna().eq(run_id).all():
            raise ValueError("Resume file belongs to a different strategy or RUN_ID")
    defaults = {"Output": None, "RawOutput": None, "STRIVE": None, "RetrievalStrategy": STRATEGY,
                "RetrievalMeta": None, "RunID": run_id, "Error": None}
    for name, default in defaults.items():
        if name not in results:
            results[name] = default
    # CSV 全空列通常被 pandas 读为 float；这些列后续写入文本，入口统一转换一次。
    text_columns = [name for name in defaults if name != "STRIVE"]
    results[text_columns] = results[text_columns].astype(object)
    return results


def save_results(results, path):
    """同目录临时文件原子替换，避免中断时留下半写入的 CSV。"""
    temporary = path.with_suffix(".csv.tmp")
    results.to_csv(temporary)
    temporary.replace(path)


def emit_progress(payload):
    """看板不可用时不影响主实验；只处理网络请求异常。"""
    url = os.environ.get("DASHBOARD_UPDATE_URL", "http://127.0.0.1:5000/update")
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"[dashboard] update skipped: {error}", flush=True)


def run_benchmark(results, result_path, video_dir, client, model, run_id):
    done = completed_mask(results)
    processed = int(done.sum())
    correct = int(results.loc[done, "Output"].map(extract_option).eq(results.loc[done, "answer"]).sum())
    pending = results.index[~done].tolist()
    print(f"Uniform baseline: {N_FRAMES} full-video frames; one answer call; pending={len(pending)}", flush=True)
    for index in pending:
        row = results.loc[index]
        video_path = video_dir / str(row["video_id"])
        payload = {
            "run_id": run_id, "qa_idx": int(index), "video_id": str(row["video_id"]),
            "video_path": str(video_path.resolve()), "category": row["question_category"],
            "question": row["question"], "correct_answer": row["answer"],
            "retrieval_strategy": STRATEGY, "total_processed": processed, "correct_count": correct,
            "accuracy": correct / processed if processed else 0,
        }
        print(f"[QA #{index}] {row['question_category']} | {row['video_id']} | {STRATEGY}", flush=True)
        emit_progress({**payload, "stage": "全视频均匀抽取 32 帧，单次模型回答"})
        try:
            frames, retrieval = sample_uniform32(video_path)
            emit_progress({**payload, "stage": "已抽取 32 帧，等待模型回答", "frames": frames})
            option, raw = answer_once(client, model, row["question_category"], row["question"], frames)
        except Exception as error:
            # 单题失败继续后续题；失败原因保留在 CSV，续跑时重新处理该题。
            results.at[index, "Error"] = str(error)
            print(f"[QA #{index}] FAILED: {error}", flush=True)
        else:
            results.at[index, "Output"] = f"Option: [{option}]"
            results.at[index, "RawOutput"] = raw
            results.at[index, "RetrievalStrategy"] = STRATEGY
            results.at[index, "RetrievalMeta"] = json.dumps(retrieval)
            results.at[index, "RunID"] = run_id
            results.at[index, "Error"] = None if option else "No valid answer extracted"
            if option:
                processed += 1
                correct += int(option == row["answer"])
            print(f"[QA #{index}] answer={option}; processed={processed}; correct={correct}", flush=True)
            emit_progress({**payload, "model_answer": option, "frames": frames, "retrieval": retrieval,
                           "total_processed": processed, "correct_count": correct,
                           "accuracy": correct / processed if processed else 0, "retry_count": 0})
        save_results(results, result_path)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument("--resume", action="store_true", default=os.environ.get("RUN_MODE") == "resume")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    args = parser.parse_args()
    run_id = safe_name(args.run_id or f"uniform32_{datetime.now():%Y%m%d_%H%M%S}")
    questions = pd.read_parquet(args.data_dir / "MCQ.parquet")
    required = {"video_id", "question_category", "question", "answer"}
    if not required.issubset(questions.columns) or questions[list(required)].isna().any().any():
        raise ValueError("Question dataset requires nonempty video_id, question_category, question and answer")
    client, model, provider = create_vlm_client()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.result_dir / f"{safe_name(model)}_{run_id}_output.csv"
    results = load_results(questions, result_path, args.resume, run_id)
    (args.result_dir / "last_run_id.txt").write_text(run_id + "\n", encoding="ascii")
    print(f"RUN_ID={run_id}\nProvider={provider}; model={model}\nResult CSV={result_path}", flush=True)
    save_results(results, result_path)
    started = time.monotonic()
    run_benchmark(results, result_path, args.data_dir / "videos", client, model, run_id)
    print(f"Finished in {time.monotonic() - started:.1f}s; valid answers={int(completed_mask(results).sum())}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
