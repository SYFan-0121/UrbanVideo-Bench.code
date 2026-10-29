# -*- coding: utf-8 -*-
"""执行 16 类 Agentic 评测、抽帧、断点续跑与结果保存。

核心流程：
1. 读取 dataset/MCQ.parquet 和 dataset/videos 下的视频。
2. 全部 16 类统一使用 Agentic 检索，按全局模式选择 no_vote 或 strive。
3. 调用 VLM 获取答案。
4. 写入 result/{model}_output.csv，并向 dashboard 推送实时状态。
"""

import cv2
import base64
import time
import json
import re
import glob
from openai import OpenAI
import os
import pandas as pd
import warnings
import copy
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import build_prompt
from confidence import (confident_answer, detect_num_options, compute_strive,
                        extract_option, extract_self_score, get_sampling_variants,
                        CATEGORY_DIFFICULTY)
from agentic_retrieval import AgenticRetrievalContext, sample_frames_agentic_retrieval
from video_locks import get_video_read_lock
from vlm_api import create_vlm_client, vlm_chat_completion
import requests

DASHBOARD_UPDATE_URL = os.environ.get("DASHBOARD_UPDATE_URL", "http://127.0.0.1:5000/update")
# 每次最终回答的基础帧预算；STRIVE 重试时仍可按原有规则增加到 48/64 帧。
BASE_N_FRAMES = 32


def to_jsonable(value):
    """Convert numpy/pandas scalar values inside dashboard payloads to JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


# RUN_ID 是一次实验的名字，会出现在 CSV 文件名和 SQLite 文件名里。
# 这里把中文、空格和特殊符号替换掉，避免 Windows/Linux 文件名不兼容。
def make_safe_run_id(value):
    """将实验名规整成可用于文件名和数据库名的 RUN_ID。"""
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "default"


# 每个模型和 RUN_ID 对应一个独立结果 CSV。
# 这样新策略重跑不会覆盖旧策略的结果。
def result_csv_path(result_dir, model, run_id):
    """根据 RUN_ID 生成本次实验专属 CSV 路径。"""
    return os.path.join(result_dir, f"{make_safe_run_id(model)}_{make_safe_run_id(run_id)}_output.csv")


# 扫描 result 目录，找出同一模型已经跑过哪些 RUN_ID。
# 启动时的“续跑已有实验”选项依赖这个函数。
def discover_run_ids(result_dir, model):
    """扫描 result 目录中已有的该模型实验 CSV，供续跑选择。"""
    model = make_safe_run_id(model)
    pattern = os.path.join(result_dir, f"{model}_*_output.csv")
    run_ids = []
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        prefix = f"{model}_"
        suffix = "_output.csv"
        if name.startswith(prefix) and name.endswith(suffix):
            run_ids.append(name[len(prefix):-len(suffix)])
    legacy = os.path.join(result_dir, f"{model}_output.csv")
    if os.path.exists(legacy) and "legacy" not in run_ids:
        run_ids.append("legacy")
    return sorted(set(run_ids))


# 决定本次运行是新实验还是续跑。
# 优先读环境变量 RUN_ID/RUN_MODE，没有时再让用户在控制台选择。
def resolve_run_config(result_dir, model, agentic_mode):
    """
    启动前决定本次实验 RUN_ID 和 CSV 文件。

    环境变量优先：
    - RUN_ID: 指定实验名。
    - RUN_MODE=new/resume: 新实验或续跑。

    没有环境变量且在交互终端中运行时，会提示用户选择。
    """
    env_run_id = os.environ.get("RUN_ID") or os.environ.get("EXPERIMENT_ID")
    env_mode = (os.environ.get("RUN_MODE") or "").strip().lower()
    existing = discover_run_ids(result_dir, model)

    if env_run_id:
        run_id = make_safe_run_id(env_run_id)
        mode = env_mode if env_mode in {"new", "resume"} else ("resume" if run_id in existing else "new")
        return run_id, mode

    if os.environ.get("CI") or not hasattr(os.sys.stdin, "isatty") or not os.sys.stdin.isatty():
        if existing:
            return existing[-1], "resume"
        return make_safe_run_id(f"{agentic_mode}_{time.strftime('%Y%m%d_%H%M%S')}"), "new"

    print("\n=== RUN_ID / 实验选择 ===")
    if existing:
        print("已有实验：")
        for i, rid in enumerate(existing, start=1):
            print(f"  {i}. {rid}")
    else:
        print("未发现已有实验 CSV。")
    print("选择运行方式：")
    print("  1. 续跑已有实验（继续使用同一个 CSV 和 dashboard SQLite）")
    print("  2. 新实验/换策略重跑（创建新的 CSV 和新的 dashboard SQLite）")
    choice = input("请输入 1 或 2 [默认 1]: ").strip() or "1"

    if choice == "1" and existing:
        selected = input(f"输入要续跑的 RUN_ID [默认 {existing[-1]}]: ").strip() or existing[-1]
        return make_safe_run_id(selected), "resume"

    default_new = make_safe_run_id(f"{agentic_mode}_{time.strftime('%Y%m%d_%H%M%S')}")
    selected = input(f"输入新 RUN_ID [默认 {default_new}]: ").strip() or default_new
    return make_safe_run_id(selected), "new"


def log_progress(qa_idx, message):
    """统一的单题控制台日志；qa_idx 用于在并发运行时区分不同题目。"""
    print(f"[QA #{qa_idx}] {message}", flush=True)


# dashboard 是可选的，没启动前端服务也不能影响主评测。
# 所以这里用短超时和异常吞掉，只负责尽力推送实时状态。
def emit_to_dashboard(payload: dict):
    """向本地 dashboard 后端推送进度；dashboard 未启动时静默跳过，不影响评测主流程。"""
    try:
        requests.post(
            DASHBOARD_UPDATE_URL,
            json=to_jsonable(payload),
            timeout=5
        )
    except Exception as exc:
        print(f"[dashboard] update skipped: {exc}")


# 给每张抽出的帧写上原始帧号和视频时间。
# 这样大模型和人工回看都能知道这张图来自视频哪个位置。
def add_timestamp_watermark(frame, frame_idx, total_frames, fps):
    """
    在帧右下角叠加帧号和时间戳。

    frame_idx: 当前帧在视频中的原始下标。
    total_frames: 视频总帧数，用于显示 Frame x/total。
    fps: 视频帧率，用于把帧号转换为秒级时间。
    """
    height, width = frame.shape[:2]
    timestamp = frame_idx / fps if fps > 0 else 0
    watermark_text = f"Frame {frame_idx + 1}/{total_frames} | t={timestamp:.1f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size = cv2.getTextSize(watermark_text, font, font_scale, thickness)[0]
    x = width - text_size[0] - 15
    y = height - 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - text_size[1] - 8), (x + text_size[0] + 8, y + 8), (0, 0, 0), -1)
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.putText(frame, watermark_text, (x, y), font, font_scale, (255, 255, 255), thickness)
    return frame


# 基础均匀抽帧函数。
# start_ratio/end_ratio 是 0 到 1 的视频时间比例，不是秒数。
# n 是希望抽出的帧数，返回值是 base64 JPEG 字符串列表。
def sample_frames_by_ratio(video, total_frames, start_ratio, end_ratio, n):
    """
    从已经打开的 cv2.VideoCapture 中按时间比例窗口均匀抽取 n 帧。

    start_ratio/end_ratio: 取值范围 [0, 1]，表示视频时间轴上的起止比例。
    n: 目标帧数；如果窗口内可用帧不足，会自动降到可用帧数。
    返回值: base64 编码后的 JPEG 帧列表，供 VLM image_url 输入使用。
    """
    start = int(total_frames * start_ratio)
    end = min(int(total_frames * end_ratio) - 1, total_frames - 1)
    fps = video.get(cv2.CAP_PROP_FPS)
    available_frames = end - start + 1
    if available_frames <= 0:
        return []
    n = min(n, available_frames)
    if n > 1:
        step = (end - start) / (n - 1) if (n - 1) > 0 else 1
        indices = [int(start + i * step) for i in range(n)]
    else:
        indices = [(start + end) // 2]
    frames = []
    for i, idx in enumerate(indices):
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            frame_with_watermark = add_timestamp_watermark(frame, idx, total_frames, fps)
            _, buf = cv2.imencode(".jpg", frame_with_watermark)
            frames.append(base64.b64encode(buf).decode("utf-8"))
    return frames


# 线程安全的路径版抽帧函数。
# 多线程同时读同一个视频时，用视频锁避免 OpenCV seek/read 互相干扰。
def sample_frames_by_ratio_from_path(video_path, total_frames, start_ratio, end_ratio, n):
    """
    文件路径版本的均匀抽帧函数。

    这里会临时打开视频并使用 get_video_read_lock 加锁，避免并发线程同时 seek/read
    同一个视频文件导致 OpenCV 状态冲突。
    """
    with get_video_read_lock(video_path):
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            return []
        try:
            return sample_frames_by_ratio(video, total_frames, start_ratio, end_ratio, n)
        finally:
            video.release()


# 动态定位的第一步。
# 只给 VLM 8 张全局 overview 帧，让它判断问题最相关的时间窗口。
def stage1_locate(client, model, question, overview_frames):
    """
    Stage1 时域定位：用 8 张全局 overview 帧让 VLM 判断问题最相关的视频时间窗口。

    返回:
        (start_ratio, end_ratio)，均为 [0, 1] 范围内的小数。
    失败或窗口过窄时回退到全视频 [0.0, 1.0]，保证后续流程不崩。
    """
    print("[Stage1] Locating relevant temporal window with 8 overview frames...", flush=True)
    prompt = (
        "You are analyzing a first-person perspective video of an agent navigating an environment. "
        "The images below are 8 uniformly sampled overview frames (start to end).\n\n"
        f"Question: {question}\n\n"
        "Based on the question, identify the most relevant temporal segment to answer it.\n"
        "Reply ONLY with JSON, no extra text:\n"
        '{"start_ratio": <0.0-1.0>, "end_ratio": <0.0-1.0>, "reason": "<brief>"}'
    )
    content = [{"type": "text", "text": prompt}]
    for f in overview_frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})
    print("[Stage1] Sending overview frames to VLM and waiting for location response...", flush=True)
    result = vlm_chat_completion(
        client,
        model,
        messages=[{"role": "user", "content": content}],
        timeout=60,
        max_tokens=100,
    )
    raw = result.choices[0].message.content.strip()
    try:
        data = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group())
        s, e = float(data['start_ratio']), float(data['end_ratio'])
        s, e = max(0.0, min(s, 1.0)), max(0.0, min(e, 1.0))
        if e - s < 0.05:
            print("[Stage1] Located window is too small; fallback to full video [0.00, 1.00].", flush=True)
            return 0.0, 1.0
        print(f"[Stage1] Located window: [{s:.2f}, {e:.2f}] reason={data.get('reason', '')}", flush=True)
        return s, e
    except:
        print("[Stage1] Failed to parse location response; fallback to full video [0.00, 1.00].", flush=True)
        return 0.0, 1.0


# STRIVE 低置信度时使用的纠错定位。
# 它会告诉模型上一次窗口导致投票分裂，请重新选择更可能正确的时间段。
def stage1_locate_corrective(client, model, question, overview_frames,
                              prev_start: float, prev_end: float):
    """
    低置信度后的修正定位。

    与 stage1_locate 的区别是：Prompt 会告诉 VLM 上一次窗口导致三票严重分裂，
    要求它重新审视全视频并给出不同或更准确的窗口。

    prev_start/prev_end: 上一次尝试使用的窗口，用于提醒模型避开错误定位。
    """
    print(
        f"[Stage1-Corrective] Re-locating after low confidence. Previous window: "
        f"[{prev_start:.2f}, {prev_end:.2f}]",
        flush=True,
    )
    prompt = (
        "You are analyzing a first-person perspective video of an agent navigating an environment. "
        "The images below are 8 uniformly sampled overview frames (start to end).\n\n"
        f"Question: {question}\n\n"
        f"IMPORTANT: A previous attempt located the segment [{prev_start:.2f}, {prev_end:.2f}], "
        f"but the model produced completely inconsistent answers across that segment (all 3 votes differed). "
        f"This strongly suggests the previous location was INCORRECT.\n"
        f"Please re-examine the entire video carefully and identify a DIFFERENT, more accurate segment.\n\n"
        "Reply ONLY with JSON, no extra text:\n"
        '{"start_ratio": <0.0-1.0>, "end_ratio": <0.0-1.0>, "reason": "<brief>"}'
    )
    content = [{"type": "text", "text": prompt}]
    for f in overview_frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})
    print("[Stage1-Corrective] Sending corrective location prompt to VLM...", flush=True)
    result = vlm_chat_completion(
        client,
        model,
        messages=[{"role": "user", "content": content}],
        timeout=60,
        max_tokens=120,
    )
    raw = result.choices[0].message.content.strip()
    try:
        data = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group())
        s, e = float(data['start_ratio']), float(data['end_ratio'])
        s, e = max(0.0, min(s, 1.0)), max(0.0, min(e, 1.0))
        if e - s < 0.05:
            print("[Stage1-Corrective] Located window is too small; fallback to full video [0.00, 1.00].", flush=True)
            return 0.0, 1.0
        print(f"[Stage1-Corrective] New window: [{s:.2f}, {e:.2f}] reason={data.get('reason', '')}", flush=True)
        return s, e
    except:
        print("[Stage1-Corrective] Failed to parse response; fallback to full video [0.00, 1.00].", flush=True)
        return 0.0, 1.0


def answer_once_global_uniform(client, model, video_path, total_frames,
                               qa_category, question, n_frames=32):
    """不走 STRIVE 的基线调用：全视频均匀抽 n_frames 帧，然后只问 VLM 一次。"""
    print(f"[NoVote] Sampling {n_frames} uniform frames from full video...", flush=True)
    frames = sample_frames_by_ratio_from_path(video_path, total_frames, 0.0, 1.0, n=n_frames)
    return answer_once_with_frames(client, model, qa_category, question, frames)


# 已经有最终帧之后，用这个函数真正向 VLM 提问。
# content 里第一项是文字 prompt，后面每一项是 base64 图片帧。
def answer_once_with_frames(client, model, qa_category, question, frames):
    """
    将已经抽好的帧和类别 Prompt 一起发送给 VLM，提取单次答案。

    qa_category: 问题类别，用于 build_prompt 选择类别专属提示词。
    frames: base64 JPEG 列表；本函数不再访问视频文件。
    返回: (选项字母, 原帧列表, 原始模型输出文本)。
    """
    if not frames:
        print("[VLM] No frames available; skip model call.", flush=True)
        return None, frames, ""

    print(f"[VLM] Preparing {len(frames)} frames for category '{qa_category}'.", flush=True)
    content = [{"type": "text", "text": build_prompt(qa_category, question)}]
    for buf in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{buf}"}
        })

    try:
        print("[VLM] Sending frames to model and waiting for answer...", flush=True)
        result = vlm_chat_completion(
            client,
            model,
            messages=[{"role": "user", "content": content}],
            timeout=120,
        )
        raw = result.choices[0].message.content
        option = extract_option(raw)
        print(f"[VLM] Model response received. Extracted option: {option}", flush=True)
        return option, frames, raw
    except Exception as e:
        err_str = str(e)
        if 'Arrearage' in err_str or 'overdue' in err_str.lower():
            raise RuntimeError(f"[FATAL] API arrearage: {err_str}")
        raise


# 单题处理的总调度函数。
# 一道题从读取视频、选择抽帧策略、调用模型、计算置信度、写 CSV、推送前端都在这里串起来。
def process_one(qa_idx, res, res_path, folder_path, model, client,
                AGENTIC_GLOBAL_MODE,
                RUN_ID,
                _lock, _fatal_event,
                _total_processed_ref, _correct_count_ref):
    """
    处理单条 QA 样本，是 run.py 的核心调度函数。

    主要职责：
    1. 读取题目、视频路径和正确答案。
    2. 根据全局模式选择 Agentic 单次回答或 Agentic STRIVE。
    3. 调用 Agentic 检索、STRIVE 投票或单次 VLM 回答。
    4. 记录答案、置信度、检索元信息和全局进度。

    _lock: 多线程写 CSV 和累计准确率时使用的互斥锁。
    _fatal_event: API 欠费等致命错误发生时通知其它线程尽快停止。
    _total_processed_ref/_correct_count_ref: 用 list 包一层，便于线程内原地更新计数。
    """
    if _fatal_event.is_set():
        return

    select_vid_name = res['video_id'].loc[qa_idx]
    qa_category     = res['question_category'].loc[qa_idx]
    qa              = res['question'].loc[qa_idx]
    video_path = os.path.join(folder_path, str(select_vid_name))
    correct_answer = str(res['answer'].loc[qa_idx]).strip().upper() if 'answer' in res.columns else None

    log_progress(qa_idx, "=" * 72)
    log_progress(qa_idx, f"Start processing video={select_vid_name}")
    log_progress(qa_idx, f"Question category: {qa_category}")
    log_progress(qa_idx, f"Question: {qa}")
    if correct_answer is not None:
        log_progress(qa_idx, f"Ground-truth answer: {correct_answer}")

    dashboard_payload = {
        # 这些字段会被 server.py 写入运行队列。
        # 完成前不会落库，完成后才写入 SQLite 历史表。
        "run_id": RUN_ID,
        "qa_idx": int(qa_idx),
        "video_id": str(select_vid_name),
        "video_path": video_path,
        "category": qa_category,
        "question": qa,
        "correct_answer": str(res['answer'].loc[qa_idx]) if 'answer' in res.columns else None,
        "total_processed": _total_processed_ref[0],
        "correct_count": _correct_count_ref[0],
        "accuracy": _correct_count_ref[0] / _total_processed_ref[0] if _total_processed_ref[0] > 0 else 0,
    }

    def emit_stage(stage, **extra):
        # stage 是前端显示的“当前正在做什么”。
        # extra 用来追加帧、召回摘要或策略名等阶段性信息。
        payload = dashboard_payload.copy()
        payload["stage"] = stage
        payload["updated_at"] = time.time()
        payload.update(extra)
        emit_to_dashboard(payload)

    emit_stage("已进入运行队列，正在打开视频")

    video = None
    with get_video_read_lock(video_path):
        log_progress(qa_idx, "Opening video and reading metadata...")
        emit_stage("正在读取视频元信息")
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            log_progress(qa_idx, f"Error: Cannot open {select_vid_name}")
            with _lock:
                res.at[qa_idx, 'Output'] = None
                res.to_csv(res_path)
            return
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        video = None

    try:
        def safe_sample_func(_video, _total_frames, s, e, n):
            # confident_answer 要求 sample_func 接收 video/total_frames 参数。
            # 当前实现为了线程安全，忽略传入的 video，始终从 video_path 临时打开读取。
            log_progress(qa_idx, f"Sampling uniform frames: window=[{s:.2f}, {e:.2f}], n={n}")
            return sample_frames_by_ratio_from_path(video_path, total_frames, s, e, n)

        # 全部类别从全视频范围进入 Agentic，类别差异由检索计划处理。
        start_ratio, end_ratio = 0.0, 1.0
        initial_strategy = f"agentic_retrieval_{AGENTIC_GLOBAL_MODE}"

        num_opts = detect_num_options(qa)
        log_progress(
            qa_idx,
            f"Selected algorithm: {initial_strategy}; initial window=[{start_ratio:.2f}, {end_ratio:.2f}], "
            f"num_options={num_opts}, total_frames={total_frames}",
        )

        dashboard_payload.update({
            "retrieval_strategy": initial_strategy,
            "stage": "已选择抽帧算法，准备抽帧",
            "total_frames": total_frames,
        })
        emit_stage(
            "已选择抽帧算法，准备抽帧",
            retrieval_strategy=initial_strategy,
            total_frames=total_frames,
        )

        retrieval_meta = None

        # Agentic no_vote：只检索一次证据帧，然后只调用一次 VLM；速度快但没有 STRIVE 置信度。
        # 适合快速测试，但没有 STRIVE 分数。
        if AGENTIC_GLOBAL_MODE == "no_vote":
            log_progress(qa_idx, "Agentic no-vote mode: start retrieval-based frame extraction.")
            emit_stage("Agentic 正在规划、构建候选池并做语义召回", retrieval_strategy=initial_strategy)
            final_frames, retrieval_meta = sample_frames_agentic_retrieval(
                video_path=video_path,
                question=qa,
                qa_category=qa_category,
                n_frames=BASE_N_FRAMES,
                client=client,
                model=model,
                variant_id=0,
            )
            log_progress(qa_idx, f"Agentic retrieval finished. final_frames={len(final_frames)}")
            emit_stage(
                "Agentic 召回完成，正在把最终帧送入模型",
                retrieval_strategy=initial_strategy,
                frames=final_frames,
                retrieval=retrieval_meta,
            )
            final_answer, final_frames, _raw_output = answer_once_with_frames(
                client=client,
                model=model,
                qa_category=qa_category,
                question=qa,
                frames=final_frames,
            )
            strive_score = None
            breakdown = {
                "strategy": initial_strategy,
                "retrieval": retrieval_meta,
                "votes": [final_answer],
            }
            retry_count = 0
        # Agentic strive：复用同一个 AgenticRetrievalContext，让 K=3 投票共享候选池和语义召回。
        # 三次投票共享同一个检索上下文，但使用不同窗口和 variant_id。
        else:  # strive；合法模式已在程序入口统一校验。
            log_progress(qa_idx, "Agentic STRIVE mode: start shared retrieval context and K=3 confidence voting.")
            emit_stage("Agentic STRIVE 正在初始化共享检索上下文", retrieval_strategy=initial_strategy)
            retrieval_metas = []
            sample_state = {"variant_id": 0}

            with AgenticRetrievalContext(
                video_path=video_path,
                question=qa,
                qa_category=qa_category,
                client=client,
                model=model,
            ) as retrieval_context:

                def agentic_sample_func(_video, _total_frames, _start_ratio, _end_ratio, n):
                    """适配 confident_answer 的抽帧回调：把 STRIVE 的扰动窗口传入 Agentic 检索。"""
                    # 每次 STRIVE 投票都会调用一次这个函数。
                    # variant_id 让三次投票在同一语义区域附近做轻微扰动。
                    variant_id = sample_state["variant_id"]
                    sample_state["variant_id"] += 1
                    log_progress(
                        qa_idx,
                        f"Agentic sampling variant={variant_id}, window=[{_start_ratio:.2f}, {_end_ratio:.2f}], n={n}",
                    )
                    frames, meta = sample_frames_agentic_retrieval(
                        video_path=video_path,
                        question=qa,
                        qa_category=qa_category,
                        n_frames=n,
                        client=client,
                        model=model,
                        variant_id=variant_id,
                        start_ratio=_start_ratio,
                        end_ratio=_end_ratio,
                        retrieval_context=retrieval_context,
                    )
                    retrieval_metas.append(meta)
                    log_progress(
                        qa_idx,
                        f"Agentic variant={variant_id} finished. frames={len(frames)}, "
                        f"selected_indices={len(meta.get('selected_indices', []))}",
                    )
                    return frames

                log_progress(qa_idx, "Entering STRIVE confidence loop for Agentic retrieval.")
                final_answer, strive_score, breakdown, retry_count, final_frames = confident_answer(
                    client=client, model=model,
                    video=None, total_frames=total_frames,
                    qa_category=qa_category, question=qa,
                    start_ratio=0.0, end_ratio=1.0,
                    num_options=num_opts, K=3, n_frames=BASE_N_FRAMES,
                    high_threshold=0.70, low_threshold=0.40, max_retries=2,
                    sample_func=agentic_sample_func,
                    initial_strategy=initial_strategy,
                    stage1_locate_func=stage1_locate,
                    stage1_locate_corrective_func=stage1_locate_corrective,
                    overview_sample_func=safe_sample_func,
                    emit_to_dashboard_func=emit_to_dashboard,
                    dashboard_payload=dashboard_payload,
                    qa_idx=qa_idx,
                )
            retrieval_meta = {
                "strategy": initial_strategy,
                "variants": retrieval_metas,
            }
            breakdown["retrieval"] = retrieval_meta

        res_str = f"Option: [{final_answer}]"
        is_correct = None
        if correct_answer is not None:
            is_correct = bool(final_answer and final_answer.strip().upper() == correct_answer)
        if strive_score is None:
            log_progress(
                qa_idx,
                f"Final decision: answer={final_answer}, STRIVE=N/A, strategy={initial_strategy}, "
                f"accepted_by=single_call, correct={is_correct}",
            )
        else:
            log_progress(
                qa_idx,
                f"Final decision: answer={final_answer}, STRIVE={strive_score:.3f}, strategy={initial_strategy}, "
                f"retry_count={retry_count}, accepted=True, correct={is_correct}",
            )
            if breakdown:
                log_progress(
                    qa_idx,
                    "Final STRIVE variables: "
                    f"V={breakdown.get('V')}, M={breakdown.get('M')}, E={breakdown.get('E')}, "
                    f"gamma_base={breakdown.get('γ_base')}, gamma_eff={breakdown.get('γ_eff')}, "
                    f"delta_factor={breakdown.get('δ_factor')}, top_answer={breakdown.get('top_answer')}",
                )

        with _lock:
            res.at[qa_idx, 'Output'] = res_str
            res.at[qa_idx, 'STRIVE'] = strive_score
            res.at[qa_idx, 'RetrievalStrategy'] = initial_strategy
            res.at[qa_idx, 'RunID'] = RUN_ID
            if retrieval_meta is not None:
                res.at[qa_idx, 'RetrievalMeta'] = json.dumps(retrieval_meta, ensure_ascii=False)
            _total_processed_ref[0] += 1
            if 'answer' in res.columns:
                correct = str(res['answer'].loc[qa_idx]).strip().upper()
                if final_answer and final_answer.strip().upper() == correct:
                    _correct_count_ref[0] += 1
            res.to_csv(res_path)
            if 'answer' in res.columns and _total_processed_ref[0] > 0:
                acc = _correct_count_ref[0] / _total_processed_ref[0]
                log_progress(
                    qa_idx,
                    f"Global progress: processed={_total_processed_ref[0]}, "
                    f"correct={_correct_count_ref[0]}, running_accuracy={acc:.4f}",
                )

        emit_to_dashboard({
            "run_id": RUN_ID,
            "qa_idx": int(qa_idx),
            "video_id": str(select_vid_name),
            "video_path": video_path,
            "category": qa_category,
            "question": qa,
            "correct_answer": str(res['answer'].loc[qa_idx]) if 'answer' in res.columns else None,
            "model_answer": final_answer,
            "total_processed": _total_processed_ref[0],
            "correct_count":   _correct_count_ref[0],
            "accuracy": _correct_count_ref[0] / _total_processed_ref[0] if _total_processed_ref[0] > 0 else 0,
            "frames": final_frames,
            "strive": breakdown,
            "retry_count": retry_count,
            "retrieval_strategy": initial_strategy,
        })

    except RuntimeError as e:
        print(f"\n[FATAL #{qa_idx}] {e}")
        _fatal_event.set()
        with _lock:
            res.to_csv(res_path)

    except Exception as e:
        print(f"[#{qa_idx}] Error: {e}")
        with _lock:
            res.at[qa_idx, 'Output'] = 'Option: [None]'
            res.to_csv(res_path)
        time.sleep(2)

    finally:
        if video is not None:
            video.release()


if __name__ == '__main__':
    # 并发处理多个 QA；每个 QA 内部可能还有 K=3 并发 VLM 投票。
    _lock = threading.Lock()
    _fatal_event = threading.Event()
    # A single local GPU should be driven serially. Keep the historical
    # default for cloud runs, and set VLM_OUTER_CONCURRENCY=1 for the Windows
    # server service.
    CONCURRENCY = max(1, int(os.environ.get("VLM_OUTER_CONCURRENCY", "4")))
    # "no_vote" = scheme A; "strive" = scheme B.
    AGENTIC_GLOBAL_MODE = os.environ.get("AGENTIC_GLOBAL_MODE", "no_vote").strip().lower()
    if AGENTIC_GLOBAL_MODE not in {"no_vote", "strive"}:
        raise ValueError("AGENTIC_GLOBAL_MODE must be either 'no_vote' or 'strive'")

    # 当前默认使用 Xiaomi MiMo。可通过环境变量覆盖：
    #   VLM_PROVIDER=mimo
    #   MIMO_MODEL=mimo-v2-omni
    #   MIMO_API_KEY=...
    client, model, vlm_provider = create_vlm_client()
    print(f"[VLM] Provider={vlm_provider}, model={model}", flush=True)

    # 原千问 / Dashscope API 配置保留在这里，便于下次切回：
    # model = "qwen-vl-max-latest"
    # api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    # if not api_key:
    #     raise RuntimeError("Set DASHSCOPE_API_KEY or OPENAI_API_KEY before running inference.")
    # client = OpenAI(
    #     api_key=api_key,
    #     base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    # )
    # 推荐切回方式：设置 VLM_PROVIDER=qwen，可选 QWEN_MODEL=qwen-vl-max-latest。

    folder_path = 'dataset/videos'
    QA_df = pd.read_parquet('dataset/MCQ.parquet')

    folder_path_result = 'result'
    if not os.path.exists(folder_path_result):
        os.makedirs(folder_path_result)

    RUN_ID, RUN_MODE = resolve_run_config(folder_path_result, model, AGENTIC_GLOBAL_MODE)
    if RUN_ID == "legacy":
        res_path = os.path.join(folder_path_result, '%s_output.csv' % make_safe_run_id(model))
    else:
        res_path = result_csv_path(folder_path_result, model, RUN_ID)

    print("=" * 72)
    print(f"RUN_ID: {RUN_ID}")
    print(f"RUN_MODE: {RUN_MODE}")
    print(f"Agentic mode: {AGENTIC_GLOBAL_MODE}; all 16 categories; base frames per answer: {BASE_N_FRAMES}")
    print(f"Result CSV: {res_path}")
    print(f"Dashboard SQLite: result/dashboard_{RUN_ID}.sqlite3")
    print(f"Dashboard update URL: {DASHBOARD_UPDATE_URL}")
    print("=" * 72)

    if os.path.exists(res_path):
        old_res = pd.read_csv(res_path, index_col=0)
        res = copy.deepcopy(QA_df)
        for col in old_res.columns:
            if col not in res.columns:
                res[col] = None
            common_idx = old_res.index.intersection(res.index)
            res.loc[common_idx, col] = old_res.loc[common_idx, col]
        if 'Output' not in res.columns:
            res['Output'] = None
    else:
        res = copy.deepcopy(QA_df)
        res['Output'] = None
    if 'RetrievalStrategy' not in res.columns:
        res['RetrievalStrategy'] = None
    if 'RetrievalMeta' not in res.columns:
        res['RetrievalMeta'] = None
    if 'RunID' not in res.columns:
        res['RunID'] = RUN_ID
    else:
        res['RunID'] = res['RunID'].fillna(RUN_ID)

    available_videos = os.listdir(folder_path)
    video_ok_mask = res['video_id'].isin(available_videos)
    output_str = res['Output'].astype(str).str.strip()
    pending_mask = (
        res['Output'].isna() |
        (output_str == '') |
        output_str.str.contains(r'\[None\]', na=False)
    )
    valid_indices = res[video_ok_mask & pending_mask].index.tolist()
    # valid_indices 就是本次真正要跑的题号列表。
    # 已完成且不是 [None] 的题会被跳过，实现断点续跑。

    if os.path.exists(res_path) and 'answer' in res.columns:
        done = res[res['Output'].notna() &
                ~res['Output'].astype(str).str.contains(r'\[None\]', na=False)]
        _total_processed = len(done)
        _correct_count = (
            done['Output'].str.extract(r'\[([A-G])\]')[0].str.upper() ==
            done['answer'].astype(str).str.upper()
        ).sum()
    else:
        _total_processed = 0
        _correct_count   = 0

    print('Pending QA items:', len(valid_indices))
    print(f'Resume accuracy source: {res_path}')

    _total_processed_ref = [_total_processed]
    _correct_count_ref   = [_correct_count]

    print(f'Start concurrent processing, concurrency: {CONCURRENCY}')
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(
                process_one, qa_idx, res, res_path, folder_path, model, client,
                AGENTIC_GLOBAL_MODE,
                RUN_ID,
                _lock, _fatal_event,
                _total_processed_ref, _correct_count_ref
            ): qa_idx
            for qa_idx in valid_indices
        }
        for future in as_completed(futures):
            if _fatal_event.is_set():
                for f in futures:
                    f.cancel()
                break
            try:
                future.result()
            except Exception as e:
                print(f"Worker exception: {e}")
