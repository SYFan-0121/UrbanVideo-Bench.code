"""
agentic_retrieval.py
Agentic 检索式抽帧模块。

目标：在不把整段视频全部送给 VLM 的情况下，先用轻量检索找出可能有用的证据帧，
再组合成最终 n_frames 张 base64 JPEG 帧交给 run.py / confidence.py 使用。

单次 Agentic 抽帧链路：
1. 读取视频元信息，建立 _FrameCache，尽量复用同一次调用中的解码结果。
2. 抽 8 张 overview 缩略图，让 VLM 生成 retrieval plan。
3. build_candidate_pool 构造全局均匀帧、场景变化帧、锚点帧、尾段帧等候选池。
4. semantic_retrieve 用 CLIP / sentence-transformers 在内存中做图文语义召回。
5. vlm_rerank 让 VLM 对候选缩略图二次排序。
6. compose_frame_budget 把全局覆盖、场景变化、语义召回、VLM 重排和锚点帧拼成最终帧预算。

STRIVE 场景下，AgenticRetrievalContext 会复用 plan、候选池和语义召回结果，
不同 variant_id 只做窗口过滤、轻微扰动、rerank 和最终帧组合，减少重复 I/O。
"""

import base64
import hashlib
import json
import math
import os
import re
import threading
from typing import Any

import cv2
from video_locks import get_video_read_lock
from vlm_api import vlm_chat_completion

# 创建缓存目录，用于存储 Agentic 抽帧中间结果。
DEFAULT_CACHE_DIR = os.path.join("cache", "agentic_retrieval")

# 环境变量读取函数，用于控制视频解码并发等运行参数。
def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


# 控制视频解码的并发线程数 ，默认值为2，避免多线程 + STRIVE 时把磁盘和 CPU 解码打满。
_VIDEO_DECODE_CONCURRENCY = _env_int("AGENTIC_VIDEO_DECODE_CONCURRENCY", 2)
# 限制同时解码视频的线程数。有界信号量方式
_VIDEO_DECODE_SEMAPHORE = threading.BoundedSemaphore(_VIDEO_DECODE_CONCURRENCY)

# 为 agentic_retrieval 模块的打印信息加上统一的 模块名前缀 ，便于在控制台输出中快速识别日志来源。
def _agentic_log(message: str) -> None:
    print(f"[AgenticRetrieval] {message}", flush=True)

# 把视频名/问题名转换成可安全用于缓存目录的字符串。
def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_")[:80] or "item"

# 根据题目和类别生成稳定hash，用作该题在缓存系统中的唯一标识键。
def _question_hash(question: str, qa_category: str) -> str:
    raw = f"{qa_category}\n{question}".encode("utf-8", errors="ignore")
    return hashlib.md5(raw).hexdigest()[:12]


# 缓存目录创建函数，不存在则创建，存在则不操作。
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# 读取视频总帧数、FPS、宽高和时长；只读取元信息，不抽帧。并通过 双重锁机制 确保多线程环境下的安全访问。
def _read_video_meta(video_path: str) -> dict[str, Any]:
    # 第一层：信号量控制并发数，避免同时解码多个视频导致资源耗尽。
    with _VIDEO_DECODE_SEMAPHORE:
        # 第二层：文件级读锁
        with get_video_read_lock(video_path):
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video.release()
    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": total_frames / fps if fps > 0 else 0.0,
    }

# 均匀分布地选取 n 个帧索引
def _uniform_indices(total_frames: int, n: int, start_ratio: float = 0.0, end_ratio: float = 1.0) -> list[int]:
    if total_frames <= 0 or n <= 0:
        return []
    start = max(0, min(total_frames - 1, int(total_frames * start_ratio)))
    end = max(start, min(total_frames - 1, int(total_frames * end_ratio) - 1))
    if n == 1:
        return [(start + end) // 2]
    if end == start:
        return [start]
    step = (end - start) / (n - 1)
    return sorted({int(round(start + i * step)) for i in range(n)})

# 规频比例窗口归一化函数，确保 0 <= start < end <= 1。
def _normalize_ratio_window(start_ratio: float = 0.0, end_ratio: float = 1.0) -> tuple[float, float]:
    try:
        start = float(start_ratio)
    except (TypeError, ValueError):
        start = 0.0
    try:
        end = float(end_ratio)
    except (TypeError, ValueError):
        end = 1.0

    start = max(0.0, min(1.0, start))
    end = max(0.0, min(1.0, end))
    if end <= start:
        end = min(1.0, start + 1e-6)
    return start, end

# 把窗口比例转换成具体帧数。
def _frame_window_bounds(total_frames: int, start_ratio: float, end_ratio: float) -> tuple[int, int]:
    if total_frames <= 0:
        return 0, 0
    start_ratio, end_ratio = _normalize_ratio_window(start_ratio, end_ratio)
    start = max(0, min(total_frames - 1, int(total_frames * start_ratio)))
    end = max(start, min(total_frames - 1, int(total_frames * end_ratio) - 1))
    return start, end

# 过滤候选帧，只保留落在当前时间窗口内的帧下标。
# Agentic 类别默认全部以全视频 [0.0, 1.0] 作为检索范围，但在 Agentic STRIVE 模式里，传进 Agentic 的窗口会被扰动。
def _filter_indices_by_ratio(indices: list[int], total_frames: int, start_ratio: float, end_ratio: float) -> list[int]:
    start, end = _frame_window_bounds(total_frames, start_ratio, end_ratio)
    return sorted({int(idx) for idx in indices if start <= int(idx) <= end})

# 把比例窗口编码成缓存文件夹名的一部分。同一个视频 + 同一个时间窗口，永远对应同一个缓存文件名，实现缓存命中 。
def _window_cache_key(start_ratio: float, end_ratio: float) -> str:
    return f"w{int(round(start_ratio * 10000)):04d}_{int(round(end_ratio * 10000)):04d}"


def _add_timestamp_watermark(frame, frame_idx: int, total_frames: int, fps: float):
    height, width = frame.shape[:2]
    #帧索引 ÷ 帧率 = 对应时间（秒），如第 120 帧 / 60fps = 2.0秒
    timestamp = frame_idx / fps if fps > 0 else 0.0
    text = f"Frame {frame_idx + 1}/{total_frames} | t={timestamp:.1f}s"
    #设置文字字体、大小、粗细
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    #计算文字大小，避免文字超出视频边界
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    x = max(8, width - text_size[0] - 15)
    y = max(text_size[1] + 12, height - 15)
    overlay = frame.copy()
    #在视频上绘制文字背景矩形
    cv2.rectangle(overlay, (x - 8, y - text_size[1] - 8), (x + text_size[0] + 8, y + 8), (0, 0, 0), -1)
    #将文字背景矩形与原始视频合并
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    #在视频上绘制文字
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness)
    return frame

# 从已打开的视频对象中 seek 到指定帧并读取图像。
def _read_frame(video, frame_idx: int):
    video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = video.read()
    return frame if ok else None


# 单次 Agentic 调用内的帧缓存。
# overview、候选池、语义召回、rerank 和最终编码都会复用它，避免反复打开视频。
class _FrameCache:
    # 初始化帧缓存，记录视频路径和总帧数。
    def __init__(self, video_path: str, total_frames: int):
        self.video_path = video_path
        self.total_frames = max(0, int(total_frames or 0))
        self._video = None
        self._frames: dict[int, Any] = {}
        self.hits = 0
        self.misses = 0

    # 先查内存缓存，没命中才真正 seek 视频文件。
    def read(self, frame_idx: int):
        if self.total_frames <= 0:
            return None
        idx = max(0, min(self.total_frames - 1, int(frame_idx)))
        if idx in self._frames:
            self.hits += 1
            return self._frames[idx]

        self.misses += 1
        with _VIDEO_DECODE_SEMAPHORE:
            with get_video_read_lock(self.video_path):
                if self._video is None:
                    self._video = cv2.VideoCapture(self.video_path)
                    if not self._video.isOpened():
                        self._video.release()
                        self._video = None
                        return None
                frame = _read_frame(self._video, idx)

        if frame is not None:
            self._frames[idx] = frame
        return frame

    # 释放视频捕获对象占用的系统资源，避免文件句柄泄漏。
    def close(self):
        if self._video is not None:
            self._video.release()
            self._video = None
    
    # 返回缓存命中情况，写入 retrieval_meta 便于性能诊断。
    def stats(self) -> dict[str, int]:
        return {
            "unique_decoded_frames": len(self._frames),
            "cache_hits": self.hits,
            "cache_misses": self.misses,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# 把最终选中的原始帧编码成 base64 JPEG。
# 这些字符串会作为 image_url 输入发给 VLM。
def _encode_indices(video_path: str, indices: list[int], total_frames: int, fps: float, frame_cache: _FrameCache | None = None) -> list[str]:
    if frame_cache is not None:
        frames = []
        for idx in indices:
            frame = frame_cache.read(idx)
            if frame is None:
                continue
            frame = _add_timestamp_watermark(frame, idx, total_frames, fps)
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                frames.append(base64.b64encode(buf).decode("utf-8"))
        return frames

    with _VIDEO_DECODE_SEMAPHORE:
        with get_video_read_lock(video_path):
            video = cv2.VideoCapture(video_path)
            frames = []
            for idx in indices:
                frame = _read_frame(video, idx)
                if frame is None:
                    continue
                frame = _add_timestamp_watermark(frame, idx, total_frames, fps)
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    frames.append(base64.b64encode(buf).decode("utf-8"))
            video.release()
            return frames


# 把候选帧编码成小尺寸缩略图。
# Planner 和 rerank 只需要看缩略图，能显著降低 token 和网络负担。
def _encode_thumbnails(
    video_path: str,
    indices: list[int],
    total_frames: int,
    fps: float,
    max_side: int = 360,
    frame_cache: _FrameCache | None = None,
) -> list[dict]:
    if frame_cache is not None:
        encoded = []
        for idx in indices:
            frame = frame_cache.read(idx)
            if frame is None:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, max_side / max(height, width))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            frame = _add_timestamp_watermark(frame, idx, total_frames, fps)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                encoded.append({
                    "frame_idx": int(idx),
                    "timestamp": idx / fps if fps > 0 else 0.0,
                    "base64": base64.b64encode(buf).decode("utf-8"),
                })
        return encoded

    with _VIDEO_DECODE_SEMAPHORE:
        with get_video_read_lock(video_path):
            video = cv2.VideoCapture(video_path)
            encoded = []
            for idx in indices:
                frame = _read_frame(video, idx)
                if frame is None:
                    continue
                height, width = frame.shape[:2]
                scale = min(1.0, max_side / max(height, width))
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                frame = _add_timestamp_watermark(frame, idx, total_frames, fps)
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    encoded.append({
                        "frame_idx": int(idx),
                        "timestamp": idx / fps if fps > 0 else 0.0,
                        "base64": base64.b64encode(buf).decode("utf-8"),
                    })
            video.release()
            return encoded

#从一段 可能包含多个 JSON 对象的文本 中，提取出 最合适的 JSON 对象 。
def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    candidates = []
    # 遍历文本，查找所有可能的 JSON 对象起始位置。
    for start in (idx for idx, char in enumerate(text) if char == "{"):
        depth = 0   # 当前括号深度
        in_string = False   # 是否在字符串中
        escaped = False   # 前一个字符是否是转义符
        for pos in range(start, len(text)):
            char = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                # 当括号深度为 0 时，说明当前括号内的内容是一个完整的 JSON 对象。
                # 尝试解析为 JSON 对象，若成功则添加到候选列表。
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:pos + 1])
                    except Exception:
                        break
                    if isinstance(parsed, dict):
                        candidates.append(parsed)
                    break

    if not candidates:
        return None

    preferred_keys = (
        "selected_frame_indices",
        "visual_targets",
        "temporal_focus",
        "evidence_mix",
    )
    #倒序遍历 ，优先选择文本中靠后的 JSON（通常是最终结论）
    for candidate in reversed(candidates):
        #检查是否包含特定关键词，若包含则返回该 JSON
        if any(key in candidate for key in preferred_keys):
            return candidate
    # 如果没有包含特定关键词的 JSON，返回最后一个 JSON
    return candidates[-1]

# VLM planner 失败时的默认检索计划。
def _fallback_plan(question: str, qa_category: str) -> dict[str, Any]:
    category = str(qa_category)
    plan = {
        "visual_targets": [],     # 可视目标目标列表
        "temporal_focus": "full", # 时间焦点
        "needs_global_path": True,  # 是否需要全局路径
        "needs_final_state": False, # 是否需要最终状态
        "query_terms": [category, question[:120]], # 查询关键词
        # evidence_mix 表示最终帧预算中各类证据的大致配额：
        "evidence_mix": {
            "global": 8,       # 全局覆盖
            "scene_change": 8, # 场景变化
            "semantic": 10,    # 语义召回
            "rerank": 4,       # VLM 重排高分帧
            "anchors": 2,      # 起止/中段锚点帧    
        },
    }
    if category in {"Action Generation", "Goal Detection", "Progress Evaluation", "Landmark Position"}:
        plan["needs_final_state"] = True
        plan["temporal_focus"] = "late"
    if category in {"Cognitive Map", "High-level Planning", "Trajectory Captioning", "Sequence Recall"}:
        plan["needs_global_path"] = True
        plan["temporal_focus"] = "full"
    return plan


# Agentic 的第一步：让 VLM 读 overview 帧并生成检索计划。
# 计划会告诉后续流程该找什么目标、用什么关键词、各类证据占多少帧预算。
def make_retrieval_plan(
    client,
    model: str,
    question: str,
    qa_category: str,
    overview_frames: list[dict],
    cache_path: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    # 从缓存中读取计划，若命中缓存则直接返回。
    if cache_path and os.path.exists(cache_path) and not force_refresh:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Planner 失败时不会中断流程。
    # fallback 计划保证候选池和最终帧预算仍然能继续执行。
    fallback = _fallback_plan(question, qa_category)
    if client is None or not overview_frames:
        return fallback

    prompt = (
        "You are a retrieval planner for a first-person drone video QA benchmark.\n"
        "Given the question, category, answer options, and overview frames, create a frame retrieval plan.\n"
        "Return ONLY JSON with these keys:\n"
        "{\n"
        '  "visual_targets": ["short target phrases"],\n'
        '  "temporal_focus": "start|middle|late|full|multi",\n'
        '  "needs_global_path": true/false,\n'
        '  "needs_final_state": true/false,\n'
        '  "query_terms": ["terms for image-text retrieval"],\n'
        '  "evidence_mix": {"global": 8, "scene_change": 8, "semantic": 10, "rerank": 4, "anchors": 2}\n'
        "}\n\n"
        f"Category: {qa_category}\n"
        f"Question:\n{question}\n"
    )
    content = [{"type": "text", "text": prompt}]
    for item in overview_frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{item['base64']}"}})

    try:
        result = vlm_chat_completion(
            client,
            model,
            messages=[{"role": "user", "content": content}],
            timeout=90,
            max_tokens=280,
        )
        # 从 VLM 回复中提取 JSON 对象。
        parsed = _extract_json_object(result.choices[0].message.content)
        if parsed:
            fallback.update(parsed)
    except Exception:
        pass

    # 缓存计划，避免重复请求 VLM。
    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(fallback, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return fallback


# 构建基础候选池。
# 它不理解题目，只从视频结构出发找全局帧、场景变化帧、锚点帧和尾段帧。
def build_candidate_pool(
    video_path: str, 
    meta: dict[str, Any],  # 视频元数据，包含 total_frames 和 fps。
    cache_dir: str,  # 缓存目录，用于存储候选帧池。
    force_refresh: bool = False,  # True 时忽略已有缓存，重新生成候选帧池。
    frame_cache: _FrameCache | None = None,  # 视频帧缓存，用于快速读取视频帧。
) -> dict[str, list[int]]:
    # 从缓存中读取候选帧池，若命中缓存则直接返回。
    cache_path = os.path.join(cache_dir, "candidates.json")
    if os.path.exists(cache_path) and not force_refresh:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: [int(x) for x in v] for k, v in data.items()}
        except Exception:
            pass

    total = int(meta["total_frames"])
    fps = float(meta.get("fps") or 0.0)
    global_dense = _uniform_indices(total, min(96, max(16, total)))   # 较密的全局均匀采样，用于场景变化检测和语义召回候选。
    global_budget = _uniform_indices(total, 16)                       # 较稀疏的全局覆盖帧，保证最终证据不只看局部。
    start_anchor = _uniform_indices(total, 3, 0.0, 0.12)              # 开头锚点，用于保留路径结构。
    middle_anchor = _uniform_indices(total, 4, 0.42, 0.58)            # 中段锚点，用于保留路径结构。
    end_anchor = _uniform_indices(total, 5, 0.82, 1.0)               # 结尾锚点，用于保留路径结构。 

    # 场景变化检测：在 global_dense 上计算相邻灰度缩略图的平均差值，差值越大越可能是关键转场。
    if frame_cache is not None:
        prev_gray = None  # 上一帧的灰度图，用于计算帧差。
        scene_scores = []
        for idx in global_dense:
            frame = frame_cache.read(idx)
            if frame is None:
                continue
            # 缩略图，用于计算灰度图的平均差值。
            small = cv2.resize(frame, (160, 90))
            # 转灰度图，用于计算帧差。
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                # 计算当前帧与上一帧的灰度图的平均差值。
                score = float(cv2.mean(cv2.absdiff(gray, prev_gray))[0])
                # 记录当前帧与上一帧的灰度图的平均差值。
                scene_scores.append((score, idx))
            prev_gray = gray
    else:
        with _VIDEO_DECODE_SEMAPHORE:
            with get_video_read_lock(video_path):
                video = cv2.VideoCapture(video_path)
                prev_gray = None
                scene_scores = []
                for idx in global_dense:
                    frame = _read_frame(video, idx)
                    if frame is None:
                        continue
                    small = cv2.resize(frame, (160, 90))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        score = float(cv2.mean(cv2.absdiff(gray, prev_gray))[0])
                        scene_scores.append((score, idx))
                    prev_gray = gray
                video.release()
    # 从场景变化检测结果中选出前 32 个帧，作为场景变化帧。
    scene_change = [idx for _, idx in sorted(scene_scores, reverse=True)[:32]]
    #去重并排序
    scene_change = sorted(set(scene_change))

    candidates = {
        "global_dense": global_dense,
        "global": global_budget,
        "scene_change": scene_change,
        "anchors": sorted(set(start_anchor + middle_anchor + end_anchor)),
        "tail": _uniform_indices(total, 12, 0.65, 1.0),
    }
    #将生成的 候选帧池 保存到 缓存文件 ，供后续复用。
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return candidates


# 语义召回阶段。
# 用 CLIP 把候选帧和 query_terms 映射到同一向量空间，再按相似度选出更相关的帧。
def semantic_retrieve(
    video_path: str,
    candidate_indices: list[int],
    query_terms: list[str],         # planner 给出的文本检索词，会拼接成 CLIP 文本查询。
    cache_dir: str,
    top_k: int = 24,
    frame_cache: _FrameCache | None = None,
) -> tuple[list[int], dict[str, Any]]:
    #初始化元信息字典
    # enabled 是否启用语义检索（初始为 False，成功后设为 True）
    # backend 语义检索模型名称（初始为 None，成功后设为模型名）
    # error 错误信息（初始为 None，失败时设为错误信息）
    info = {"enabled": False, "backend": None, "error": None}
    # 检查环境变量：如果设置了 AGENTIC_DISABLE_SEMANTIC=1，直接禁用语义检索
    if os.environ.get("AGENTIC_DISABLE_SEMANTIC", "").strip() == "1":
        info["error"] = "disabled by AGENTIC_DISABLE_SEMANTIC=1" 
        return [], info    # 返回空结果和元信息
    # 如果没有候选帧 或 没有检索词，直接返回空结果
    if not candidate_indices or not query_terms:
        return [], info
    # 检查是否允许从HuggingFace下载模型
    allow_download = os.environ.get("AGENTIC_SEMANTIC_ALLOW_DOWNLOAD", "").strip() == "1"
    # # 如果不允许下载模型，设置离线模式：不允许 semantic retrieval 自动联网下载 Hugging Face / Transformers 模型文件，避免运行时突然联网、卡住、失败或下载大文件。
    if not allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        # PIL：图像处理库，用于将numpy数组转为PIL图像
        from PIL import Image
        # SentenceTransformer：加载CLIP模型，编码文本/图像特征
        # util：工具函数，计算余弦相似度
        from sentence_transformers import SentenceTransformer, util
    # 依赖库未安装/不可用，记录错误并返回空结果
    except Exception as exc:
        info["error"] = f"sentence_transformers/PIL unavailable: {exc}"
        return [], info

    try:
        # 从环境变量获取模型名，默认使用 clip-ViT-B-32（通用CLIP模型）
        model_name = os.environ.get("AGENTIC_RETRIEVAL_EMBED_MODEL", "clip-ViT-B-32")
        embedder_kwargs = {} # 模型初始化参数
        retrieval_device = os.environ.get("AGENTIC_RETRIEVAL_DEVICE", "").strip()
        if retrieval_device:
            # 可在共享 GPU 服务器上把 CLIP 固定到 CPU，避免与 VLM 额外争显存。
            embedder_kwargs["device"] = retrieval_device
        # 不允许下载模型时，设置仅使用本地文件加载模型
        if not allow_download:
            embedder_kwargs["local_files_only"] = True
        try:
            # 加载CLIP模型：优先带离线参数加载，失败则用默认方式加载
            embedder = SentenceTransformer(model_name, **embedder_kwargs)
        except TypeError:
            # 兼容不接受 local_files_only 的旧版本；保留显式设备设置。
            fallback_kwargs = dict(embedder_kwargs)
            fallback_kwargs.pop("local_files_only", None)
            embedder = SentenceTransformer(model_name, **fallback_kwargs)
        # 模型加载成功，更新元信息：启用检索 + 记录模型名称
        info["enabled"] = True
        info["backend"] = model_name

        # 定义变量：有效帧序号、有效帧图像、内部创建的缓存对象
        valid_indices = []
        images = []
        owned_cache = None
        # 如果外部没有传入帧缓存对象，内部创建一个并返回给外部
        if frame_cache is None:
            meta = _read_video_meta(video_path)
            owned_cache = _FrameCache(video_path, meta["total_frames"])   # 初始化帧缓存
            frame_cache = owned_cache   # 返回内部创建的缓存对象给外部

        try:
            # 遍历所有候选帧序号，读取并预处理视频帧，将有效帧序号和图像添加到列表中
            for idx in candidate_indices:
                # 读取视频帧
                frame = frame_cache.read(idx)
                if frame is None:
                    continue
                # opencv读取的帧是BGR格式，CLIP需要RGB格式，转换颜色空间为RGB格式
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 将numpy数组转为PIL图像（CLIP模型要求的输入格式）
                images.append(Image.fromarray(rgb))
                # 记录有效帧序号
                valid_indices.append(idx)
        finally:
            # 关闭内部创建的帧缓存对象
            if owned_cache is not None:
                owned_cache.close()

        # 没有读取到有效帧，返回空结果
        if not images:
            return [], info

        # 1.对有效帧图像进行编码，得到图像特征向量（tensor格式）
        image_emb = embedder.encode(
            images,                        # 有效帧图像列表
            convert_to_tensor=True,        # 转换为tensor格式
            show_progress_bar=False        # 不显示进度条，提高效率
            )
        # 2. 对检索词进行编码，得到文本特征向量（tensor格式）
        text_emb = embedder.encode(
            " ".join(query_terms),       # 检索词列表，合并为一个字符串
             convert_to_tensor=True,        # 转换为tensor格式
             show_progress_bar=False        # 不显示进度条，提高效率
             )
        # 3. 计算文本特征向量与图像特征向量的余弦相似度（语义相似度）
        scores = util.cos_sim(text_emb, image_emb)[0]
        # 4. 对相似度进行排序
        ranked = sorted(zip(scores.tolist(), valid_indices), reverse=True)
        # 5. 返回前 top_k 个相似度最高的有效帧序号
        return [idx for _, idx in ranked[:top_k]], info
    # 捕获并记录异常，返回空结果
    except Exception as exc:
        info["error"] = str(exc)
        return [], info


# VLM 重排阶段。
# 它把候选缩略图和完整题目交给多模态模型，让模型挑出最能支持答案的 frame_idx。
def vlm_rerank(
    client,
    model: str,
    question: str,
    qa_category: str,
    plan: dict[str, Any],          # 检索计划，包含检索参数和上下文
    thumbnails: list[dict],        # 候选缩略图列表，每个元素包含 frame_idx、timestamp、base64 编码的图像
    top_k: int = 12,               # 返回的证据帧数量，默认12个
) -> tuple[list[int], dict[str, Any]]:
    # 初始化元信息，包括是否启用、原始响应、错误信息
    info = {"enabled": False, "raw": None, "error": None}
    # 检查输入参数是否有效
    if client is None or not thumbnails:
        return [], info

    prompt = (
        "You are selecting evidence frames for drone-video question answering.\n"
        "The images are candidate frames in chronological order. Select the frame_idx values that are most useful evidence.\n"
        "Prefer frames that directly support answering the question, but keep path/context diversity.\n"
        "Return ONLY JSON: {\"selected_frame_indices\": [int, ...], \"reason\": \"brief\"}\n\n"
        f"Category: {qa_category}\n"
        f"Question:\n{question}\n\n"
        f"Retrieval plan:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
        "Candidates:\n"
        + "\n".join([f"frame_idx={x['frame_idx']}, t={x['timestamp']:.1f}s" for x in thumbnails])
    )
    content = [{"type": "text", "text": prompt}]
    # 遍历所有候选缩略图，将每个缩略图的base64编码添加到内容列表中
    for item in thumbnails:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{item['base64']}"}})

    try:
        result = vlm_chat_completion(
            client,
            model,
            messages=[{"role": "user", "content": content}],
            timeout=120,
            max_tokens=240,
        )
        # result.choices[0]：取第一个回答
        # message.content：模型输出的原始文本
        raw = result.choices[0].message.content
        info["enabled"] = True         # 标记：VLM 调用成功
        info["raw"] = raw              # 把模型原始回答存到日志/信息里，方便调试
        # _extract_json_object 是工具函数：从文本里自动截取 JSON 字符串
        parsed = _extract_json_object(raw) or {}
        # 从解析后的 JSON 里，取出模型选中的帧序号列表
        selected = parsed.get("selected_frame_indices", [])
        # 确保模型选中的帧序号在有效范围内
        #x["frame_idx"] for x in thumbnails：从候选缩略图列表里提取所有帧序号
        # max((x["frame_idx"] for x in thumbnails), default=-1) + 1：找到所有帧序号中的最大值，加1
        # _coerce_frame_indices 是工具函数：确保帧序号在有效范围内，避免超出视频范围。过滤掉超出范围的索引，确保索引是非负整数，去重
        selected = _coerce_frame_indices(selected, max((x["frame_idx"] for x in thumbnails), default=-1) + 1)
        return selected[:top_k], info
    except Exception as exc:
        info["error"] = str(exc)
        return [], info

#去除相邻过近的索引，保证相邻索引的差值至少为 min_gap
def _dedupe_close(indices: list[int], min_gap: int) -> list[int]:
    selected = []
    # 对去重后的索引列表进行排序，确保相邻索引是有序的
    for idx in sorted(set(int(x) for x in indices)):
        # 如果是第一个索引，或者当前索引与上一个选中索引的差值 >= 最小间隔，直接添加到结果列表
        if not selected or idx - selected[-1] >= min_gap:
            selected.append(idx)
    return selected

#从一堆帧序号里，均匀选出固定数量的帧序号，维持时间覆盖
def _limit_evenly(indices: list[int], n: int) -> list[int]:
    indices = sorted(set(int(x) for x in indices))
    #如果本来就不多于 n 个，直接返回原列表
    if len(indices) <= n:
        return indices
    #如果 n 为 1，直接返回中间帧
    if n <= 1:
        return [indices[len(indices) // 2]]
    # 计算步长
    step = (len(indices) - 1) / (n - 1)
    #按步长均匀取帧，并返回排序后的结果
    return sorted({indices[int(round(i * step))] for i in range(n)})

#根据一个编号（variant_id），把帧序号列表 “旋转 / 滚动” 一下，让不同变体拿到不同的起始位置，避免每次都从同一个帧开始筛选，增加多样性。
#让不同 variant_id 的 STRIVE 投票在中间证据候选上产生轻微差异，用来检验答案是否稳定，而不是三次都看完全一样的候选帧。
def _rotate_for_variant(indices: list[int], variant_id: int) -> list[int]:
    # 对输入列表进行去重和排序，确保相邻索引是有序的
    indices = sorted(set(int(x) for x in indices))
    #如果本来就不多于 1 个，直接返回原列表
    if len(indices) <= 1:
        return indices
    # 计算旋转偏移量
    offset = int(variant_id) % len(indices)
    # 旋转列表
    return indices[offset:] + indices[:offset]

#合并多路检索结果（语义检索 + VLM 选帧）时，去掉重复帧，同时保留优先级顺序。
def _ordered_unique(indices: list[int]) -> list[int]:
    """保持输入顺序的去重函数，用于合并多路候选。"""
    seen = set()        # 用于记录已经遇到的索引，避免重复添加
    unique = []         # 存储最终去重、保序的结果
    for idx in indices:
        idx = int(idx)
        #判断当前帧号是否已经出现过，如果出现过，直接跳过，否则添加到结果列表中
        if idx in seen:
            continue
        seen.add(idx)
        unique.append(idx)
    return unique

#根据当前视频长度、有效窗口大小，计算一个帧偏移量（gap），用于给不同变体（variant）做帧位置小扰动
def _variant_neighbor_gap(total_frames: int, n_frames: int, start_ratio: float, end_ratio: float) -> int:
    window_start, window_end = _frame_window_bounds(total_frames, start_ratio, end_ratio)
    window_span = max(1, window_end - window_start + 1)
    return max(1, min(max(1, window_span // max(n_frames * 4, 1)), max(1, total_frames // 80)))


def _perturb_indices_by_variant(
    indices: list[int],
    total_frames: int,
    variant_id: int,
    start_ratio: float,
    end_ratio: float,
    n_frames: int,
) -> list[int]:
    # 对输入列表进行去重和排序，确保相邻索引是有序的
    indices = _filter_indices_by_ratio(_coerce_frame_indices(indices, total_frames), total_frames, start_ratio, end_ratio)
    if not indices:
        return []

    # variant_id 对 3 取模 → 永远得到 0、1、2 三种模式
    phase = int(variant_id) % 3
    if phase == 0:
        return indices

    #计算帧偏移量（gap），用于给不同变体（variant）做帧位置小扰动
    gap = _variant_neighbor_gap(total_frames, n_frames, start_ratio, end_ratio)
    # phase=1 → 方向=-1 → 向前偏移
    # phase=2 → 方向=1 → 向后偏移
    direction = -1 if phase == 1 else 1
    window_start, window_end = _frame_window_bounds(total_frames, start_ratio, end_ratio)
    # 对每一个帧 idx：
    # 1. idx + direction * gap → 向前/向后挪 gap 帧
    # 2. max(window_start, ...) → 不能偏移到窗口左边以外
    # 3. min(window_end, ...) → 不能偏移到窗口右边以外
    shifted = [max(window_start, min(window_end, idx + direction * gap)) for idx in indices]
    return _filter_indices_by_ratio(shifted, total_frames, start_ratio, end_ratio)

#清洗 VLM/缓存/外部输入给出的帧号，转成合法整数下标。
def _coerce_frame_indices(indices, total_frames: int) -> list[int]:
    if total_frames <= 0 or not indices:
        return []
    cleaned = []
    for value in indices:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(float(idx)):
            continue
        cleaned.append(max(0, min(total_frames - 1, idx)))
    return cleaned


# 最终帧预算融合。
# 把全局覆盖、场景变化、语义召回、VLM 重排和锚点帧合并成最终 n_frames 张证据帧。
def compose_frame_budget(
    pools: dict[str, list[int]],        # 全局/场景变化/锚点/尾段候选
    semantic_hits: list[int],           # 语义召回帧
    rerank_hits: list[int],             # VLM rerank 认为最有证据价值的帧
    plan: dict[str, Any],               # planner 给出的 evidence_mix，控制各来源的大致配额
    total_frames: int,                  # 视频总帧数
    n_frames: int,                      # 最终证据帧数量
    variant_id: int = 0,                # STRIVE 变体 ID，用于帧位置小扰动
    start_ratio: float = 0.0,           # 有效窗口开始比例
    end_ratio: float = 1.0,             # 有效窗口结束比例
) -> list[int]:
    try:
        total_frames = int(total_frames)
    except (TypeError, ValueError):
        return []
    try:
        n_frames = int(n_frames)
    except (TypeError, ValueError):
        return []
    if total_frames <= 0 or n_frames <= 0:
        return []
    start_ratio, end_ratio = _normalize_ratio_window(start_ratio, end_ratio)

    # 清洗并过滤候选帧，确保它们在有效窗口内。
    pools = pools or {}
    pools = {
        key: _filter_indices_by_ratio(_coerce_frame_indices(value, total_frames), total_frames, start_ratio, end_ratio)
        for key, value in pools.items()
    }
    # 清洗并过滤语义召回帧和 VLM rerank 认为最有证据价值的帧，确保它们在有效窗口内。
    semantic_hits = _filter_indices_by_ratio(_coerce_frame_indices(semantic_hits, total_frames), total_frames, start_ratio, end_ratio)
    rerank_hits = _filter_indices_by_ratio(_coerce_frame_indices(rerank_hits, total_frames), total_frames, start_ratio, end_ratio)

    # evidence_mix 是 planner 对最终帧预算的建议配比。
    mix = plan.get("evidence_mix") or {}
    q_global = int(mix.get("global", 8))
    q_scene = int(mix.get("scene_change", 8))
    q_semantic = int(mix.get("semantic", 10))
    q_rerank = int(mix.get("rerank", 4))
    q_anchors = int(mix.get("anchors", 2))

    # shift 让不同 STRIVE 变体在候选列表上错开，避免三票证据完全相同。
    shift = max(0, int(variant_id)) % 3
    # 从当前时间窗口里均匀采样全局覆盖帧。
    #q_global 是 planner 建议的 global 帧配额，比如 8。加上 shift 后，不同 variant 会采不同数量：
    #variant 0：采 q_global
    #variant 1：采 q_global + 1
    #variant 2：采 q_global + 2
    global_pick = _uniform_indices(total_frames, q_global + shift, start_ratio, end_ratio)
    # 从已有的 anchors 和 tail 候选里挑锚点帧。
    anchor_pick = _limit_evenly(pools.get("anchors", []) + pools.get("tail", []), max(q_anchors, 0))
    # 从场景变化候选帧里挑帧。这个写法有两个扰动：先取 q_scene + shift 个候选。再用 [shift:] 跳过前面 shift 个。
    scene_pick = _limit_evenly(pools.get("scene_change", []), q_scene + shift)[shift:]
    # 对语义召回帧做变体扰动
    semantic_variant_hits = _perturb_indices_by_variant(
        semantic_hits, total_frames, variant_id, start_ratio, end_ratio, n_frames
    )
    # 对 VLM rerank 认为最有证据价值的帧做变体扰动
    rerank_variant_hits = _perturb_indices_by_variant(
        rerank_hits, total_frames, variant_id, start_ratio, end_ratio, n_frames
    )
    # 从扰动后的语义命中帧里保留一部分。
    # 逻辑和 scene_pick 类似：
    #从 semantic_variant_hits 中均匀保留 q_semantic + shift 个。
    #再跳过前 shift 个。
    #这样不同 STRIVE variant 的语义证据不会完全一样。
    semantic_pick = _limit_evenly(semantic_variant_hits, q_semantic + shift)[shift:]
    # 从扰动后的 rerank 命中帧里保留一部分
    rerank_pick = _limit_evenly(rerank_variant_hits, q_rerank + shift)[shift:]

    # 合并顺序：锚点 -> 全局覆盖 -> 场景变化 -> 语义召回 -> VLM 重排。
    selected = []
    selected.extend(anchor_pick)
    selected.extend(global_pick)
    selected.extend(scene_pick)
    selected.extend(semantic_pick)
    selected.extend(rerank_pick)

    # 对 rerank 高价值帧补充前后邻域，给模型一点局部上下文。
    window_start, window_end = _frame_window_bounds(total_frames, start_ratio, end_ratio)
    #遍历 VLM 精挑的前 4 个最优帧
    for idx in rerank_pick[:4]:
        # 计算上下文间隔：至少 1 帧，默认取总帧数的 1/160
        context_gap = max(1, total_frames // 160)
        # 补充上下文帧
        selected.extend([max(window_start, idx - context_gap), min(window_end, idx + context_gap)])
    # 确保所有帧都在有效窗口内。
    selected = _filter_indices_by_ratio(selected, total_frames, start_ratio, end_ratio)
    window_span = max(1, window_end - window_start + 1)
    # 窗口越长、要选的帧越多，间隔越小
    # 保证帧不会挤在一起
    min_gap = max(1, window_span // max(n_frames * 6, 1))
    # 去重，确保每个帧之间的间隔大于 min_gap
    selected = _dedupe_close(selected, min_gap=min_gap)

    #生成3倍的候选帧，如果不足则用当前窗口内的均匀帧补齐
    filler_pool = _uniform_indices(total_frames, n_frames * 3, start_ratio, end_ratio)
    for idx in filler_pool:
        if len(selected) >= n_frames:
            break
        # 确保候选帧与已选帧之间的间隔大于 min_gap
        if all(abs(idx - existing) >= min_gap for existing in selected):
            selected.append(idx)

    if len(selected) < n_frames:
        for idx in _uniform_indices(total_frames, n_frames * 2, start_ratio, end_ratio):
            if len(selected) >= n_frames:
                break
            if idx not in selected:
                selected.append(idx)

    return _limit_evenly(selected, n_frames)


# Agentic 共享上下文。
# 在 STRIVE K=3 中，三个变体共享 planner、候选池、语义召回和 FrameCache。
class AgenticRetrievalContext:
    def __init__(
        self,
        video_path: str,
        question: str,
        qa_category: str,
        client=None,
        model: str | None = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
        force_refresh: bool = False,
    ):
        # 把传入的参数全部存为成员变量，全局可用
        self.video_path = video_path
        self.question = question
        self.qa_category = qa_category
        self.client = client
        self.model = model
        self.cache_dir = cache_dir
        self.force_refresh = force_refresh

        # 从视频路径中提取“安全文件名”（去掉特殊字符、扩展名）
        self.video_name = _safe_name(os.path.splitext(os.path.basename(video_path))[0])
        # 计算题目哈希值，用于缓存目录名
        self.q_hash = _question_hash(question, qa_category)
        # question_cache: 与具体题目相关的缓存，存储：planner 计划、变体信息、问题相关结果。
        # shared_cache: 同一视频可复用的缓存，存储：视频帧、候选帧、通用特征。
        self.question_cache = os.path.join(cache_dir, self.video_name, self.q_hash)
        self.shared_cache = os.path.join(cache_dir, self.video_name, "_shared")
        _ensure_dir(self.question_cache)
        _ensure_dir(self.shared_cache)

        # 读取视频元数据（如总帧数、帧率等）
        self.meta = _read_video_meta(video_path)
        # 初始化 FrameCache，用于缓存视频帧
        self.frame_cache = _FrameCache(video_path, self.meta["total_frames"])

        self.plan: dict[str, Any] | None = None               # VLM planner 输出。
        self.pools: dict[str, list[int]] | None = None        # 候选池。
        self.semantic_hits: list[int] | None = None           # 全视频语义召回命中帧。
        self.semantic_info: dict[str, Any] | None = None      # 语义召回后端/错误/缓存信息。

    def close(self):
        self.frame_cache.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    #生成视频概览 → 让 VLM 做检索计划 → 构建多路候选帧池 → 执行语义检索。确保共享检索阶段已经完成。
    def _ensure_shared_retrieval(self):
        # 如果已经完成共享检索阶段，直接返回
        if self.plan is not None and self.pools is not None and self.semantic_hits is not None:
            _agentic_log(
                f"Reuse shared retrieval context for video={self.video_name}, category={self.qa_category}."
            )
            return

        #开始初始化
        _agentic_log(
            f"Start shared retrieval setup: video={self.video_name}, category={self.qa_category}, "
            f"total_frames={self.meta['total_frames']}, fps={self.meta['fps']:.2f}"
        )
        # 从视频**从头到尾均匀采8帧**，作为视频概览
        overview_indices = _uniform_indices(self.meta["total_frames"], 8, 0.0, 1.0)
        _agentic_log(f"Encoding 8 overview thumbnails for planner: indices={overview_indices}")
        # 把这8帧转成缩略图传给大模型
        overview = _encode_thumbnails(
            self.video_path,
            overview_indices,
            self.meta["total_frames"],
            self.meta["fps"],
            frame_cache=self.frame_cache,
        )
        # 让大模型根据视频概览生成检索计划
        _agentic_log("Sending overview thumbnails to planner VLM and waiting for retrieval plan...")
        # 调用大模型生成检索计划
        self.plan = make_retrieval_plan(
            client=self.client,
            model=self.model,
            question=self.question,
            qa_category=self.qa_category,
            overview_frames=overview,
            cache_path=os.path.join(self.question_cache, "plan.json"),
            force_refresh=self.force_refresh,
        )
        _agentic_log(
            f"Planner finished. intent={self.plan.get('intent')}, "
            f"query_terms={self.plan.get('query_terms')}"
        )
        # 构建多路候选帧池
         # 包括：全局均匀帧、稠密采样帧、场景切换帧、关键锚点帧、尾部帧
        _agentic_log("Building candidate pools from cached video frames...")
        self.pools = build_candidate_pool(
            self.video_path,
            self.meta,
            self.shared_cache,
            force_refresh=self.force_refresh,
            frame_cache=self.frame_cache,
        )
        _agentic_log(
            "Candidate pools ready: "
            + ", ".join(f"{key}={len(value)}" for key, value in (self.pools or {}).items())
        )
        # 执行语义检索
         # 从计划里取检索词，如果没有就用问题+类别
        query_terms = self.plan.get("query_terms") or [self.qa_category, self.question]
        # 语义检索的候选来源：全局稠密帧 + 场景切换帧 + 尾部帧
        semantic_candidates = sorted(set(
            self.pools.get("global_dense", [])
            + self.pools.get("scene_change", [])
            + self.pools.get("tail", [])
        ))
        _agentic_log(
            f"Running semantic retrieval over {len(semantic_candidates)} candidates; "
            f"query_terms={[str(x) for x in query_terms]}"
        )
        # 用CLIP模型，文本搜索匹配的视频帧
        self.semantic_hits, self.semantic_info = semantic_retrieve(
            video_path=self.video_path,
            candidate_indices=semantic_candidates,
            query_terms=[str(x) for x in query_terms],
            cache_dir=self.shared_cache,
            top_k=64,
            frame_cache=self.frame_cache,
        )
        _agentic_log(
            f"Semantic retrieval finished. hits={len(self.semantic_hits or [])}, "
            f"backend={(self.semantic_info or {}).get('backend')}, "
            f"cache_hit={(self.semantic_info or {}).get('cache_hit')}"
        )

    # 为某一个 Agentic/STRIVE 变体生成最终帧。
    # start_ratio/end_ratio 会在这里真正裁剪候选，保证时域扰动有效。
    def sample(
        self,
        n_frames: int = 32,
        variant_id: int = 0,
        start_ratio: float = 0.0,
        end_ratio: float = 1.0,
    ) -> tuple[list[str], dict[str, Any]]:
        # 先确保共享检索上下文已经准备好。
        # 再归一化当前 STRIVE 传入的时间窗口。
        self._ensure_shared_retrieval()
        start_ratio, end_ratio = _normalize_ratio_window(start_ratio, end_ratio)
        _agentic_log(
            f"Start variant sampling: variant_id={variant_id}, window=[{start_ratio:.2f}, {end_ratio:.2f}], "
            f"n_frames={n_frames}"
        )
        #为这个【变体+时间窗口】创建专属缓存目录
        window_key = _window_cache_key(start_ratio, end_ratio)
        item_cache = os.path.join(self.question_cache, f"variant_{variant_id}_{window_key}")
        _ensure_dir(item_cache)

        # 只保留在 [start_ratio, end_ratio] 时间段内的帧
        windowed_pools = {
            key: _filter_indices_by_ratio(value, self.meta["total_frames"], start_ratio, end_ratio)
            for key, value in (self.pools or {}).items()
        }
        windowed_semantic_hits = _filter_indices_by_ratio(
            self.semantic_hits or [],
            self.meta["total_frames"],
            start_ratio,
            end_ratio,
        )
        _agentic_log(
            "Window filtering finished: "
            + ", ".join(f"{key}={len(value)}" for key, value in windowed_pools.items())
            + f", semantic_hits={len(windowed_semantic_hits)}"
        )
        # 对语义召回结果应用 variant 扰动，避免 K=3 语义证据完全一致。
        semantic_candidate_hits = _perturb_indices_by_variant(
            windowed_semantic_hits,
            self.meta["total_frames"],
            variant_id,
            start_ratio,
            end_ratio,
            n_frames,
        )

        # VLM rerank 的候选来自多路召回，最多保留 42 张缩略图控制 token 成本。
        rerank_candidates = _ordered_unique(
            _rotate_for_variant(windowed_pools.get("global", []), variant_id)[:10]
            + _rotate_for_variant(windowed_pools.get("scene_change", []), variant_id)[:18]
            + semantic_candidate_hits[:18]
            + _rotate_for_variant(windowed_pools.get("tail", []), variant_id)[:8]
            + _rotate_for_variant(windowed_pools.get("anchors", []), variant_id)[:8]
        )[:42]
        _agentic_log(
            f"Preparing VLM rerank: candidates={len(rerank_candidates)}, variant_id={variant_id}"
        )
        # 把帧编号转成缩略图
        rerank_thumbs = _encode_thumbnails(
            self.video_path,
            rerank_candidates,
            self.meta["total_frames"],
            self.meta["fps"],
            max_side=320,
            frame_cache=self.frame_cache,
        )
        _agentic_log("Sending rerank thumbnails to VLM and waiting for ranked evidence frames...")
        # 发给 VLM 大模型打分排序，返回 16 个证据帧
        rerank_hits, rerank_info = vlm_rerank(
            client=self.client,
            model=self.model,
            question=self.question,
            qa_category=self.qa_category,
            plan=self.plan or {},
            thumbnails=rerank_thumbs,
            top_k=16,
        )
        _agentic_log(
            f"Rerank finished. hits={len(rerank_hits)}, "
            f"cache_hit={(rerank_info or {}).get('cache_hit')}"
        )
        # rerank 命中也记录扰动版本，便于 retrieval_meta 对比原始命中和变体命中。
        rerank_variant_hits = _perturb_indices_by_variant(
            rerank_hits,
            self.meta["total_frames"],
            variant_id,
            start_ratio,
            end_ratio,
            n_frames,
        )

        # 最终帧预算组合：把全局覆盖、场景变化、语义召回、rerank、锚点帧合并为 n_frames。
        selected_indices = compose_frame_budget(
            pools=windowed_pools,
            semantic_hits=windowed_semantic_hits,
            rerank_hits=rerank_hits,
            plan=self.plan or {},
            total_frames=self.meta["total_frames"],
            n_frames=n_frames,
            variant_id=variant_id,
            start_ratio=start_ratio,
            end_ratio=end_ratio,
        )
        _agentic_log(
            f"Composed final frame budget: selected={len(selected_indices)}, "
            f"indices={selected_indices}"
        )
        # 根据最终选中的序号，取出高清帧并转 Base64 字符串
        frames = _encode_indices(
            self.video_path,
            selected_indices,
            self.meta["total_frames"],
            self.meta["fps"],
            frame_cache=self.frame_cache,
        )
        _agentic_log(f"Final base64 encoding finished: frames={len(frames)}")
        window_start_idx, window_end_idx = _frame_window_bounds(
            self.meta["total_frames"], start_ratio, end_ratio
        )
        selected_within_window = all(
            window_start_idx <= idx <= window_end_idx for idx in selected_indices
        )

        # retrieval_meta 会写入 CSV / json 缓存，是排查抽帧行为的主要依据。
        # retrieval_meta 会写入 CSV/SQLite，也会给前端展示。
        # 它记录本次 Agentic 到底用了哪些证据和窗口。
        retrieval_meta = {
            "strategy": "agentic_retrieval",
            "variant_id": variant_id,
            "start_ratio": start_ratio,
            "end_ratio": end_ratio,
            "window_key": window_key,
            "video_path": self.video_path,
            "category": self.qa_category,
            "n_frames_requested": n_frames,
            "n_frames_returned": len(frames),
            "selected_indices": selected_indices,
            "selected_timestamps": [idx / self.meta["fps"] if self.meta["fps"] > 0 else 0.0 for idx in selected_indices],
            "window_frame_bounds": [window_start_idx, window_end_idx],
            "selected_within_window": selected_within_window,
            "variant_perturbation": {
                "phase": int(variant_id) % 3,
                "neighbor_gap": _variant_neighbor_gap(self.meta["total_frames"], n_frames, start_ratio, end_ratio),
                "canonical_semantic_hits": windowed_semantic_hits[:16],
                "variant_semantic_hits": semantic_candidate_hits[:16],
                "canonical_rerank_hits": rerank_hits[:16],
                "variant_rerank_hits": rerank_variant_hits[:16],
            },
            "frame_cache": self.frame_cache.stats(),
            "context_reuse": {
                "shared_plan": True,
                "shared_candidate_pool": True,
                "shared_semantic_retrieval": True,
                "decode_concurrency_limit": _VIDEO_DECODE_CONCURRENCY,
            },
            "plan": self.plan,
            "semantic": self.semantic_info,
            "rerank": rerank_info,
        }
        try:
            with open(os.path.join(item_cache, "retrieval_meta.json"), "w", encoding="utf-8") as f:
                json.dump(retrieval_meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return frames, retrieval_meta


# Agentic 抽帧对外入口。
# run.py 和 confidence.py 都通过这个函数拿到最终 base64 帧和可解释的 retrieval_meta。
def sample_frames_agentic_retrieval(
    video_path: str,
    question: str,
    qa_category: str,
    n_frames: int = 32,
    client=None,
    model: str | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    variant_id: int = 0,
    force_refresh: bool = False,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
    retrieval_context: AgenticRetrievalContext | None = None,
) -> tuple[list[str], dict[str, Any]]:
    if retrieval_context is not None:
        return retrieval_context.sample(
            n_frames=n_frames,
            variant_id=variant_id,
            start_ratio=start_ratio,
            end_ratio=end_ratio,
        )

    with AgenticRetrievalContext(
        video_path=video_path,
        question=question,
        qa_category=qa_category,
        client=client,
        model=model,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    ) as context:
        return context.sample(
            n_frames=n_frames,
            variant_id=variant_id,
            start_ratio=start_ratio,
            end_ratio=end_ratio,
        )
