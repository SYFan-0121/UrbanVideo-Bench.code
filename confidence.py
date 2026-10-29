"""
confidence.py
STRIVE 置信度与自适应重试模块。

STRIVE = (V * M)^alpha * E^beta * gamma_eff * delta_factor

变量含义：
V: Vote consistency，K 次投票中最高票答案所占比例。
M: Margin，第一名和第二名票数差距，衡量答案胜出是否明显。
E: Entropy concentration，由投票分布信息熵转换而来，越集中越接近 1。
gamma_eff: 类别难度修正系数，问题越难基础值越低，但高一致性会提升它。
delta_factor: 模型自评置信度修正项，由模型输出的 Confidence: [1-5] 归一化得到。

confident_answer() 是主入口：它会进行 K=3 多窗口采样、多次 VLM 投票、
STRIVE 评分，并在低置信度时触发重采样或 Stage1 时域重定位。
"""

import math
import re
import time
import cv2
import base64
import hashlib
from collections import Counter
from openai import OpenAI
from prompts import build_prompt, detect_num_options
from vlm_api import vlm_chat_completion
from concurrent.futures import ThreadPoolExecutor, as_completed


def _console_log(qa_idx, message):
    """统一的 STRIVE 控制台日志；qa_idx 为空时可用于单元测试。"""
    prefix = f"[QA #{qa_idx}] " if qa_idx is not None else ""
    print(f"{prefix}{message}", flush=True)


def _is_agentic_strategy(strategy: str | None) -> bool:
    """判断当前策略是否是 Agentic 检索策略。"""
    return str(strategy or "").startswith("agentic_retrieval")


def _supports_stage1_relocation(strategy: str | None) -> bool:
    """判断低置信度时该策略是否允许通过 Stage1 重新定位时间窗口。"""
    return strategy in ("global", "tail") or _is_agentic_strategy(strategy)


def _frame_set_signature(frames: list[str]) -> str:
    """
    为一组 base64 帧生成稳定签名。

    用途：检测 STRIVE 的不同时间扰动窗口是否最终抽到了完全相同的帧。
    如果窗口不同但帧集合相同，置信度会被视为退化并被压低。
    """
    digest = hashlib.sha1()
    digest.update(str(len(frames)).encode("ascii"))
    for frame in frames:
        data = (frame or "").encode("utf-8", errors="ignore")
        digest.update(len(data).to_bytes(8, "little", signed=False))
        digest.update(data)
    return digest.hexdigest()


def _windows_are_distinct(windows: list[tuple[float, float]]) -> bool:
    """判断 K 个采样窗口是否真的不同，避免把同一窗口误判为退化采样。"""
    rounded = {(round(float(s), 4), round(float(e), 4)) for s, e in windows}
    return len(rounded) > 1

# ────────────────────────────────────────────────────────────
# 1. 类别难度校准系数 γ_cat
#    基于任务认知复杂度设定，取值范围 [0.65, 1.0]
#    越难的类别先验置信天花板越低，避免虚高置信
# ────────────────────────────────────────────────────────────
# 不同题型天然难度不同。
# 数值越低表示这个类别越难，STRIVE 会相应降低先验置信度。
CATEGORY_DIFFICULTY = {
    # 极难：需要抽象推理或反事实思维
    'Counterfactual':       0.65,
    'Causal':               0.68,
    'Cognitive Map':        0.70,
    'High-level Planning':  0.72,
    'Association Reasoning':0.74,

    # 中等：需要跨帧整合信息
    'Trajectory Captioning':0.78,
    'Sequence Recall':      0.78,
    'Duration':             0.80,
    'Scene Recall':         0.80,
    'Object Recall':        0.82,
    'Start/End Position':   0.82,

    # 较易：答案集中在特定帧段
    'Landmark Position':    0.85,
    'Progress Evaluation':  0.85,
    'Action Generation':    0.88,
    'Goal Detection':       0.90,
    'Proximity':            0.92,
}

# ────────────────────────────────────────────────────────────
# 2. 采样策略变体
#    K=3 次不同时间窗口的采样，用于投票
# ────────────────────────────────────────────────────────────
# STRIVE 的三票不是完全相同输入。
# 这个函数生成几个略有差异的时间窗口，用来测试答案是否稳定。
def get_sampling_variants(start_ratio: float, end_ratio: float, qa_category: str) -> list:
    """
    根据类别和已定位时间段，生成 K=3 个略有差异的采样变体。
    目的：通过时间扰动检验答案稳定性，而非纯随机。

    Returns:
        list of (start, end) tuples，每组对应一次采样窗口
    """
    span = end_ratio - start_ratio

    if span >= 0.8:
        # 全局或长窗口：保留一次全窗口，另外两次分别轻微裁掉边缘。
        trim = min(span * 0.05, 0.05)
        tail_trim = min(span * 0.10, 0.10)
        variants = [
            (start_ratio, end_ratio),
            (min(1.0, start_ratio + trim), max(start_ratio, end_ratio - trim)),
            (start_ratio, max(start_ratio, end_ratio - tail_trim)),
        ]
    elif qa_category in ("Causal", "Counterfactual"):
        # 因果/反事实问题经常依赖事件前后文，所以扰动时优先向前/向后扩展。
        variants = [
            (start_ratio, end_ratio),                          # 完整定位段
            (max(0.0, start_ratio - 0.1), end_ratio),          # 向前扩展
            (start_ratio, min(1.0, end_ratio + 0.1)),          # 向后扩展
        ]
    else:
        # 通用：在定位段内三等分，各取覆盖全段 + 略偏前 + 略偏后
        shift = span * 0.08
        variants = [
            (start_ratio, end_ratio),
            (max(0.0, start_ratio - shift), max(0.0, end_ratio - shift)),
            (min(1.0, start_ratio + shift), min(1.0, end_ratio + shift)),
        ]

    return variants


# ────────────────────────────────────────────────────────────
# 3. 从模型回答中提取选项字母
# ────────────────────────────────────────────────────────────
# 从模型回复中提取 A-G 选项。
# 模型输出格式不稳定，所以这里兼容多种写法。
def extract_option(text: str) -> str | None:
    """
    从模型输出中提取选项字母（A-G）。
    支持格式：'Option: [A]' / 'Option: A' / 单独字母行
    """
    # 优先匹配标准格式
    m = re.search(r'Option\s*:\s*\[?\s*([A-Ga-g])\s*\]?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 退化匹配：独立字母
    m = re.search(r'\b([A-Ga-g])\b', text)
    if m:
        return m.group(1).upper()
    return None


# ────────────────────────────────────────────────────────────
# 4. 从模型回答中提取自评分 δ_self
# ────────────────────────────────────────────────────────────
# 从模型回复中提取 Confidence: [1-5]。
# 这个自评分会成为 STRIVE 的 delta_factor。
def extract_self_score(text: str) -> float | None:
    """
    解析模型输出中的自评置信分（1-5），归一化为 [0, 1]。
    支持格式：'Confidence: [3]' / 'Confidence: 3'
    未找到时返回 None。
    """
    m = re.search(r'Confidence\s*:\s*\[?\s*([1-5])\s*\]?', text, re.IGNORECASE)
    if m:
        score = int(m.group(1))
        return (score - 1) / 4.0  # 1→0.0, 3→0.5, 5→1.0
    return None  # 未找到时返回 None


# ────────────────────────────────────────────────────────────
# 5. STRIVE 置信度公式核心计算
# ────────────────────────────────────────────────────────────
# STRIVE 五维置信度的公式实现。
# V 看投票一致性，M 看胜出边际，E 看分布集中度，gamma 看题型难度，delta 看模型自评。
def compute_strive(
    votes: list[str],
    self_scores: list[float],
    qa_category: str,
    num_options: int = 5,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[float, dict]:
    """
    计算 STRIVE 置信度分数，并返回可解释的 breakdown。

    Args:
        votes:        K 次采样各自提取出的答案字母列表，e.g. ['A', 'A', 'B']
        self_scores:  K 次采样各自的模型自评分（已归一化），e.g. [0.75, 1.0, 0.5]
        qa_category:  问题类别
        num_options:  题目实际选项总数（当前题库为 2–7）
        alpha:        V×M 项指数，控制一致性的惩罚强度
        beta:         熵项指数，控制分布集中度的惩罚强度

    Returns:
        (strive_score, breakdown_dict)
    """
    K = len(votes)  # 实际投票次数，通常为 3。
    valid_votes = [v for v in votes if v is not None]

    # ── 若全部无效 ──
    if not valid_votes:
        return 0.0, {
    "top_answer": None, "vote_dist": {},
    "V": 0, "M": 0, "E": 0, "H_norm": 0,
    "γ_base": CATEGORY_DIFFICULTY.get(qa_category, 0.8),
    "γ_eff":  CATEGORY_DIFFICULTY.get(qa_category, 0.8),
    "δ_factor": 1.0, "STRIVE": 0.0,
    }

    # V: 投票一致性。三票全一致时 V=1，两票一致时 V=2/3，三票全分裂时 V=1/3。
    count = Counter(valid_votes)
    top_answer, top_count = count.most_common(1)[0]
    V = top_count / K

    # M: 胜出边际。用于区分 2-1 小胜和 3-0 压倒性一致。
    if len(count) >= 2:
        second_count = count.most_common(2)[1][1]
    else:
        second_count = 0
    M = (top_count - second_count) / K

    # E: 投票分布集中度。先算熵 H，再归一化，最后用 1-H_norm 转成“越高越可信”。
    probs = [c / len(valid_votes) for c in count.values()]
    H = -sum(p * math.log(p + 1e-12) for p in probs)
    H_max = math.log(num_options)
    H_norm = H / H_max if H_max > 0 else 0.0
    E = 1.0 - H_norm  # 熵越低 → E 越高

    # gamma_eff: 类别先验 + 一致性提升。难题即使投票一致，也不会无限拔高置信度。
    gamma_base = CATEGORY_DIFFICULTY.get(qa_category, 0.80)
    gamma_eff = gamma_base + (1.0 - gamma_base) * (V ** 2)

    # delta_factor: 模型自评分修正。没有自评分时取 1.0，避免因解析失败额外惩罚。
    valid_self = [s for s in self_scores if s is not None]
    if not valid_self:
        delta_factor = 1.0
    else:
        raw_delta = sum(valid_self) / len(valid_self)
        delta_factor = (1.0 + raw_delta) / 2.0

    # 最终 STRIVE：一致性、胜出边际、分布集中度、类别难度、自评置信共同决定。
    strive = (V * M) ** alpha * E ** beta * gamma_eff * delta_factor

    breakdown = {
        "top_answer":    top_answer,
        "vote_dist":     dict(count),
        "V":             round(V, 3),
        "M":             round(M, 3),
        "E":             round(E, 3),
        "H_norm":        round(H_norm, 3),
        "γ_base":        round(gamma_base, 3),
        "γ_eff":         round(gamma_eff, 3),
        "δ_factor":      round(delta_factor, 3),
        "STRIVE":        round(strive, 4),
    }
    return strive, breakdown


# ────────────────────────────────────────────────────────────
# 6. 带置信度的主推理函数
# ────────────────────────────────────────────────────────────
# STRIVE 主流程。
# 它负责抽 K 组帧、并发调用模型投票、计算置信度，并在低分时决定是否重采样或重新定位。
def confident_answer(
    client: OpenAI,
    model: str,
    video,
    total_frames: int,
    qa_category: str,
    question: str,
    start_ratio: float,
    end_ratio: float,
    num_options: int = 5,
    K: int = 3,
    n_frames: int = 32,
    high_threshold: float = 0.70,
    low_threshold: float = 0.40,
    max_retries: int = 2,
    sample_func=None,
    initial_strategy: str = "global",
    stage1_locate_func=None,
    stage1_locate_corrective_func=None,
    overview_sample_func=None,
    emit_to_dashboard_func=None,
    dashboard_payload=None,
    qa_idx=None,
):
    """
    STRIVE 主推理函数：多窗口采样、多票 VLM 回答、置信度判断和自适应重试。

    参数说明：
    client/model: OpenAI 兼容接口客户端和模型名称。
    video/total_frames: 视频对象及总帧数；当前主流程多传 video=None，由 sample_func 自己读视频。
    qa_category/question: 问题类别和题干，用于 prompt 和类别难度系数。
    start_ratio/end_ratio: 初始时间窗口。
    K: 每轮投票次数，默认 3。
    n_frames: 每次投票抽取的基础帧数。
    high_threshold: 达到该分数直接采纳。
    low_threshold: 低于该分数触发诊断式重试。
    sample_func: 抽帧回调，必须支持 (video, total_frames, s, e, n)。
    initial_strategy: 外层策略名，例如 global/tail/dynamic/agentic_retrieval_strive。
    stage1_locate_func: 低置信度时可用的重新定位函数。
    overview_sample_func: Agentic 策略低置信度时用于抽 overview 帧的轻量函数。

    返回:
        (final_answer, strive_score, breakdown, retry_count, final_frames)
    """
    retry_count = 0             # 已执行的重试次数。
    best_answer = None          # 历史最高 STRIVE 对应的答案；达到最大重试次数时兜底采用。
    best_strive = -1.0          # 历史最高 STRIVE 分数。
    best_breakdown = {}         # 历史最佳轮次的完整可解释变量。
    best_frames = []            # 历史最佳轮次使用的帧。

    current_start, current_end = start_ratio, end_ratio  # 当前轮使用的时间窗口。
    current_n = n_frames                                  # 当前轮每票抽帧数量，可在重试时增加。
    final_frames = []                                     # 用于返回和 dashboard 展示的帧。

    while retry_count <= max_retries:
        votes = []        # K 次投票解析出的选项，如 ["A", "A", "B"]。
        self_scores = []  # K 次投票解析出的模型自评置信度，范围 [0, 1]。
        raw_outputs = []  # K 次投票的原始文本输出，便于调试。

        # 根据当前窗口生成 K 个轻微扰动的采样窗口。
        variants = get_sampling_variants(current_start, current_end, qa_category)

        sample_windows = variants[:K]
        _console_log(
            qa_idx,
            f"[STRIVE] Round {retry_count + 1}/{max_retries + 1}: "
            f"strategy={initial_strategy}, base_window=[{current_start:.2f}, {current_end:.2f}], "
            f"n_frames={current_n}, K={K}",
        )
        for vote_id, (s, e) in enumerate(sample_windows, start=1):
            _console_log(qa_idx, f"[STRIVE] Sampling vote {vote_id}: window=[{s:.2f}, {e:.2f}]")
        # 先串行完成抽帧，再并行调用 VLM；避免多个线程同时操作同一个视频对象。
        pre_sampled_frames = []
        for s, e in sample_windows:
            frames = sample_func(video, total_frames, s, e, n=current_n)
            pre_sampled_frames.append(frames)
            _console_log(qa_idx, f"[STRIVE] Sampled {len(frames)} frames for window=[{s:.2f}, {e:.2f}]")
        frame_signatures = [_frame_set_signature(frames) for frames in pre_sampled_frames]
        # 退化检测：如果请求窗口不同但得到的帧集合完全相同，说明鲁棒性投票已经失真。
        sampling_degenerate = (
            len(frame_signatures) > 1
            and len(set(frame_signatures)) == 1
            and _windows_are_distinct(sample_windows)
        )
        if sampling_degenerate:
            _console_log(
                qa_idx,
                "[STRIVE] Warning: distinct requested windows produced identical frame sets; "
                "confidence will be capped and diagnostics will continue.",
            )

        # dashboard 和最终结果默认展示第一个投票变体的帧。
        current_frames = pre_sampled_frames[0] if pre_sampled_frames else []
        final_frames = current_frames

        if emit_to_dashboard_func and dashboard_payload:
            update_payload = dashboard_payload.copy()
            update_payload['frames'] = current_frames
            update_payload['retry_count'] = retry_count
            emit_to_dashboard_func(update_payload)

        # ── 并行投票：K=3 次同时发起 API 请求 ──────────────────
        # 注意：此处只处理已采样的帧，不再访问视频对象，完全避免线程冲突
        def _single_vote(vote_id, frames):
            """单次 VLM 投票：构造 Prompt + 帧输入，返回选项、自评分和原始输出。"""
            if not frames:
                _console_log(qa_idx, f"[STRIVE] Vote {vote_id}: no frames; skip VLM call.")
                return None, None, ""

            prompt_with_conf = build_prompt(qa_category, question) + (
                "\n\nAdditionally, after your answer, rate your confidence:\n"
                "Confidence: [1-5]"
            )
            content = [{"type": "text", "text": prompt_with_conf}]
            for buf in frames:
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{buf}"}})
            try:
                _console_log(
                    qa_idx,
                    f"[STRIVE] Vote {vote_id}: sending {len(frames)} frames to VLM and waiting for response...",
                )
                result = vlm_chat_completion(
                    client,
                    model,
                    messages=[{"role": "user", "content": content}],
                    timeout=120,
                )
                raw = result.choices[0].message.content
                option = extract_option(raw)
                self_score = extract_self_score(raw)
                _console_log(
                    qa_idx,
                    f"[STRIVE] Vote {vote_id}: model answered option={option}, self_score={self_score}",
                )
                return option, self_score, raw
            except Exception as e:
                err_str = str(e)
                if 'Arrearage' in err_str or 'overdue' in err_str.lower():
                    raise RuntimeError(f"[FATAL] API 欠费: {err_str}")
                _console_log(qa_idx, f"[STRIVE] Vote {vote_id}: API error during voting: {e}")
                return None, None, ""

        # 并行发起 K 次投票（基于预采样的帧）
        _console_log(qa_idx, f"[STRIVE] Launching {K} parallel VLM votes.")
        with ThreadPoolExecutor(max_workers=K) as executor:
            futures = [
                executor.submit(_single_vote, vote_id, frames)
                for vote_id, frames in enumerate(pre_sampled_frames, start=1)
            ]
            votes, self_scores, raw_outputs = [], [], []
            for future in as_completed(futures):
                option, self_score, raw = future.result()  # RuntimeError 会在此处传播
                votes.append(option)
                self_scores.append(self_score)
                raw_outputs.append(raw)

        # ── 投票结束后，立即计算 STRIVE ──
        strive, breakdown = compute_strive(
            votes, self_scores, qa_category,
            num_options=num_options
        )
        # 将采样窗口和帧签名写入 breakdown，方便事后判断 STRIVE 是否真的使用了不同证据。
        breakdown["sampling_windows"] = [
            {
                "start_ratio": s,
                "end_ratio": e,
                "frame_count": len(frames),
                "frame_signature": sig,
            }
            for (s, e), frames, sig in zip(sample_windows, pre_sampled_frames, frame_signatures)
        ]
        breakdown["sampling_degenerate"] = sampling_degenerate
        if sampling_degenerate:
            # 即使三票答案一致，如果帧集合完全相同，也不能证明对时间扰动鲁棒。
            raw_strive = strive
            strive = min(strive, low_threshold - 1e-3)
            breakdown["raw_strive_before_sampling_cap"] = raw_strive
            breakdown["sampling_cap_reason"] = "distinct requested windows produced identical frame sets"
            breakdown["capped_strive"] = strive
        top_answer = breakdown.get("top_answer")
        _console_log(
            qa_idx,
            "[STRIVE] Scores: "
            f"votes={votes}, self_scores={self_scores}, top_answer={top_answer}, "
            f"V={breakdown.get('V')}, M={breakdown.get('M')}, E={breakdown.get('E')}, "
            f"gamma_base={breakdown.get('γ_base')}, gamma_eff={breakdown.get('γ_eff')}, "
            f"delta_factor={breakdown.get('δ_factor')}, STRIVE={strive:.4f}",
        )
                
        # 向前端发送置信度数据
        if emit_to_dashboard_func and dashboard_payload:
            update_payload = dashboard_payload.copy()
            update_payload['frames'] = current_frames
            update_payload['strive'] = breakdown
            update_payload['retry_count'] = retry_count
            emit_to_dashboard_func(update_payload)

        # 记录历史最佳
        if strive > best_strive:
            best_strive = strive
            best_answer = top_answer
            best_breakdown = breakdown
            best_frames = current_frames

        # ── 快速路径：三次投票全一致，直接返回 ──
        valid_vs = [v for v in votes if v is not None]
        if sampling_degenerate and len(valid_vs) == K and len(set(valid_vs)) == 1:
            _console_log(
                qa_idx,
                f"[STRIVE] Decision: unanimous answer={votes[0]} but sampling is degenerate; "
                "do not accept yet, continue diagnostics.",
            )
        elif len(valid_vs) == K and len(set(valid_vs)) == 1:
            _console_log(
                qa_idx,
                f"[STRIVE] Decision: accept. Reason=all {K} votes agree, answer={votes[0]}, STRIVE={strive:.4f}",
            )
            return top_answer, strive, breakdown, retry_count, current_frames

        # ── 阈值判断 ──
        if strive >= high_threshold:
            _console_log(
                qa_idx,
                f"[STRIVE] Decision: accept. Reason=STRIVE {strive:.4f} >= high_threshold {high_threshold:.2f}",
            )
            return top_answer, strive, breakdown, retry_count, current_frames

        elif strive < low_threshold and retry_count < max_retries:
            # ── 诊断驱动的策略切换 ──
            V_val = breakdown.get("V", 0)
            M_val = breakdown.get("M", 0)
            E_val = breakdown.get("E", 0)

            _console_log(
                qa_idx,
                f"[STRIVE] Decision: reject current answer for now. Reason=low confidence "
                f"{strive:.4f} < low_threshold {low_threshold:.2f}; "
                f"diagnostics V={V_val:.3f}, M={M_val:.3f}, E={E_val:.3f}",
            )

            if sampling_degenerate or V_val < 0.5:
                # 三票严重分裂或采样退化：优先怀疑“时间窗口/证据选择”有问题。
                overview_func = overview_sample_func or sample_func
                is_agentic_strategy = _is_agentic_strategy(initial_strategy)
                if is_agentic_strategy:
                    if retry_count > 0 and stage1_locate_corrective_func:
                        overview = overview_func(video, total_frames, 0.0, 1.0, n=8)
                        prev_s, prev_e = current_start, current_end
                        new_s, new_e = stage1_locate_corrective_func(
                            client, model, question, overview,
                            prev_start=current_start, prev_end=current_end
                        )
                        current_start, current_end = new_s, new_e
                        current_n = n_frames
                        breakdown["retry_action"] = "agentic_corrective_relocate"
                        _console_log(
                            qa_idx,
                            f"[STRIVE] Next action: agentic corrective Stage1 relocation. "
                            f"window {prev_s:.2f}~{prev_e:.2f} -> {new_s:.2f}~{new_e:.2f}",
                        )
                    elif stage1_locate_func:
                        overview = overview_func(video, total_frames, 0.0, 1.0, n=8)
                        new_s, new_e = stage1_locate_func(client, model, question, overview)
                        current_start, current_end = new_s, new_e
                        current_n = n_frames
                        breakdown["retry_action"] = "agentic_stage1_relocate"
                        _console_log(
                            qa_idx,
                            f"[STRIVE] Next action: agentic Stage1 relocation, new window={new_s:.2f}~{new_e:.2f}",
                        )
                    else:
                        current_n = n_frames
                        breakdown["retry_action"] = "agentic_resample_no_relocator"
                        _console_log(
                            qa_idx,
                            "[STRIVE] Next action: agentic resample with same strategy; no uniform fallback.",
                        )
                elif _supports_stage1_relocation(initial_strategy):
                    if is_agentic_strategy and retry_count > 0 and stage1_locate_corrective_func:
                        overview = overview_func(video, total_frames, 0.0, 1.0, n=8)
                        prev_s, prev_e = current_start, current_end
                        new_s, new_e = stage1_locate_corrective_func(
                            client, model, question, overview,
                            prev_start=current_start, prev_end=current_end
                        )
                        current_start, current_end = new_s, new_e
                        current_n = n_frames
                        _console_log(
                            qa_idx,
                            f"[STRIVE] Next action: corrective Stage1 relocation. "
                            f"window {prev_s:.2f}~{prev_e:.2f} -> {new_s:.2f}~{new_e:.2f}",
                        )
                    elif stage1_locate_func:
                        overview = overview_func(video, total_frames, 0.0, 1.0, n=8)
                        new_s, new_e = stage1_locate_func(client, model, question, overview)
                        current_start, current_end = new_s, new_e
                        current_n = n_frames
                        _console_log(
                            qa_idx,
                            f"[STRIVE] Next action: Stage1 relocation for {initial_strategy}, "
                            f"new window={new_s:.2f}~{new_e:.2f}",
                        )
                else:
                    if stage1_locate_corrective_func:
                        overview = overview_func(video, total_frames, 0.0, 1.0, n=8)
                        prev_s, prev_e = current_start, current_end
                        new_s, new_e = stage1_locate_corrective_func(
                            client, model, question, overview,
                            prev_start=current_start, prev_end=current_end
                        )
                        current_start, current_end = new_s, new_e
                        current_n = n_frames
                        _console_log(
                            qa_idx,
                            f"[STRIVE] Next action: dynamic corrective relocation, "
                            f"new window={new_s:.2f}~{new_e:.2f}",
                        )

            elif V_val >= 0.5 and M_val <= 0.34:
                # ── 2-1 分裂 → 同策略同窗口重采，不切换 ──
                # 投票分歧可能只是采样随机性，不是窗口选错了
                # current_start/end 和 initial_strategy 均保持不变
                current_n = n_frames   # 帧数也不变，纯粹换一组随机变体
                _console_log(
                    qa_idx,
                    f"[STRIVE] Next action: 2-1 split; resample with same strategy/window "
                    f"[{current_start:.2f}, {current_end:.2f}], n_frames={current_n}",
                )

            else:
                # ── V 较高但 STRIVE 仍低 → 信号弱，帧密度不够 ──
                # 窗口没有选错，只是每帧信息不足，加倍帧数
                current_n = n_frames * 2   # 32 → 64
                _console_log(
                    qa_idx,
                    f"[STRIVE] Next action: weak signal; increase frame density {n_frames}->{current_n}, "
                    "window unchanged.",
                )

            retry_count += 1
            time.sleep(2)

        elif retry_count < max_retries:
            _console_log(
                qa_idx,
                f"[STRIVE] Decision: not accepted yet. Reason=medium confidence STRIVE={strive:.4f}; "
                "next action=dense resample.",
            )
            # 密集重采：增加帧数，保持同一时间窗口
            current_n = int(n_frames * 1.5)   # 32 → 48
            retry_count += 1
            time.sleep(2)

        else:
            # 达到最大重试次数，采纳历史最佳
            _console_log(
                qa_idx,
                f"[STRIVE] Decision: max retries reached; accept best historical answer={best_answer}, "
                f"best_STRIVE={best_strive:.4f}",
            )
            break

    return best_answer, best_strive, best_breakdown, retry_count, best_frames


# ────────────────────────────────────────────────────────────
# 8. 单元测试（验证公式计算正确性）
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("STRIVE Score 公式验证测试")
    print("=" * 55)

    test_cases = [
        {
            "desc": "理想情况：3票全一致 + 自评高分",
            "votes": ["A", "A", "A"],
            "self_scores": [1.0, 0.75, 1.0],
            "category": "Proximity",
            "num_options": 3,
        },
        {
            "desc": "中等情况：2-1分裂 + 中等自评",
            "votes": ["B", "B", "C"],
            "self_scores": [0.5, 0.75, 0.25],
            "category": "Trajectory Captioning",
            "num_options": 5,
        },
        {
            "desc": "最差情况：三票全不同 + 低自评（困难类别）",
            "votes": ["A", "B", "C"],
            "self_scores": [0.25, 0.0, 0.25],
            "category": "Counterfactual",
            "num_options": 5,
        },
        {
            "desc": "Action Generation：2-1分裂 + 中高自评（6选项）",
            "votes": ["C", "C", "D"],
            "self_scores": [0.75, 1.0, 0.5],
            "category": "Action Generation",
            "num_options": 6,
        },
    ]

    for tc in test_cases:
        strive, breakdown = compute_strive(
            votes=tc["votes"],
            self_scores=tc["self_scores"],
            qa_category=tc["category"],
            num_options=tc["num_options"],
        )
        label = "✅ 高" if strive >= 0.70 else ("⚠️  中" if strive >= 0.40 else "❌ 低")
        print(f"\n{tc['desc']}")
        print(f"  Votes={tc['votes']}  Category={tc['category']}")
        for k, v in breakdown.items():
            if k not in ("vote_dist", "top_answer"):
                print(f"    {k:12s} = {v}")
        print(f"  → STRIVE = {strive:.4f}  [{label}置信]")
