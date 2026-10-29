# -*- coding: utf-8 -*-
"""提供实时进度接口、Socket.IO 推送、SQLite 历史查询与视频回看。

设计要点：
1. 已完成题目写入 SQLite，不再启动时把全部历史读入内存。
2. 运行中的题目只保存在 active_states 内存队列中；其中可包含当前帧，题目完成后不会落库。
3. 所有数据库写入统一加锁并使用 UPSERT，避免并发完成时重复写或写丢。
4. 前端通过 Socket.IO 接收三类事件：
   - state_update: 当前收到的单题状态。
   - active_queue_update: 正在运行的题目队列。
   - completed_update/history_item_added: 最新完成题目。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import base64
from typing import Any

from flask import Flask, Response, abort, jsonify, render_template, request, send_file
from flask_socketio import SocketIO, emit


app = Flask(__name__)
app.config["SECRET_KEY"] = "vqa-dashboard-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

VIDEO_DIR = os.path.abspath("dataset/videos")
RESULT_DIR = "result"
HISTORY_FILE = os.path.join(RESULT_DIR, "dashboard_history.json")
DEFAULT_RUN_ID = "default"

# latest_states 保存每个 RUN_ID 最近一次收到的状态。
# 前端刚打开页面时，会先用它恢复当前画面。
latest_states: dict[str, dict[str, Any]] = {}
# latest_completed_by_run 保存每个 RUN_ID 最近完成的一道题。
# 前端“最新完成”卡片会读取这里。
latest_completed_by_run: dict[str, dict[str, Any]] = {}
# active_states 只保存正在运行的题目。
# key 是 (run_id, qa_idx)，这样不同实验和题号不会互相覆盖。
active_states: dict[tuple[str, int], dict[str, Any]] = {}
_state_lock = threading.RLock()
_db_lock = threading.RLock()


def safe_run_id(value: str | None) -> str:
    # RUN_ID 会进入文件名和接口参数，所以只保留安全字符。
    value = str(value or DEFAULT_RUN_ID).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or DEFAULT_RUN_ID


def db_file_for(run_id: str | None = None) -> str:
    # 每个实验使用独立 SQLite 文件，避免新旧策略结果互相覆盖。
    return os.path.join(RESULT_DIR, f"dashboard_{safe_run_id(run_id)}.sqlite3")


def request_run_id() -> str:
    # 前端接口可以通过 ?run_id=xxx 指定要查看哪个实验。
    return safe_run_id(request.args.get("run_id") or request.args.get("run") or os.environ.get("RUN_ID"))


# 每个 RUN_ID 对应一个 SQLite 数据库文件。
# WAL 模式允许评测线程写入时，前端仍然能比较顺畅地读取历史。
def _connect(run_id: str | None = None) -> sqlite3.Connection:
    os.makedirs(RESULT_DIR, exist_ok=True)
    conn = sqlite3.connect(db_file_for(run_id), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


# 初始化 dashboard 数据库。
# qa_results 表保存历史题目的答案、正确性、STRIVE 摘要和 Agentic 摘要。
def init_db(run_id: str | None = None):
    with _db_lock, _connect(run_id) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_results (
                run_id TEXT,
                qa_idx INTEGER PRIMARY KEY,
                video_id TEXT,
                video_path TEXT,
                category TEXT,
                question TEXT,
                correct_answer TEXT,
                model_answer TEXT,
                is_correct INTEGER,
                total_processed INTEGER,
                correct_count INTEGER,
                accuracy REAL,
                retry_count INTEGER,
                retrieval_strategy TEXT,
                strive_score REAL,
                strive_json TEXT,
                strive_summary_json TEXT,
                retrieval_summary_json TEXT,
                raw_json TEXT,
                created_at REAL,
                updated_at REAL
            )
            """
        )
        try:
            conn.execute("ALTER TABLE qa_results ADD COLUMN run_id TEXT")
        except sqlite3.OperationalError:
            pass
        conn.execute("UPDATE qa_results SET run_id = COALESCE(run_id, ?)", (safe_run_id(run_id),))
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_category ON qa_results(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_correct ON qa_results(is_correct)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_video ON qa_results(video_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_updated ON qa_results(updated_at)")


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | None, default=None):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _video_path_from_id(video_id: str | None) -> str | None:
    # 历史记录只需要保存 video_id。
    # 回看时再从 dataset/videos 拼出真实视频路径。
    if not video_id:
        return None
    safe_name = os.path.basename(str(video_id).replace("\\", "/"))
    candidate = os.path.abspath(os.path.join(VIDEO_DIR, safe_name))
    return candidate if os.path.exists(candidate) else None


def _is_safe_video_path(path: str) -> bool:
    # 只允许访问 dataset/videos 目录内的文件。
    # 这是为了避免 /video 接口被用来读取任意本地文件。
    try:
        abs_path = os.path.abspath(path)
        return os.path.commonpath([VIDEO_DIR, abs_path]) == VIDEO_DIR and os.path.exists(abs_path)
    except Exception:
        return False


def _resolve_video_request_path() -> str | None:
    # 优先按 video_id 找视频。
    # path 参数只是兼容旧前端或旧数据库记录。
    video_id = request.args.get("id") or request.args.get("video_id")
    if video_id:
        resolved = _video_path_from_id(video_id)
        if resolved:
            return resolved

    path = request.args.get("path", "")
    if path and _is_safe_video_path(path):
        return os.path.abspath(path)
    return None


def _sample_preview_frames(path: str, limit: int = 12) -> list[str]:
    """Return small base64 JPEG preview frames for videos the browser cannot play."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            indices = list(range(max(1, limit)))
        elif limit <= 1:
            indices = [total // 2]
        else:
            indices = [round(i * (total - 1) / (limit - 1)) for i in range(limit)]

        frames: list[str] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            max_width = 720
            if width > max_width:
                scale = max_width / float(width)
                frame = cv2.resize(frame, (max_width, int(height * scale)))
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                frames.append(base64.b64encode(buf).decode("ascii"))
        return frames
    finally:
        cap.release()


# 浏览器播放 mp4 时常用 Range 请求只取一段文件。
# 这里显式支持 Range，解决历史回看视频一直加载的问题。
def _send_video_with_range(path: str):
    """Serve mp4 with explicit Range support so browser video metadata loads reliably."""
    file_size = os.path.getsize(path)
    range_header = request.headers.get("Range")
    if not range_header:
        response = send_file(path, mimetype="video/mp4", conditional=True)
        response.headers["Accept-Ranges"] = "bytes"
        return response

    match = re.match(r"bytes=(\d*)-(\d*)", range_header)
    if not match:
        abort(416)

    start_text, end_text = match.groups()
    if start_text == "" and end_text == "":
        abort(416)
    if start_text == "":
        suffix = int(end_text)
        start = max(0, file_size - suffix)
        end = file_size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    length = end - start + 1

    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(length)

    response = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
    response.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Content-Length"] = str(length)
    return response


# Agentic 原始 meta 很大，不适合完整展示。
# 这里压缩出前端真正需要看的字段，例如视觉目标、证据配比、语义召回和 rerank 状态。
def _compact_retrieval(retrieval: Any) -> dict[str, Any] | None:
    if not isinstance(retrieval, dict):
        return None

    variants = retrieval.get("variants")
    if isinstance(variants, list) and variants:
        first = next((v for v in variants if isinstance(v, dict)), {})
    else:
        first = retrieval
    if not isinstance(first, dict):
        first = {}

    semantic = first.get("semantic") if isinstance(first.get("semantic"), dict) else {}
    rerank = first.get("rerank") if isinstance(first.get("rerank"), dict) else {}
    plan = first.get("plan") if isinstance(first.get("plan"), dict) else {}
    frame_cache = first.get("frame_cache") if isinstance(first.get("frame_cache"), dict) else {}

    return {
        "strategy": retrieval.get("strategy") or first.get("strategy"),
        "variant_count": len(variants) if isinstance(variants, list) else 1,
        "selected_indices": first.get("selected_indices", []),
        "selected_timestamps": first.get("selected_timestamps", []),
        "start_ratio": first.get("start_ratio"),
        "end_ratio": first.get("end_ratio"),
        "selected_within_window": first.get("selected_within_window"),
        "plan": {
            "temporal_focus": plan.get("temporal_focus"),
            "visual_targets": plan.get("visual_targets", []),
            "query_terms": plan.get("query_terms", []),
            "evidence_mix": plan.get("evidence_mix", {}),
        },
        "semantic": {
            "enabled": semantic.get("enabled"),
            "backend": semantic.get("backend"),
            "error": semantic.get("error"),
            "cache_hit": semantic.get("cache_hit"),
        },
        "rerank": {
            "enabled": rerank.get("enabled"),
            "error": rerank.get("error"),
            "cache_hit": rerank.get("cache_hit"),
        },
        "frame_cache": frame_cache,
    }


def _get_any(d: dict[str, Any], *keys):
    for key in keys:
        if key in d and d.get(key) is not None:
            return d.get(key)
    return None


def _extract_strive_summary(strive: Any) -> dict[str, Any] | None:
    if not isinstance(strive, dict):
        return None
    return {
        "top_answer": strive.get("top_answer"),
        "vote_dist": strive.get("vote_dist", {}),
        "V": strive.get("V"),
        "M": strive.get("M"),
        "E": strive.get("E"),
        "H_norm": strive.get("H_norm"),
        "gamma_base": _get_any(strive, "γ_base", "gamma_base"),
        "gamma_eff": _get_any(strive, "γ_eff", "gamma_eff"),
        "delta_factor": _get_any(strive, "δ_factor", "delta_factor"),
        "STRIVE": strive.get("STRIVE"),
        "sampling_degenerate": strive.get("sampling_degenerate"),
        "sampling_windows": strive.get("sampling_windows", []),
        "retry_action": strive.get("retry_action"),
        "retrieval": _compact_retrieval(strive.get("retrieval")),
    }


# run.py 推来的 payload 字段可能不完整。
# 这里统一补齐 run_id、video_path、is_correct、strive_summary 和 retrieval_summary。
def normalize_payload(data: dict[str, Any] | None, include_frames: bool = True) -> dict[str, Any]:
    item = dict(data or {})
    item["run_id"] = safe_run_id(item.get("run_id") or os.environ.get("RUN_ID"))
    item["video_path"] = item.get("video_path") or _video_path_from_id(item.get("video_id"))

    correct = str(item.get("correct_answer") or "").strip().upper() or None
    model = str(item.get("model_answer") or "").strip().upper() or None
    item["is_correct"] = bool(correct and model and correct == model)

    strive = item.get("strive")
    item["strive_summary"] = item.get("strive_summary") or _extract_strive_summary(strive)
    if isinstance(strive, dict) and not item.get("retrieval_summary"):
        item["retrieval_summary"] = _compact_retrieval(strive.get("retrieval"))
    if not item.get("retrieval_summary"):
        item["retrieval_summary"] = _compact_retrieval(item.get("retrieval"))

    if not include_frames:
        item.pop("frames", None)
    return item


# 运行队列只展示摘要，不携带大量 base64 帧。
# 这样几千题并发展示时不会把浏览器内存撑爆。
def queue_item(data: dict[str, Any]) -> dict[str, Any]:
    """运行队列只发送摘要，避免队列列表重复携带大量 base64 帧。"""
    item = normalize_payload(data, include_frames=False)
    return {
        "run_id": item.get("run_id"),
        "qa_idx": item.get("qa_idx"),
        "video_id": item.get("video_id"),
        "category": item.get("category"),
        "question": item.get("question"),
        "retrieval_strategy": item.get("retrieval_strategy") or _strategy_from_item(item),
        "retry_count": item.get("retry_count"),
        "frame_count": len(data.get("frames") or []),
        "stage": infer_stage(item, len(data.get("frames") or [])),
        "updated_at": item.get("updated_at") or time.time(),
    }


def infer_stage(item: dict[str, Any], frame_count: int = 0) -> str:
    if item.get("stage"):
        return str(item.get("stage"))
    if item.get("model_answer"):
        return "已完成"
    strategy = item.get("retrieval_strategy") or _strategy_from_item(item)
    if str(strategy).startswith("agentic_retrieval"):
        if item.get("retrieval_summary"):
            return "Agentic 召回完成，等待模型"
        if frame_count:
            return "Agentic 已选帧，等待模型"
        return "Agentic 规划/候选池/语义召回"
    if item.get("strive") or item.get("strive_summary"):
        return "STRIVE 投票/置信度计算"
    if frame_count:
        return "抽帧完成，等待模型"
    return "抽帧或时域定位"


def _strategy_from_item(item: dict[str, Any]) -> str | None:
    retrieval = item.get("retrieval_summary")
    if isinstance(retrieval, dict):
        return retrieval.get("strategy")
    strive_summary = item.get("strive_summary")
    if isinstance(strive_summary, dict) and isinstance(strive_summary.get("retrieval"), dict):
        return strive_summary["retrieval"].get("strategy")
    return None


def row_to_item(row: sqlite3.Row) -> dict[str, Any]:
    item = {
        "run_id": row["run_id"] if "run_id" in row.keys() else DEFAULT_RUN_ID,
        "qa_idx": row["qa_idx"],
        "video_id": row["video_id"],
        "video_path": row["video_path"],
        "category": row["category"],
        "question": row["question"],
        "correct_answer": row["correct_answer"],
        "model_answer": row["model_answer"],
        "is_correct": bool(row["is_correct"]),
        "total_processed": row["total_processed"],
        "correct_count": row["correct_count"],
        "accuracy": row["accuracy"],
        "retry_count": row["retry_count"],
        "retrieval_strategy": row["retrieval_strategy"],
        "strive": _json_loads(row["strive_json"], None),
        "strive_summary": _json_loads(row["strive_summary_json"], None),
        "retrieval_summary": _json_loads(row["retrieval_summary_json"], None),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    return normalize_payload(item, include_frames=False)


# 完成题目写入 SQLite。
# UPSERT 表示同一 qa_idx 重复上报时更新旧记录，而不是插入重复行。
def upsert_result(item: dict[str, Any]) -> bool:
    """写入或更新完成结果。返回 True 表示这是第一次写入该 qa_idx。"""
    now = time.time()
    run_id = safe_run_id(item.get("run_id"))
    init_db(run_id)
    qa_idx = _safe_int(item.get("qa_idx"), -1)
    if qa_idx < 0:
        return False
    strive = item.get("strive") if isinstance(item.get("strive"), dict) else None
    strive_summary = item.get("strive_summary")
    retrieval_summary = item.get("retrieval_summary")
    with _db_lock, _connect(run_id) as conn:
        existed = conn.execute(
            "SELECT 1 FROM qa_results WHERE qa_idx = ?",
            (qa_idx,),
        ).fetchone() is not None
        conn.execute(
            """
            INSERT INTO qa_results (
                run_id, qa_idx, video_id, video_path, category, question, correct_answer,
                model_answer, is_correct, total_processed, correct_count, accuracy,
                retry_count, retrieval_strategy, strive_score, strive_json,
                strive_summary_json, retrieval_summary_json, raw_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(qa_idx) DO UPDATE SET
                run_id=excluded.run_id,
                video_id=excluded.video_id,
                video_path=excluded.video_path,
                category=excluded.category,
                question=excluded.question,
                correct_answer=excluded.correct_answer,
                model_answer=excluded.model_answer,
                is_correct=excluded.is_correct,
                total_processed=excluded.total_processed,
                correct_count=excluded.correct_count,
                accuracy=excluded.accuracy,
                retry_count=excluded.retry_count,
                retrieval_strategy=excluded.retrieval_strategy,
                strive_score=excluded.strive_score,
                strive_json=excluded.strive_json,
                strive_summary_json=excluded.strive_summary_json,
                retrieval_summary_json=excluded.retrieval_summary_json,
                raw_json=excluded.raw_json,
                updated_at=excluded.updated_at
            """,
            (
                run_id,
                qa_idx,
                item.get("video_id"),
                item.get("video_path"),
                item.get("category"),
                item.get("question"),
                item.get("correct_answer"),
                item.get("model_answer"),
                1 if item.get("is_correct") else 0,
                _safe_int(item.get("total_processed"), None),
                _safe_int(item.get("correct_count"), None),
                _safe_float(item.get("accuracy")),
                _safe_int(item.get("retry_count"), None),
                item.get("retrieval_strategy") or _strategy_from_item(item),
                _safe_float(strive.get("STRIVE") if strive else None),
                _json_dumps(strive) if strive else None,
                _json_dumps(strive_summary) if strive_summary else None,
                _json_dumps(retrieval_summary) if retrieval_summary else None,
                _json_dumps({k: v for k, v in item.items() if k != "frames"}),
                now,
                now,
            ),
        )
    return not existed


def db_count(run_id: str | None = None) -> int:
    init_db(run_id)
    with _db_lock, _connect(run_id) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM qa_results").fetchone()[0])


def db_recent(limit: int = 30, run_id: str | None = None) -> list[dict[str, Any]]:
    init_db(run_id)
    with _db_lock, _connect(run_id) as conn:
        rows = conn.execute(
            "SELECT * FROM qa_results ORDER BY qa_idx DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [row_to_item(row) for row in rows]


def db_latest_completed(run_id: str | None = None) -> dict[str, Any]:
    init_db(run_id)
    with _db_lock, _connect(run_id) as conn:
        row = conn.execute(
            "SELECT * FROM qa_results ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
    return row_to_item(row) if row else {}


def db_categories(run_id: str | None = None) -> list[str]:
    init_db(run_id)
    with _db_lock, _connect(run_id) as conn:
        rows = conn.execute(
            "SELECT DISTINCT category FROM qa_results WHERE category IS NOT NULL ORDER BY category"
        ).fetchall()
    return [row[0] for row in rows if row[0]]


def query_history(page: int, per_page: int, q: str, category: str, correct_filter: str, run_id: str | None = None):
    init_db(run_id)
    where = []
    params: list[Any] = []
    if q:
        where.append("(question LIKE ? OR video_id LIKE ? OR CAST(qa_idx AS TEXT) LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    if category:
        where.append("category = ?")
        params.append(category)
    if correct_filter in {"true", "false"}:
        where.append("is_correct = ?")
        params.append(1 if correct_filter == "true" else 0)
    where_sql = " WHERE " + " AND ".join(where) if where else ""
    offset = (page - 1) * per_page
    with _db_lock, _connect(run_id) as conn:
        total = int(conn.execute(f"SELECT COUNT(*) FROM qa_results{where_sql}", params).fetchone()[0])
        all_total = int(conn.execute("SELECT COUNT(*) FROM qa_results").fetchone()[0])
        rows = conn.execute(
            f"SELECT * FROM qa_results{where_sql} ORDER BY qa_idx DESC LIMIT ? OFFSET ?",
            params + [per_page, offset],
        ).fetchall()
    return [row_to_item(row) for row in rows], total, all_total


def migrate_jsonl_once():
    marker = os.path.join(RESULT_DIR, ".dashboard_jsonl_migrated")
    if os.path.exists(marker) or not os.path.exists(HISTORY_FILE):
        return
    migrated = 0
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = normalize_payload(json.loads(line), include_frames=False)
                    if item.get("model_answer") is not None:
                        upsert_result(item)
                        migrated += 1
                except Exception:
                    continue
        with open(marker, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
        print(f"已从 JSONL 迁移 {migrated} 条历史记录到 SQLite")
    except Exception as exc:
        print(f"迁移 JSONL 历史失败: {exc}")


def save_history_jsonl_compat(item: dict[str, Any]):
    """兼容保留 JSONL 追加，主存储仍以 SQLite 为准。"""
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(_json_dumps(item) + "\n")
    except Exception:
        pass


def active_queue(run_id: str | None = None) -> list[dict[str, Any]]:
    run_id = safe_run_id(run_id)
    with _state_lock:
        items = [queue_item(value) for (rid, _), value in active_states.items() if rid == run_id]
    return sorted(items, key=lambda x: _safe_int(x.get("qa_idx"), -1))


def list_runs() -> list[dict[str, Any]]:
    os.makedirs(RESULT_DIR, exist_ok=True)
    run_ids = set()
    for name in os.listdir(RESULT_DIR):
        match = re.match(r"dashboard_(.+)\.sqlite3$", name)
        if match:
            run_ids.add(match.group(1))
    with _state_lock:
        run_ids.update(rid for rid, _ in active_states.keys())
        run_ids.update(latest_states.keys())
        run_ids.update(latest_completed_by_run.keys())
    if not run_ids:
        run_ids.add(DEFAULT_RUN_ID)
    runs = []
    for rid in sorted(run_ids):
        runs.append({
            "run_id": rid,
            "db_path": db_file_for(rid),
            "history_total": db_count(rid),
            "active_count": len(active_queue(rid)),
        })
    return runs


init_db(DEFAULT_RUN_ID)
migrate_jsonl_once()
latest_completed_by_run[DEFAULT_RUN_ID] = db_latest_completed(DEFAULT_RUN_ID)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/state")
def get_state():
    run_id = request_run_id()
    with _state_lock:
        state = normalize_payload(latest_states.get(run_id), include_frames=True) if latest_states.get(run_id) else {}
    return jsonify({
        "run_id": run_id,
        "latest_state": state,
        "latest_completed": latest_completed_by_run.get(run_id) or db_latest_completed(run_id),
        "active_queue": active_queue(run_id),
        "history_total": db_count(run_id),
        "runs": list_runs(),
    })


@app.route("/runs")
def get_runs():
    return jsonify({"runs": list_runs(), "default_run_id": safe_run_id(os.environ.get("RUN_ID"))})


@app.route("/active")
def get_active():
    run_id = request_run_id()
    return jsonify({"run_id": run_id, "active": active_queue(run_id)})


@app.route("/active/<int:qa_idx>")
def get_active_item(qa_idx):
    run_id = request_run_id()
    with _state_lock:
        item = active_states.get((run_id, qa_idx))
    if not item:
        return jsonify({"error": "not found"}), 404
    return jsonify(item)


@app.route("/video")
def serve_video():
    path = _resolve_video_request_path()
    if not path:
        abort(404)
    return _send_video_with_range(path)


@app.route("/video_frames")
def serve_video_frames():
    path = _resolve_video_request_path()
    if not path:
        abort(404)
    limit = max(1, min(32, _safe_int(request.args.get("limit"), 12)))
    frames = _sample_preview_frames(path, limit=limit)
    return jsonify({
        "video_id": request.args.get("id") or request.args.get("video_id"),
        "frame_count": len(frames),
        "frames": frames,
    })


@app.route("/history")
def get_history():
    # 历史分页接口支持搜索、类别筛选和正确/错误筛选。
    # 返回的是轻量记录，不包含 base64 帧。
    run_id = request_run_id()
    page = max(1, _safe_int(request.args.get("page"), 1))
    per_page = max(1, min(200, _safe_int(request.args.get("per_page"), 50)))
    q = str(request.args.get("q", "")).strip()
    category = str(request.args.get("category", "")).strip()
    correct_filter = str(request.args.get("correct", "")).strip().lower()

    items, total, all_total = query_history(page, per_page, q, category, correct_filter, run_id)
    return jsonify({
        "run_id": run_id,
        "history": items,
        "total": total,
        "all_total": all_total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "categories": db_categories(run_id),
        "runs": list_runs(),
    })


@app.route("/history/<int:qa_idx>")
def get_history_item(qa_idx):
    # 点击历史题目时，前端通过这个接口加载单题详情。
    run_id = request_run_id()
    init_db(run_id)
    with _db_lock, _connect(run_id) as conn:
        row = conn.execute("SELECT * FROM qa_results WHERE qa_idx = ?", (qa_idx,)).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row_to_item(row))


@app.route("/update", methods=["POST"])
# run.py 每到一个关键阶段都会 POST 到 /update。
# 这个接口负责更新内存队列、写入完成结果，并通过 Socket.IO 推送给前端。
def update():
    raw = request.get_json(force=True)
    data = normalize_payload(raw, include_frames=True)
    run_id = safe_run_id(data.get("run_id"))
    data["updated_at"] = time.time()
    qa_idx = _safe_int(data.get("qa_idx"), -1)
    completed = data.get("model_answer") is not None

    with _state_lock:
        latest_states[run_id] = data
        if qa_idx >= 0 and not completed:
            active_states[(run_id, qa_idx)] = data
        elif qa_idx >= 0 and completed:
            active_states.pop((run_id, qa_idx), None)

    if completed:
        history_item = normalize_payload(data, include_frames=False)
        is_new_result = upsert_result(history_item)
        if is_new_result:
            save_history_jsonl_compat(history_item)
        latest_completed_by_run[run_id] = history_item
        socketio.emit("completed_update", history_item)
        if is_new_result:
            socketio.emit("history_item_added", history_item)

    socketio.emit("state_update", data)
    socketio.emit("active_queue_update", {"run_id": run_id, "active": active_queue(run_id)})
    socketio.emit("runs_update", {"runs": list_runs()})
    return jsonify({"status": "ok"})


@socketio.on("connect")
def on_connect():
    run_id = safe_run_id(os.environ.get("RUN_ID"))
    with _state_lock:
        state = latest_states.get(run_id)
    if state:
        emit("state_update", state)
    emit("active_queue_update", {"run_id": run_id, "active": active_queue(run_id)})
    latest_completed = latest_completed_by_run.get(run_id) or db_latest_completed(run_id)
    if latest_completed:
        emit("completed_update", latest_completed)
    emit("history_initial", {
        "run_id": run_id,
        "recent_items": db_recent(30, run_id),
        "total_count": db_count(run_id),
        "runs": list_runs(),
    })


if __name__ == "__main__":
    print("=" * 64)
    print("  UrbanVideo-Bench Dashboard -> http://localhost:5000")
    print(f"  Default RUN_ID: {DEFAULT_RUN_ID}")
    print(f"  Default SQLite: {db_file_for(DEFAULT_RUN_ID)}")
    print(f"  已完成历史: {db_count(DEFAULT_RUN_ID)} 条")
    print("=" * 64)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
