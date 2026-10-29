"""加载本地 Qwen2.5-VL 并提供 OpenAI 兼容多图推理接口。"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import logging
import os
import secrets
import threading
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict
from PIL import Image

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
# This is deliberately fixed at one: the server owns one shared GPU and this
# process must not create concurrent model generations accidentally.
MAX_CONCURRENCY = 1
REQUEST_TIMEOUT_SECONDS = max(1, int(os.environ.get("VLM_REQUEST_TIMEOUT_SECONDS", "900")))
MAX_NEW_TOKENS = max(1, min(1024, int(os.environ.get("VLM_MAX_NEW_TOKENS", "256"))))
MAX_IMAGES = max(1, int(os.environ.get("VLM_MAX_IMAGES", "64")))
MIN_PIXELS = max(28 * 28, int(os.environ.get("VLM_MIN_PIXELS", str(112 * 112))))
MAX_PIXELS = max(MIN_PIXELS, int(os.environ.get("VLM_MAX_PIXELS", str(336 * 336))))
MAX_INPUT_TOKENS = max(128, int(os.environ.get("VLM_MAX_INPUT_TOKENS", "12288")))
MAX_IMAGE_BYTES = max(1_000_000, int(os.environ.get("VLM_MAX_IMAGE_BYTES", str(8 * 1024 * 1024))))
API_KEY = os.environ.get("VLM_API_KEY", "").strip()

app = FastAPI(title="Local Video VLM", version="0.1.0")
_request_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)
_model = None
_processor = None
_model_name = None
_revision = None
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool | None = False


def _decode_data_url(value: str) -> Image.Image:
    if not isinstance(value, str) or not value.startswith("data:image/"):
        raise ValueError("Only base64 data:image URLs are supported")
    try:
        header, encoded = value.split(",", 1)
        if ";base64" not in header.lower():
            raise ValueError("Image URL must use base64 encoding")
        if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise ValueError("Encoded image is too large")
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError, binascii.Error) as exc:
        raise ValueError("Invalid base64 image data") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image payload exceeds {MAX_IMAGE_BYTES} bytes")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Decoded image exceeds 16 million pixels")
            image.load()
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError("The image payload could not be decoded") from exc


def _to_qwen_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        raise ValueError("messages must not be empty")
    converted = []
    image_count = 0
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Only system, user, and assistant roles are supported")
        content = message.get("content", "")
        if isinstance(content, str):
            converted.append({"role": message.get("role", "user"), "content": content})
            continue
        parts = []
        if not isinstance(content, list):
            raise ValueError("content must be text or an array of content items")
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Each content item must be an object")
            if item.get("type") == "text":
                parts.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                image_count += 1
                if image_count > MAX_IMAGES:
                    raise ValueError(f"Too many images; maximum is {MAX_IMAGES}")
                parts.append({
                    "type": "image",
                    "image": _decode_data_url(url),
                    # qwen-vl-utils reads these per-image limits before the
                    # processor, so the request cannot silently exceed the
                    # documented visual-token budget.
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                })
            else:
                raise ValueError(f"Unsupported content item type: {item.get('type')}")
        converted.append({"role": message.get("role", "user"), "content": parts})
    return converted


def _load_model(model_name: str, revision: str | None = None) -> None:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    global _model, _processor, _model_name, _revision
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the CUDA-enabled PyTorch installation")
    torch.set_num_threads(max(1, int(os.environ.get("VLM_CPU_THREADS", "4"))))
    budget_gib = float(os.environ.get("VLM_CUDA_MEMORY_GIB", "0"))
    if budget_gib > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = budget_gib * 1024**3 / total
        if not 0 < fraction <= 1:
            raise ValueError("VLM_CUDA_MEMORY_GIB must fit within GPU memory")
        # An allocator limit, not a GPU scheduler or reservation for others.
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "attn_implementation": os.environ.get("VLM_ATTN_IMPLEMENTATION", "sdpa"),
    }
    if revision:
        load_kwargs["revision"] = revision
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
    processor_kwargs = {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
    if revision:
        processor_kwargs["revision"] = revision
    _processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
    _model.eval()
    _model_name = model_name
    _revision = getattr(_model.config, "_commit_hash", None) or revision


def _generate(
    messages: list[dict[str, Any]],
    max_new_tokens: int,
    temperature: float | None,
    top_p: float | None,
) -> tuple[str, int, int, str]:
    from qwen_vl_utils import process_vision_info

    qwen_messages = _to_qwen_messages(messages)
    text = _processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(qwen_messages)
    inputs = _processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    input_length = inputs["input_ids"].shape[1]
    if input_length > MAX_INPUT_TOKENS:
        raise ValueError(f"Prompt has {input_length} tokens; maximum is {MAX_INPUT_TOKENS}")
    context_limit = getattr(_model.config, "max_position_embeddings", None)
    if context_limit is not None and input_length + max_new_tokens > context_limit:
        raise ValueError("Prompt plus requested generation exceeds model context length")
    inputs = {key: value.to("cuda:0") if hasattr(value, "to") else value for key, value in inputs.items()}
    do_sample = temperature is not None and temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p if top_p is not None else 1.0
    with torch.inference_mode():
        generated_ids = _model.generate(**inputs, **generation_kwargs)
    generated_ids_trimmed = [output_ids[input_length:] for output_ids in generated_ids]
    answer = _processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    completion_tokens = int(generated_ids_trimmed[0].shape[0])
    finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"
    return answer, int(input_length), completion_tokens, finish_reason


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _model is not None else "loading",
        "model": _model_name,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_concurrency": MAX_CONCURRENCY,
        "revision": _revision,
        "max_images": MAX_IMAGES,
        "max_pixels": MAX_PIXELS,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "generation_default": {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS},
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    model_id = _model_name or DEFAULT_MODEL
    return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]}


def _check_api_key(authorization: str | None) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if not secrets.compare_digest((authorization or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid local VLM API key")


@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_api_key(authorization)
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported")
    requested_model = request.model or _model_name
    if requested_model != _model_name:
        raise HTTPException(status_code=404, detail=f"Only model {_model_name} is loaded")
    if request.max_tokens is not None:
        max_new_tokens = request.max_tokens
    elif request.max_completion_tokens is not None:
        max_new_tokens = request.max_completion_tokens
    else:
        max_new_tokens = MAX_NEW_TOKENS
    if max_new_tokens < 1 or max_new_tokens > 1024:
        raise HTTPException(status_code=400, detail="max_tokens must be between 1 and 1024")
    if request.temperature is not None and not 0 <= request.temperature <= 2:
        raise HTTPException(status_code=400, detail="temperature must be between 0 and 2")
    if request.top_p is not None and not 0 < request.top_p <= 1:
        raise HTTPException(status_code=400, detail="top_p must be in (0, 1]")
    if request.top_p is not None and not request.temperature:
        raise HTTPException(status_code=400, detail="top_p requires temperature > 0")
    if request.max_tokens is not None and request.max_completion_tokens is not None:
        raise HTTPException(status_code=400, detail="Supply only one token limit parameter")
    if not _request_slots.acquire(timeout=REQUEST_TIMEOUT_SECONDS):
        raise HTTPException(status_code=429, detail="Server is busy; retry later")
    started = time.perf_counter()
    try:
        answer, prompt_tokens, completion_tokens, finish_reason = _generate(
            request.messages,
            max_new_tokens=max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=507,
            detail="GPU out of memory; reduce frame count, image pixels, or max_tokens",
        ) from exc
    except Exception as exc:
        logger.exception("Local VLM generation failed")
        raise HTTPException(status_code=500, detail="Inference failed; inspect server log") from exc
    finally:
        _request_slots.release()
    return {
        "id": f"chatcmpl-local-{uuid.uuid4().hex}",
        "object": "chat.completion", "created": int(time.time()),
        "model": _model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "x_inference_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--host", default=os.environ.get("VLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLM_PORT", "8000")))
    parser.add_argument("--revision", default=os.environ.get("VLM_REVISION"))
    args = parser.parse_args()
    _load_model(args.model, revision=args.revision)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
