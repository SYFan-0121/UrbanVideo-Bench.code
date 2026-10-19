# -*- coding: utf-8 -*-
"""
VLM API provider adapter.

The benchmark sends locally sampled video frames as base64 JPEG image inputs.
This module keeps provider-specific endpoint and token-parameter differences in
one place, so run.py / confidence.py / agentic_retrieval.py do not need to know
whether the backend is MiMo or Qwen.
"""

import os
import math
from copy import deepcopy
from urllib.parse import urlparse
from openai import OpenAI


MIMO_SYSTEM_PROMPT = (
    "You are MiMo, an AI assistant developed by Xiaomi. "
    "Answer the video question based only on the provided visual inputs."
)


def _provider_name() -> str:
    return os.environ.get("VLM_PROVIDER", "mimo").strip().lower()


def create_vlm_client():
    """
    Create the OpenAI-compatible VLM client.

    Default provider is Xiaomi MiMo for the current experiment. Set
    VLM_PROVIDER=qwen to switch back to the original Dashscope/Qwen endpoint.
    """
    provider = _provider_name()

    if provider == "mimo":
        model = os.environ.get("MIMO_MODEL", "mimo-v2-omni").strip() or "mimo-v2-omni"
        api_key = os.environ.get("MIMO_API_KEY")
        if not api_key:
            raise RuntimeError("Set MIMO_API_KEY before running MiMo inference.")
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"),
        )
        return client, model, provider

    if provider == "qwen":
        model = os.environ.get("QWEN_MODEL", "qwen-vl-max-latest").strip() or "qwen-vl-max-latest"
        base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if urlparse(base_url).hostname in {"127.0.0.1", "localhost", "::1"}:
            # A local tunnel should not accidentally pick up a cloud Dashscope
            # key left in the shell environment.
            api_key = os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "local"
        else:
            api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set DASHSCOPE_API_KEY or OPENAI_API_KEY before running Qwen inference.")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        return client, model, provider

    raise ValueError("VLM_PROVIDER must be 'mimo' or 'qwen'.")


def _with_mimo_system_message(messages):
    """Add MiMo's system message unless the caller already supplied one."""
    if any(msg.get("role") == "system" for msg in messages):
        return messages
    return [{"role": "system", "content": MIMO_SYSTEM_PROMPT}] + messages


def vlm_chat_completion(client, model, messages, timeout=None, max_tokens=None, **kwargs):
    """
    Call chat.completions with provider-specific compatibility handling.

    Qwen/Dashscope accepts max_tokens. MiMo's example uses
    max_completion_tokens, so this adapter maps the common max_tokens argument
    when VLM_PROVIDER=mimo.
    """
    provider = _provider_name()
    payload = dict(kwargs)
    # A local VLM may queue behind another request and take several minutes.
    # VLM_TIMEOUT_SECONDS is an opt-in floor, so existing cloud defaults remain
    # unchanged unless the user explicitly sets it for a local run.
    timeout_floor = os.environ.get("VLM_TIMEOUT_SECONDS")
    if timeout_floor:
        try:
            floor = float(timeout_floor)
            if not math.isfinite(floor) or floor <= 0:
                raise ValueError("timeout must be positive and finite")
            timeout = floor if timeout is None else max(float(timeout), floor)
        except ValueError as exc:
            raise ValueError("VLM_TIMEOUT_SECONDS must be a positive finite number") from exc
    if timeout is not None:
        payload["timeout"] = timeout

    request_messages = deepcopy(messages)
    if provider == "mimo":
        request_messages = _with_mimo_system_message(request_messages)
        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens
    else:
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

    return client.chat.completions.create(
        model=model,
        messages=request_messages,
        **payload,
    )
