"""Send 1-64 synthetic JPEG frames to the local Windows VLM service."""

from __future__ import annotations

import base64
import argparse
import io
import os

import httpx
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, choices=range(1, 65), default=1)
    args = parser.parse_args()
    content = [{"type": "text", "text": "Describe these images in one short sentence."}]
    for index in range(args.frames):
        image = Image.new("RGB", (336, 336), (80, 120, 100 + index))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    headers = {}
    if os.environ.get("VLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['VLM_API_KEY']}"
    payload = {
        "model": os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
        "messages": [{
            "role": "user",
            "content": content,
        }],
        "max_tokens": 32,
        "temperature": 0,
    }
    base_url = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    with httpx.Client(timeout=900) as client:
        health = client.get(base_url.removesuffix("/v1") + "/health")
        health.raise_for_status()
        print("health:", health.json())
        response = client.post(
            base_url + "/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        print(response.json())


if __name__ == "__main__":
    main()
