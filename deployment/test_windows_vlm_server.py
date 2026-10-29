"""使用模拟生成器检查 VLM 服务接口、并发与请求适配。

Run from the repository root: python -m unittest deployment.test_windows_vlm_server -v
"""

import ast
import base64
import glob
import io
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

from deployment import windows_vlm_server as service
import vlm_api


class ServiceContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(service.app)
        self.model_patch = patch.multiple(
            service, _model=object(), _processor=object(),
            _model_name="test-model", API_KEY="", _request_slots=threading.BoundedSemaphore(1),
        )
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.payload = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}

    def test_completion_usage_and_sampling_parameters(self):
        payload = dict(self.payload, temperature=0.6, top_p=0.8, max_tokens=24)
        with patch.object(service, "_generate", return_value=("A", 12, 2, "stop")) as generate:
            response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["usage"]["total_tokens"], 14)
        self.assertEqual(generate.call_args.kwargs["temperature"], 0.6)
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 24)

    def test_rejects_incompatible_requests(self):
        for extra, code in [
            ({"model": "wrong"}, 404), ({"stream": True}, 400),
            ({"max_tokens": 0}, 400), ({"max_tokens": 1025}, 400),
            ({"temperature": -1}, 400), ({"top_p": 0.8}, 400),
            ({"seed": 42}, 422),
        ]:
            with self.subTest(extra=extra):
                response = self.client.post("/v1/chat/completions", json=dict(self.payload, **extra))
                self.assertEqual(response.status_code, code)

    def test_bearer_key(self):
        with patch.object(service, "API_KEY", "test-secret"), \
                patch.object(service, "_generate", return_value=("A", 12, 2, "stop")):
            self.assertEqual(self.client.post("/v1/chat/completions", json=self.payload).status_code, 401)
            response = self.client.post(
                "/v1/chat/completions", json=self.payload,
                headers={"Authorization": "Bearer test-secret"},
            )
            self.assertEqual(response.status_code, 200)

    def test_serial_generation(self):
        active = 0
        maximum = 0
        counter_lock = threading.Lock()

        def generate(*args, **kwargs):
            nonlocal active, maximum
            with counter_lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with counter_lock:
                active -= 1
            return "A", 12, 2, "stop"

        with patch.object(service, "_generate", side_effect=generate), ThreadPoolExecutor(3) as pool:
            futures = [pool.submit(self.client.post, "/v1/chat/completions", json=self.payload) for _ in range(3)]
            self.assertTrue(all(f.result().status_code == 200 for f in futures))
        self.assertEqual(maximum, 1)

    def test_oom_releases_slot(self):
        with patch.object(service, "_generate", side_effect=service.torch.cuda.OutOfMemoryError):
            response = self.client.post("/v1/chat/completions", json=self.payload)
            self.assertEqual(response.status_code, 507)
        with patch.object(service, "_generate", return_value=("A", 12, 2, "stop")):
            self.assertEqual(self.client.post("/v1/chat/completions", json=self.payload).status_code, 200)

    def test_image_conversion_limits(self):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), "red").save(buffer, format="JPEG")
        url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        image_item = {"type": "image_url", "image_url": {"url": url}}
        converted = service._to_qwen_messages([{"role": "user", "content": [image_item]}])
        self.assertEqual(converted[0]["content"][0]["max_pixels"], service.MAX_PIXELS)
        with patch.object(service, "MAX_IMAGES", 1), self.assertRaises(ValueError):
            service._to_qwen_messages([{"role": "user", "content": [image_item, image_item]}])
        for invalid in [None, "https://example.invalid/x.jpg", "data:image/jpeg;base64,???"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                service._decode_data_url(invalid)

    def test_generation_controls_and_context_limit(self):
        processor = Mock(return_value={"input_ids": SimpleNamespace(shape=(1, 4))})
        processor.apply_chat_template.return_value = "prompt"
        processor.batch_decode.return_value = ["A"]
        model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=100),
            generate=Mock(return_value=service.torch.tensor([[1, 2, 3, 4, 5, 6]])),
        )
        vision_utils = SimpleNamespace(process_vision_info=lambda messages: (None, None))
        with patch.multiple(service, _processor=processor, _model=model), \
                patch.dict(sys.modules, {"qwen_vl_utils": vision_utils}):
            result = service._generate(self.payload["messages"], 2, 0, None)
            self.assertEqual(result, ("A", 4, 2, "length"))
            self.assertFalse(model.generate.call_args.kwargs["do_sample"])
            service._generate(self.payload["messages"], 8, 0.5, 0.8)
            self.assertTrue(model.generate.call_args.kwargs["do_sample"])
            self.assertEqual(model.generate.call_args.kwargs["top_p"], 0.8)
            model.generate.reset_mock()
            with patch.object(service, "MAX_INPUT_TOKENS", 3), self.assertRaises(ValueError):
                service._generate(self.payload["messages"], 8, 0, None)
            model.generate.assert_not_called()


class ProjectIntegrationTests(unittest.TestCase):
    def test_timeout_floor_and_cloud_default(self):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return kwargs

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.dict(os.environ, {"VLM_PROVIDER": "qwen", "VLM_TIMEOUT_SECONDS": "900"}, clear=True):
            vlm_api.vlm_chat_completion(client, "test", [], timeout=60, max_tokens=10)
            self.assertEqual(captured["timeout"], 900)
        with patch.dict(os.environ, {"VLM_PROVIDER": "qwen"}, clear=True):
            vlm_api.vlm_chat_completion(client, "test", [], timeout=60)
            self.assertEqual(captured["timeout"], 60)

    def test_local_key_does_not_use_cloud_key(self):
        with patch.dict(os.environ, {
            "VLM_PROVIDER": "qwen", "QWEN_BASE_URL": "http://127.0.0.1:8000/v1",
            "DASHSCOPE_API_KEY": "unused-cloud-key", "VLM_API_KEY": "local-key",
        }, clear=True):
            client, _, _ = vlm_api.create_vlm_client()
            self.assertEqual(client.api_key, "local-key")
            client.close()

    def test_result_path_and_resume_discovery(self):
        # Execute only these pure helpers: importing run.py would pull in the
        # benchmark and optional video/retrieval dependencies unnecessarily.
        tree = ast.parse(Path("run.py").read_text(encoding="utf-8"))
        names = {"make_safe_run_id", "result_csv_path", "discover_run_ids"}
        helpers = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        namespace = {"os": os, "re": re, "glob": glob}
        exec(compile(helpers, "run.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = namespace["result_csv_path"](directory, "Qwen/Qwen2.5-VL-7B-Instruct", "test")
            self.assertEqual(os.path.dirname(path), directory)
            Path(path).touch()
            self.assertEqual(namespace["discover_run_ids"](directory, "Qwen/Qwen2.5-VL-7B-Instruct"), ["test"])


if __name__ == "__main__":
    unittest.main()
