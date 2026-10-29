"""检查无水印均匀基准的长短视频采样、单次回答与失败题续跑。"""

from contextlib import redirect_stdout
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd

import run_uniform32 as baseline


class Uniform32Tests(unittest.TestCase):
    def test_sampling_preserves_raw_images_without_watermark(self):
        video = Mock()
        video.isOpened.return_value = True
        video.get.side_effect = lambda prop: 120 if prop == cv2.CAP_PROP_FRAME_COUNT else 24.0
        original = np.zeros((400, 600, 3), dtype=np.uint8)
        video.read.return_value = (True, original)
        with patch.object(baseline.cv2, "VideoCapture", return_value=video):
            frames, metadata = baseline.sample_uniform32("sample.mp4")
        expected_indices = [int(i * (119 / 31)) for i in range(32)]
        self.assertEqual(metadata["selected_indices"], expected_indices)
        self.assertEqual(len(frames), 32)
        self.assertEqual(video.read.call_count, 32)
        video.release.assert_called_once()
        _, jpeg = cv2.imencode(".jpg", original)
        self.assertEqual(frames, [base64.b64encode(jpeg).decode("utf-8")] * 32)
        self.assertFalse(metadata["watermark"])

    def test_short_videos_use_every_frame_including_single_frame(self):
        for total in (1, 17, 31, 32):
            with self.subTest(total=total):
                video = Mock()
                video.isOpened.return_value = True
                video.get.return_value = total
                original = np.zeros((10, 10, 3), dtype=np.uint8)
                video.read.return_value = (True, original)
                with patch.object(baseline.cv2, "VideoCapture", return_value=video):
                    frames, metadata = baseline.sample_uniform32("short.mp4")
                self.assertEqual(len(frames), total)
                self.assertEqual(metadata["selected_indices"], list(range(total)))
                self.assertEqual(video.read.call_count, total)
                video.release.assert_called_once()

    def test_empty_or_undecodable_video_reports_decode_failure(self):
        for total, success in [(0, True), (120, False)]:
            with self.subTest(total=total, success=success):
                video = Mock()
                video.isOpened.return_value = True
                video.get.side_effect = lambda prop: total if prop == cv2.CAP_PROP_FRAME_COUNT else 24.0
                video.read.return_value = (success, None)
                with patch.object(baseline.cv2, "VideoCapture", return_value=video):
                    with self.assertRaises(ValueError):
                        baseline.sample_uniform32("sample.mp4")
                video.release.assert_called_once()

    def test_resume_retries_gaps_and_keeps_existing_answers(self):
        self.assertEqual(baseline.completed_mask(pd.DataFrame({"Output": ["Option: [G]", float("nan"), "Option: [None]"]})).tolist(),
                         [True, False, False])
        questions = pd.DataFrame({
            "video_id": ["a.mp4", "b.mp4", "c.mp4"],
            "question_category": ["Object Recall"] * 3,
            "question": ["Where?\nChoices: A. Here\nB. There"] * 3,
            "answer": ["G", "A", "B"],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            results = baseline.load_results(questions, path, False, "test-run")
            results["Output"] = ["Option: [G]", "Option: [None]", "Option: [B]"]
            baseline.save_results(results, path)
            results = baseline.load_results(questions, path, True, "test-run")
            with (
                redirect_stdout(io.StringIO()),
                patch.object(baseline, "sample_uniform32", return_value=(["frame"] * 32, {"actual_frames": 32})),
                patch.object(baseline, "answer_once", return_value=("A", "Option: [A]")) as answer,
                patch.object(baseline, "emit_progress"),
            ):
                baseline.run_benchmark(results, path, Path(directory), Mock(), "model", "test-run")
            answer.assert_called_once()
            saved = pd.read_csv(path, index_col=0)
            self.assertEqual(saved["Output"].tolist(), ["Option: [G]", "Option: [A]", "Option: [B]"])
            self.assertEqual(saved.at[1, "RawOutput"], "Option: [A]")
            self.assertEqual(json.loads(saved.at[1, "RetrievalMeta"])["actual_frames"], 32)
            with self.assertRaises(ValueError):
                baseline.load_results(questions, path, True, "another-run")

    def test_answer_uploads_all_selected_long_and_short_video_frames_once(self):
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "Option: [G]\nConfidence: [5]"
        question = "Where?\nChoices: " + "\n".join(f"{letter}. Place" for letter in "ABCDEFG")
        for count in (1, 17, 32):
            with self.subTest(count=count):
                with patch.object(baseline, "vlm_chat_completion", return_value=response) as api:
                    option, raw = baseline.answer_once(Mock(), "model", "Object Recall", question, ["frame"] * count)
                self.assertEqual(option, "G")
                self.assertIn("[G]", raw)
                api.assert_called_once()
                content = api.call_args.kwargs["messages"][0]["content"]
                self.assertEqual(len(content), count + 1)
                self.assertIn("ONE letter from A to G", content[0]["text"])


if __name__ == "__main__":
    unittest.main()
