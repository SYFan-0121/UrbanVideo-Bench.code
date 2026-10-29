"""检查均匀 32 帧基准、全部 16 类 Agentic 模式与结果保存。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from eval import QUESTION_CATEGORIES
import run


class AgenticModeTests(unittest.TestCase):
    def check_category(self, category, mode, frame_sampling="agentic"):
        frames = [f"frame-{i}" for i in range(32)]
        results = pd.DataFrame([{
            "video_id": "sample.mp4", "question_category": category,
            "question": "Question: Where?\nChoices: A. Here\nB. There\nC. Elsewhere",
            "answer": "A", "Output": None,
        }])
        processed, correct = [0], [0]
        video = Mock()
        video.isOpened.return_value = True
        video.get.return_value = 120

        def vote_once(**kwargs):
            # Exercise the real callback, including its window and variant forwarding.
            for start, end in [(0.0, 1.0), (0.05, 0.95), (0.0, 0.9)]:
                selected = kwargs["sample_func"](None, 120, start, end, kwargs["n_frames"])
            return "A", 0.8, {"V": 1.0}, 0, selected

        with (
            tempfile.TemporaryDirectory() as directory,
            redirect_stdout(io.StringIO()),
            patch.object(run.cv2, "VideoCapture", return_value=video),
            patch.object(run, "emit_to_dashboard"),
            patch.object(run, "stage1_locate") as locate,
            patch.object(run, "AgenticRetrievalContext") as context,
            patch.object(run, "sample_frames_agentic_retrieval",
                         return_value=(frames, {"selected_indices": list(range(32))})) as retrieve,
            patch.object(run, "sample_frames_by_ratio_from_path", return_value=frames) as uniform,
            patch.object(run, "answer_once_with_frames", return_value=("A", frames, "Option: [A]")) as answer,
            patch.object(run, "confident_answer", side_effect=vote_once) as vote,
        ):
            run.process_one(
                0, results, str(Path(directory) / "results.csv"), directory,
                "test-model", Mock(), mode, "test-run", threading.Lock(),
                threading.Event(), processed, correct, frame_sampling=frame_sampling,
            )

            self.assertEqual(results.at[0, "Output"], "Option: [A]")
            strategy = "uniform_32_no_vote" if frame_sampling == "uniform" else f"agentic_retrieval_{mode}"
            self.assertEqual(results.at[0, "RetrievalStrategy"], strategy)
            self.assertEqual(processed, [1])
            self.assertEqual(correct, [1])
            locate.assert_not_called()

            if frame_sampling == "uniform":
                context.assert_not_called()
                retrieve.assert_not_called()
                vote.assert_not_called()
                uniform.assert_called_once_with(str(Path(directory) / "sample.mp4"), 120, 0.0, 1.0, n=32)
                answer.assert_called_once()
                self.assertEqual(answer.call_args.args[-1], frames)
                self.assertEqual(json.loads(results.at[0, "RetrievalMeta"])["actual_frames"], 32)
                self.assertTrue(pd.isna(results.at[0, "STRIVE"]))
            elif mode == "no_vote":
                context.assert_not_called()
                vote.assert_not_called()
                answer.assert_called_once()
                retrieve.assert_called_once()
                self.assertEqual(retrieve.call_args.kwargs["qa_category"], category)
                self.assertEqual(retrieve.call_args.kwargs["n_frames"], 32)
            else:
                answer.assert_not_called()
                context.assert_called_once()
                self.assertEqual(context.call_args.kwargs["qa_category"], category)
                vote.assert_called_once()
                self.assertEqual(vote.call_args.kwargs["n_frames"], 32)
                self.assertEqual(vote.call_args.kwargs["K"], 3)
                self.assertEqual([call.kwargs["variant_id"] for call in retrieve.call_args_list], [0, 1, 2])
                self.assertEqual(
                    [(call.kwargs["start_ratio"], call.kwargs["end_ratio"])
                     for call in retrieve.call_args_list],
                    [(0.0, 1.0), (0.05, 0.95), (0.0, 0.9)],
                )
                self.assertEqual(len(json.loads(results.at[0, "RetrievalMeta"])["variants"]), 3)

    def test_all_categories_use_agentic_no_vote(self):
        for category in QUESTION_CATEGORIES:
            with self.subTest(category=category):
                self.check_category(category, "no_vote")

    def test_all_categories_use_agentic_strive(self):
        for category in QUESTION_CATEGORIES:
            with self.subTest(category=category):
                self.check_category(category, "strive")

    def test_uniform_baseline_skips_agentic_and_strive_for_all_categories(self):
        # 即使会话残留 strive 设置，均匀基准也只抽帧并回答一次。
        for category in QUESTION_CATEGORIES:
            with self.subTest(category=category):
                self.check_category(category, "strive", frame_sampling="uniform")

    def test_uniform_baseline_rejects_incomplete_frame_budget(self):
        with (
            redirect_stdout(io.StringIO()),
            patch.object(run, "sample_frames_by_ratio_from_path", return_value=["frame"] * 31),
            patch.object(run, "answer_once_with_frames") as answer,
        ):
            with self.assertRaisesRegex(ValueError, "requires 32 frames; decoded 31"):
                run.answer_once_global_uniform(Mock(), "model", "video.mp4", 120, "Causal", "Question?")
            answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
