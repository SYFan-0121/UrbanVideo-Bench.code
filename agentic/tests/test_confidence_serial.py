"""Regression test for the GPU-safe, serial STRIVE vote loop."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import confidence


class SerialStriveTests(unittest.TestCase):
    def test_votes_are_sent_one_at_a_time(self):
        active = 0
        max_active = 0
        calls = []

        def sample_func(_video, _total_frames, start, end, n):
            return [f"frame-{start}-{end}-{i}" for i in range(n)]

        def fake_completion(_client, _model, messages, timeout):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            calls.append(messages[0]["content"][1]["image_url"]["url"])
            active -= 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Option: [A]\nConfidence: [5]"))]
            )

        with patch.object(confidence, "vlm_chat_completion", side_effect=fake_completion):
            answer, score, breakdown, retries, frames = confidence.confident_answer(
                client=object(),
                model="test-model",
                video=None,
                total_frames=100,
                qa_category="Goal Detection",
                question="Question: Where?\nChoices: A. Here\nB. There",
                start_ratio=0.0,
                end_ratio=1.0,
                num_options=2,
                K=3,
                n_frames=2,
                high_threshold=0.0,
                low_threshold=-1.0,
                max_retries=0,
                sample_func=sample_func,
            )

        self.assertEqual(answer, "A")
        self.assertEqual(retries, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(max_active, 1)
        self.assertEqual(len(frames), 2)
        self.assertEqual(breakdown["vote_dist"], {"A": 3})


if __name__ == "__main__":
    unittest.main()
