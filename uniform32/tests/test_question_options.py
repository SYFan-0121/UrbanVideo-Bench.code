"""检查 2–7 项题目的提示范围、答案提取与 G 选项评估。"""

import unittest

from answers import extract_option
from eval import extract_option_letter
from prompts import build_prompt, detect_num_options


class QuestionOptionTests(unittest.TestCase):
    def test_actual_option_count_drives_prompt_for_any_category(self):
        for count in range(2, 8):
            for category in ("Action Generation", "Proximity", "Counterfactual"):
                with self.subTest(count=count, category=category):
                    last = chr(ord('A') + count - 1)
                    options = [f"{chr(ord('A') + i)}. Place {i}" for i in range(count)]
                    question = "Question: Where?\nChoices: " + "\n".join(options)
                    self.assertEqual(detect_num_options(question), count)
                    self.assertIn(f"ONE letter from A to {last} inside", build_prompt(category, question))

    def test_seventh_option_survives_answer_extraction_and_evaluation(self):
        for letter in "ABCDEFG":
            for response in (letter, f"Option: [{letter}]\nConfidence: [5]", f"Option: {letter.lower()}"):
                with self.subTest(response=response):
                    self.assertEqual(extract_option(response), letter)
                    self.assertEqual(extract_option_letter(response), letter)

    def test_answer_prefix_and_indented_options(self):
        question = "**Question:** Where?\nchoose:\nAnswer: A. Here\n    B. There"
        self.assertEqual(detect_num_options(question), 2)

    def test_missing_answer_is_not_an_option(self):
        self.assertIsNone(extract_option("Option: [None]"))
        self.assertIsNone(extract_option_letter("Option: [None]"))


if __name__ == "__main__":
    unittest.main()
