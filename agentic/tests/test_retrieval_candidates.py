"""Checks that VLM rerank keeps its reduced candidate budget source-balanced."""

import unittest

from agentic_retrieval import RERANK_CANDIDATE_LIMIT, _balanced_unique_candidates


class BalancedRerankCandidateTests(unittest.TestCase):
    def test_round_robin_halves_candidate_budget_without_dropping_sources(self):
        sources = [list(range(start, start + 12)) for start in (0, 20, 40, 60, 80)]

        selected = _balanced_unique_candidates(sources, RERANK_CANDIDATE_LIMIT)

        self.assertEqual(len(selected), 21)
        counts = [sum(candidate in source for candidate in selected) for source in sources]
        self.assertEqual(counts, [5, 4, 4, 4, 4])

    def test_duplicate_frames_do_not_waste_the_candidate_budget(self):
        sources = [[1, 2, 3, 4], [1, 5, 6], [2, 7], [8], [9]]

        selected = _balanced_unique_candidates(sources, RERANK_CANDIDATE_LIMIT)

        self.assertEqual(selected, [1, 5, 2, 8, 9, 3, 6, 7, 4])


if __name__ == "__main__":
    unittest.main()
