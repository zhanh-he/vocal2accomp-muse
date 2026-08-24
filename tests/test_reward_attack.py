import math
import unittest

from mir.reward_attack import (
    best_of_k_kl,
    largest_connected_radius,
    select_best_of_k,
    violates_noninferiority,
)


class RewardAttackTest(unittest.TestCase):
    def test_best_of_k_kl(self):
        self.assertEqual(best_of_k_kl(1), 0.0)
        self.assertAlmostEqual(best_of_k_kl(4), math.log(4) - 0.75)

    def test_select_best_of_k_uses_fixed_prefix(self):
        rows = [
            {"prompt_id": "a", "candidate_index": 0, "scores": {"r": 0.2}},
            {"prompt_id": "a", "candidate_index": 1, "scores": {"r": 0.8}},
            {"prompt_id": "a", "candidate_index": 2, "scores": {"r": 1.0}},
            {"prompt_id": "b", "candidate_index": 0, "scores": {"r": 0.7}},
            {"prompt_id": "b", "candidate_index": 1, "scores": {"r": 0.4}},
            {"prompt_id": "b", "candidate_index": 2, "scores": {"r": 0.9}},
        ]
        selected = select_best_of_k(rows, "r", 2)
        self.assertEqual([row["candidate_index"] for row in selected], [1, 0])

    def test_radius_stops_before_second_consecutive_failure(self):
        radius = largest_connected_radius(
            [0.0, 0.2, 0.5, 0.8, 1.0],
            [False, True, False, True, True],
            consecutive=2,
        )
        self.assertEqual(radius, 0.5)

    def test_noninferiority_respects_metric_direction(self):
        self.assertTrue(violates_noninferiority(0.89, 0.92, direction="higher", delta=0.02))
        self.assertFalse(violates_noninferiority(0.91, 0.92, direction="higher", delta=0.02))
        self.assertTrue(violates_noninferiority(0.04, 0.01, direction="lower", delta=0.02))
        self.assertFalse(violates_noninferiority(0.02, 0.01, direction="lower", delta=0.02))


if __name__ == "__main__":
    unittest.main()
