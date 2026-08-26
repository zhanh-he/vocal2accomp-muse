import math
import unittest

from mir.reward_attack import (
    best_of_k_kl,
    largest_connected_radius,
    select_best_of_k,
    top1_stability_under_noise,
    violates_noninferiority,
    within_prompt_scale,
)

import numpy as np


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

    def test_within_prompt_scale_removes_prompt_offsets(self):
        rows = [
            {"prompt_id": "a", "scores": {"r": 0.0}},
            {"prompt_id": "a", "scores": {"r": 2.0}},
            {"prompt_id": "b", "scores": {"r": 10.0}},
            {"prompt_id": "b", "scores": {"r": 12.0}},
        ]
        self.assertAlmostEqual(within_prompt_scale(rows, "r"), math.sqrt(4 / 3))

    def test_top1_stability_responds_to_margin(self):
        rng = np.random.default_rng(7)
        stable = top1_stability_under_noise(
            [0.0, 10.0], noise_std=0.1, draws=500, rng=rng
        )
        rng = np.random.default_rng(7)
        fragile = top1_stability_under_noise(
            [0.0, 0.01], noise_std=0.1, draws=500, rng=rng
        )
        self.assertEqual(stable, 1.0)
        self.assertLess(fragile, 0.65)


if __name__ == "__main__":
    unittest.main()
