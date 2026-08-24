import unittest

import numpy as np

from finetuning.grpo_muse import gate_group_scores
from mir import (
    FrozenComponentNormalizer,
    audit_pairs,
    combine_components,
    provisional_safe_radius,
    risk_coverage,
)


class RewardMathTest(unittest.TestCase):
    def test_frozen_scaling_uses_calibration_statistics(self):
        normalizer = FrozenComponentNormalizer.fit(
            {"beat": [0.2, 0.4, 0.6], "coverage": [0.8, 0.9, 1.0]}
        )
        result = combine_components(
            {"beat": [0.4, 0.6], "coverage": [0.95, 0.8]},
            {"beat": 0.5, "coverage": 0.5},
            normalizer=normalizer,
            mode="frozen_std",
        )
        self.assertEqual(result.shape, (2,))
        self.assertGreater(result[0], result[1])

    def test_pair_audit_keeps_ties_and_abstentions_visible(self):
        audit = audit_pairs(
            [0.5, -0.4, 0.0, 0.3],
            [1, 1, -1, -1],
            abstain=[False, False, False, True],
            high_margin_threshold=0.35,
        )
        self.assertEqual(audit.decided, 2)
        self.assertEqual(audit.ties, 1)
        self.assertEqual(audit.abstained, 1)
        self.assertAlmostEqual(audit.accuracy, 0.5)
        self.assertAlmostEqual(audit.high_margin_wrong_rate, 0.5)

    def test_risk_coverage_orders_by_confidence(self):
        coverage, risk = risk_coverage([0.2, 0.9, 0.6], [False, True, True])
        np.testing.assert_allclose(coverage, [1 / 3, 2 / 3, 1.0])
        np.testing.assert_allclose(risk, [0.0, 0.0, 1 / 3])

    def test_safe_radius_stops_before_repeated_failure(self):
        radius = provisional_safe_radius(
            [0.0, 0.02, 0.04, 0.06, 0.08],
            [False, False, True, True, False],
            consecutive=2,
        )
        self.assertAlmostEqual(radius, 0.02)

    def test_group_gate_neutralizes_invalid_candidate(self):
        result = gate_group_scores([0.2, 0.8, 1.0], [True, True, False])
        self.assertTrue(result.accepted)
        self.assertEqual(result.values, (0.2, 0.8, 0.5))
        self.assertEqual(result.valid, (True, True, False))


if __name__ == "__main__":
    unittest.main()
