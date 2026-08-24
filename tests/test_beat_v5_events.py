import unittest

import numpy as np

from mir.reward_function.beat_v5 import BeatV5Config, score_onset_grid_events


class BeatV5EventTest(unittest.TestCase):
    def test_aligned_grid_scores_above_shifted_grid(self):
        config = BeatV5Config(grid_subdivision=1, min_confidence=0.0)
        onsets = np.arange(0.5, 20.0, 0.5)
        aligned = score_onset_grid_events(onsets, onsets, 20.0, config=config)
        shifted = score_onset_grid_events(onsets, onsets + 0.18, 20.0, config=config)
        self.assertGreater(aligned.score, 0.99)
        self.assertGreater(aligned.score, shifted.score + 0.80)

    def test_sparse_evidence_abstains(self):
        result = score_onset_grid_events([1.0], [1.0], 30.0)
        self.assertTrue(result.abstain)
        self.assertIn("low_onset_or_beat_evidence", result.reasons)

    def test_local_tail_penalizes_one_bad_region(self):
        config = BeatV5Config(grid_subdivision=1, min_confidence=0.0)
        beats = np.arange(0.5, 40.0, 0.5)
        onsets = beats.copy()
        corrupted = onsets.copy()
        corrupted[(corrupted >= 16.0) & (corrupted < 24.0)] += 0.18
        clean_result = score_onset_grid_events(onsets, beats, 40.0, config=config)
        bad_result = score_onset_grid_events(corrupted, beats, 40.0, config=config)
        self.assertGreater(clean_result.components["local_soft_tail"], bad_result.components["local_soft_tail"])


if __name__ == "__main__":
    unittest.main()
