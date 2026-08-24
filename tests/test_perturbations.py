import unittest

import numpy as np

from mir.perturbations import constant_shift, contiguous_gap, gain, local_shift


class PerturbationTest(unittest.TestCase):
    def test_constant_shift_never_wraps(self):
        values = np.arange(6, dtype=np.float32)
        np.testing.assert_array_equal(constant_shift(values, 2), [0, 0, 0, 1, 2, 3])
        np.testing.assert_array_equal(constant_shift(values, -2), [2, 3, 4, 5, 0, 0])

    def test_local_shift_preserves_outside_window(self):
        values = np.arange(10, dtype=np.float32)
        shifted = local_shift(values, start_sample=3, length_samples=4, shift_samples=1)
        np.testing.assert_array_equal(shifted[:3], values[:3])
        np.testing.assert_array_equal(shifted[7:], values[7:])
        np.testing.assert_array_equal(shifted[3:7], [0, 3, 4, 5])

    def test_gap_fill_modes_have_fixed_length(self):
        values = np.linspace(-1, 1, 100, dtype=np.float32)
        for fill in ("silence", "noise", "loop"):
            changed = contiguous_gap(
                values,
                start_sample=25,
                length_samples=20,
                fill=fill,
                seed=7,
            )
            self.assertEqual(changed.shape, values.shape)

    def test_gain_has_expected_ratio(self):
        values = np.asarray([0.25, -0.5], dtype=np.float32)
        np.testing.assert_allclose(gain(values, 6.0), values * 10 ** (6 / 20), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
