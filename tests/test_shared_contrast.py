"""Contrast tests load the numeric module without optional microscopy GUI dependencies."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location(
    "shared_contrast", Path(__file__).resolve().parents[1] / "automated_report/shared_contrast.py"
)
contrast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contrast)


class SharedContrastTests(unittest.TestCase):
    def field(self, background, nuclear):
        pixels = np.asarray(background + nuclear, dtype=float)
        image = pixels.reshape(1, 1, -1)
        mask = np.asarray([0] * len(background) + [1] * len(nuclear)).reshape(1, -1)
        return contrast.field_histograms(image, mask)

    def test_sparse_signal_uses_every_nuclear_pixel(self):
        field = self.field([2] * 100000, [10] * 990 + [1000] * 10)
        self.assertEqual(contrast.compute_shared_contrast([field]), {0: (2, 1000)})

    def test_fields_have_equal_weight_despite_size(self):
        small = self.field([2], [100])
        large = self.field([8] * 10000, [200] * 10000)
        limits = contrast.compute_shared_contrast([small, large], [(25, 40)])
        self.assertEqual(limits, {0: (2, 100)})

    def test_background_is_outside_actual_mask(self):
        image = np.asarray([[[1, 10, 2], [20, 3, 30]]])
        mask = np.asarray([[0, 1, 0], [2, 0, 2]])
        field = contrast.field_histograms(image, mask)
        self.assertEqual(contrast.compute_shared_contrast([field]), {0: (2, 30)})
        self.assertEqual(sum(field[0][0][1]), 3)
        self.assertEqual(sum(field[0][1][1]), 3)

    def test_groups_share_thresholds_and_keep_unlisted_conditions_independent(self):
        groups = contrast.resolve_contrast_groups(['WT', 'KO', 'Drug', 'Other'], [['WT', 'KO']])
        fields = {name: [self.field([1], [value])] for name, value in
                  [('WT', 10), ('KO', 20), ('Drug', 100), ('Other', 200)]}
        limits = {}
        for group in groups:
            shared = contrast.compute_shared_contrast([f for name in group for f in fields[name]])
            limits.update({name: shared for name in group})
        self.assertEqual(limits['WT'], limits['KO'])
        self.assertEqual(limits['WT'][0], (1, 20))
        self.assertEqual(limits['Drug'][0], (1, 100))
        self.assertEqual(limits['Other'][0], (1, 200))

    def test_invalid_groups_rejected(self):
        for groups in [[['missing']], [['WT'], ['WT']], [[]], [['WT', 'WT']]]:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                contrast.resolve_contrast_groups(['WT', 'KO'], groups)

    def test_empty_populations_nonfinite_and_constant_data(self):
        field = self.field([float('nan'), 2], [float('inf'), 10])
        self.assertEqual(contrast.compute_shared_contrast([field]), {0: (2, 10)})
        self.assertEqual(contrast.compute_shared_contrast([self.field([], [10])]), {0: (0, 10)})
        self.assertEqual(contrast.compute_shared_contrast([self.field([4], [4])]), {0: (4, 5)})
        with self.assertRaises(ValueError):
            contrast.compute_shared_contrast([self.field([1], [])])
        self.assertEqual(contrast.compute_shared_contrast([
            self.field([2], []), self.field([2], [10])]), {0: (2, 10)})

    def test_channel_specific_percentiles_and_last_pair_reuse(self):
        one = self.field([1, 2, 3], [10, 20, 30])
        field = one * 3
        limits = contrast.compute_shared_contrast([field], [(0, 100), (100, 0)])
        self.assertEqual(limits, {0: (1, 30), 1: (3, 10), 2: (3, 10)})
        with self.assertRaises(ValueError):
            contrast.compute_shared_contrast([one, field])

    def test_per_field_settings_require_ordered_percentiles_and_no_groups(self):
        contrast.validate_contrast_settings("shared", [(90, 50)], [["WT", "KO"]])
        contrast.validate_contrast_settings("per_field")
        contrast.validate_contrast_settings("per_field", [(0.5, 99.5)])
        for mode, pairs, groups in [
            ("invalid", None, None),
            ("per_field", [(99, 50)], None),
            ("per_field", [(50, 50)], None),
            ("per_field", None, [["WT", "KO"]]),
        ]:
            with self.subTest(mode=mode, pairs=pairs, groups=groups), self.assertRaises(ValueError):
                contrast.validate_contrast_settings(mode, pairs, groups)

    def test_float_intensity_range_is_preserved(self):
        self.assertEqual(contrast.compute_shared_contrast([self.field([0.1], [0.2])]),
                         {0: (0.1, 0.2)})

    def test_validation_and_input_unchanged(self):
        for pairs in [[], [(float('nan'), 99)], [(50, 101)], [(-1, 99)]]:
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                contrast.validate_percentiles(pairs)
        image = np.array([[[2, 10]]], dtype=np.uint16)
        original = image.copy()
        contrast.compute_shared_contrast([contrast.field_histograms(image, np.array([[0, 1]]))])
        np.testing.assert_array_equal(image, original)


if __name__ == '__main__':
    unittest.main()
