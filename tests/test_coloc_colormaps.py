import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location(
    'coloc_heatmap_test', Path(__file__).resolve().parents[1] / 'automated_report/coloc_heatmap.py')
coloc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coloc)


class ColocColormapTests(unittest.TestCase):
    def test_white_to_black_endpoints_clipping_and_midpoint(self):
        data = np.array([[-1., 0., 5., 10., 20.]])
        original = data.copy()
        rgb, scale = coloc.render_coloc_heatmap(data, vmax=10, colormap='white_to_black', return_scale=True)
        np.testing.assert_array_equal(rgb[0, :2], 255)
        np.testing.assert_array_equal(rgb[0, 3:], 0)
        self.assertTrue(np.all(np.abs(rgb[0, 2].astype(int) - 127) <= 1))
        self.assertEqual(scale['vmax'], 10)
        self.assertEqual(scale['colormap'], 'white_to_black')
        np.testing.assert_array_equal(data, original)

    def test_scale_bar_matches_heatmap(self):
        bar = coloc.render_coloc_scale_bar(height=3, width=2, colormap='white_to_black')
        heatmap = coloc.render_coloc_heatmap(np.array([[10., 5., 0.]]), vmax=10, colormap='white_to_black')
        np.testing.assert_array_equal(bar[:, 0], heatmap[0])

    def test_zero_image_is_white_and_oslo_default_stays_black(self):
        zeros = np.zeros((2, 3))
        np.testing.assert_array_equal(coloc.render_coloc_heatmap(zeros, colormap='white_to_black'), 255)
        np.testing.assert_array_equal(coloc.render_coloc_heatmap(zeros), 0)


if __name__ == '__main__':
    unittest.main()
