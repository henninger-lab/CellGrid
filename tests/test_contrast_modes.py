"""Exercise contrast mode selection through the report pipeline on synthetic fields."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from automated_report import report_generator as report


class ContrastModeTests(unittest.TestCase):
    def run_pipeline(self, mode=None):
        images = {
            'WT_1': np.array([[[1, 1], [10, 20]]], dtype=np.uint16),
            'WT_2': np.array([[[2, 2], [100, 200]]], dtype=np.uint16),
        }
        mask = np.array([[0, 0], [1, 1]])
        prefetcher = Mock()
        prefetcher.get.side_effect = lambda name: (
            images[name], SimpleNamespace(name=name, scale={'X': 1e-6, 'Y': 1e-6},
                                         channel_wavelengths=[], channel_colors=[]))
        def extract(**kwargs):
            return [SimpleNamespace(source_filename=kwargs['source_filename'],
                                    crop_meta=SimpleNamespace(channel_contrasts=[None]))]
        with TemporaryDirectory() as directory, ExitStack() as stack:
            config = report.ReportConfig(input_dir=directory, output_dir=directory, genotypes=['WT'])
            if mode is not None:
                config.contrast_mode = mode
            for name, replacement in {
                'find_supported_image_files': Mock(return_value=list(images)),
                'ImagePrefetcher': Mock(return_value=prefetcher),
                'segment_image': Mock(return_value=mask),
                'extract_nuclei_from_mask': extract,
                'generate_multipage_pdf': Mock(),
                'save_metadata_csv': Mock(),
            }.items():
                stack.enter_context(patch.object(report, name, replacement))
            with redirect_stdout(StringIO()):
                report.generate_qc_report_with_progress(config)
            pdf_crops = report.generate_multipage_pdf.call_args.args[0]
            csv_crops = report.save_metadata_csv.call_args.args[0]['WT']
            self.assertEqual(pdf_crops, csv_crops)
            return [(c.crop_meta.channel_contrasts[0].min_val,
                     c.crop_meta.channel_contrasts[0].max_val) for c in pdf_crops]

    def test_default_shares_actual_limits_between_fields(self):
        limits = self.run_pipeline()
        self.assertEqual(limits, [(1, 200), (1, 200)])

    def test_per_field_restores_independent_full_image_percentiles(self):
        limits = self.run_pipeline('per_field')
        np.testing.assert_allclose(limits, [(1, 19.85), (2, 198.5)])


if __name__ == '__main__':
    unittest.main()
