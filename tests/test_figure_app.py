"""Check the actual Streamlit controls without loading microscopy data."""
from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest


class FigureAppTests(unittest.TestCase):
    def test_startup_and_contrast_mode_switching(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'figure_app.py'))
        app.run(timeout=60)
        self.assertFalse(app.exception, str(app.exception))
        mode = next(widget for widget in app.radio if widget.label == 'Contrast normalization')
        self.assertEqual(mode.value, 'shared')
        for selected in ('per_field', 'shared'):
            mode.set_value(selected).run(timeout=60)
            self.assertFalse(app.exception, str(app.exception))
            scale = next(widget for widget in app.number_input if widget.label == 'Scale bar length (µm)')
            self.assertEqual(scale.max, 50.0)
            self.assertEqual(scale.value, 5.0)
            low = next(widget for widget in app.number_input if widget.key == f'contrast_low_{selected}_0')
            self.assertEqual(low.value, 0.5 if selected == 'per_field' else 50.0)
            mode = next(widget for widget in app.radio if widget.label == 'Contrast normalization')


if __name__ == '__main__':
    unittest.main()
