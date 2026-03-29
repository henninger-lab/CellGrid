from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
PARSE_CZI_PATH = REPO_ROOT / "automated_report" / "parse_czi.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import microscopy_loader as loader_module

parse_spec = importlib.util.spec_from_file_location("cellgrid_parse_czi", PARSE_CZI_PATH)
assert parse_spec is not None and parse_spec.loader is not None
parse_czi = importlib.util.module_from_spec(parse_spec)
parse_spec.loader.exec_module(parse_czi)

CziImage = loader_module.CziImage
MicroscopyImage = loader_module.MicroscopyImage
find_supported_image_files = loader_module.find_supported_image_files


def _write_ome_tiff(path: Path, arr: np.ndarray, metadata: dict[str, object]) -> None:
    tifffile.imwrite(path, arr, ome=True, metadata=metadata)


class OmeTiffCompatibilityTest(unittest.TestCase):
    def test_cziimage_reads_ome_tiff_projection_and_metadata(self) -> None:
        arr_czyx = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
        arr_to_write = np.transpose(arr_czyx, (1, 0, 2, 3))
        metadata = {
            "axes": "ZCYX",
            "PhysicalSizeX": 0.108,
            "PhysicalSizeY": 0.108,
            "PhysicalSizeZ": 0.3,
            "PhysicalSizeXUnit": "\u00b5m",
            "PhysicalSizeYUnit": "\u00b5m",
            "PhysicalSizeZUnit": "\u00b5m",
            "Channel": {
                "Name": ["fixed", "moving"],
                "Color": [0xFF00FF00, 0xFFFF00FF],
                "ExcitationWavelength": [488.0, 561.0],
                "EmissionWavelength": [520.0, 610.0],
            },
        }

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "aligned_sample_global3d_aligned.ome.tiff"
            _write_ome_tiff(path, arr_to_write, metadata)

            image = CziImage(path)

            self.assertEqual(image.name, "aligned_sample_global3d_aligned")
            self.assertEqual((image.C, image.Z, image.Y, image.X), (2, 3, 4, 5))
            self.assertEqual(image.T, 1)
            self.assertEqual(image.channel_names, ["fixed", "moving"])
            self.assertEqual(image.channel_colors, ["#00FF00", "#FF00FF"])
            self.assertEqual(image.channel_wavelengths[0]["excitation"], 488.0)
            self.assertEqual(image.channel_wavelengths[1]["emission"], 610.0)
            self.assertEqual(image.channel_fluorophores, ["mStayGold", "mScarlet"])
            self.assertAlmostEqual(image.scale["X"], 0.108e-6)
            self.assertAlmostEqual(image.scale["Z"], 0.3e-6)
            np.testing.assert_array_equal(image.read_channel(1), arr_czyx[1])
            np.testing.assert_array_equal(image.getChannelSlice(1), arr_czyx[1])
            np.testing.assert_array_equal(image.image_data, arr_czyx)
            np.testing.assert_array_equal(image.zmax(), arr_czyx.max(axis=1))

            shim_image = parse_czi.CziImage(path)
            self.assertEqual(shim_image.name, image.name)
            np.testing.assert_array_equal(shim_image.read_channel(1), arr_czyx[1])
            np.testing.assert_array_equal(shim_image.zmax(), arr_czyx.max(axis=1))

    def test_cziimage_reads_timelapse_ome_tiff_with_time_index(self) -> None:
        arr_ctzyx = np.arange(2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 2, 3, 4, 5)
        arr_to_write = np.transpose(arr_ctzyx, (1, 2, 0, 3, 4))
        metadata = {
            "axes": "TZCYX",
            "PhysicalSizeX": 0.2,
            "PhysicalSizeY": 0.2,
            "PhysicalSizeXUnit": "\u00b5m",
            "PhysicalSizeYUnit": "\u00b5m",
        }

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "timelapse_aligned.ome.tif"
            _write_ome_tiff(path, arr_to_write, metadata)

            image = CziImage(path)

            self.assertEqual((image.C, image.T, image.Z, image.Y, image.X), (2, 2, 3, 4, 5))
            np.testing.assert_array_equal(image.read_channel(0, time_index=1), arr_ctzyx[0, 1])
            np.testing.assert_array_equal(image.zmax(time_index=1), arr_ctzyx[:, 1].max(axis=1))

    def test_find_supported_image_files_includes_ome_tiffs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "sample_a.czi").touch()
            (root / "sample_b.ome.tiff").touch()
            (root / "sample_c.ome.tif").touch()
            (root / "sample_d.tif").touch()

            found = [Path(path).name for path in find_supported_image_files(str(root))]

            self.assertEqual(
                found,
                ["sample_a.czi", "sample_b.ome.tiff", "sample_c.ome.tif"],
            )


if __name__ == "__main__":
    unittest.main()
