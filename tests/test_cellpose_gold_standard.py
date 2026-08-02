import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "automated_report" / "cellpose_nuclear_seg.py"
SPEC = importlib.util.spec_from_file_location("cellgrid_cellpose_nuclear_seg", MODULE_PATH)
cellpose_nuclear_seg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cellpose_nuclear_seg)

DEFAULT_DIAMETER = cellpose_nuclear_seg.DEFAULT_DIAMETER
DEFAULT_MODEL_NAME = cellpose_nuclear_seg.DEFAULT_MODEL_NAME
DEFAULT_PAD_TO_SHAPE = cellpose_nuclear_seg.DEFAULT_PAD_TO_SHAPE
SEGMENTATION_PIPELINE_VERSION = cellpose_nuclear_seg.SEGMENTATION_PIPELINE_VERSION
segment_nuclei = cellpose_nuclear_seg.segment_nuclei
segment_nuclei_batch_2D = cellpose_nuclear_seg.segment_nuclei_batch_2D


class RecordingModel:
    def __init__(self):
        self.calls = []

    def eval(self, image, **kwargs):
        self.calls.append((image.copy(), kwargs.copy()))
        mask = np.zeros(image.shape, dtype=np.uint16)
        mask[2:-2, 2:-2] = 1
        return mask, None, None


def test_gold_standard_defaults_are_pinned():
    assert DEFAULT_MODEL_NAME == "cpsam_v2"
    assert DEFAULT_DIAMETER == 300
    assert DEFAULT_PAD_TO_SHAPE == (2576, 2576)
    assert SEGMENTATION_PIPELINE_VERSION == "radial_gold_v1"


def test_segment_nuclei_uses_standard_cellpose_eval_and_crops_padding(tmp_path):
    model = RecordingModel()
    image = np.ones((5, 6), dtype=np.uint16)

    mask = segment_nuclei(
        image,
        "sample",
        save_dir=tmp_path,
        model=model,
        min_area=0,
        max_area=100,
        save_plot=False,
        pad_to_shape=(9, 10),
    )

    run_image, params = model.calls[0]
    assert run_image.shape == (9, 10)
    assert mask.shape == image.shape
    assert params == {
        "diameter": 300,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "batch_size": 8,
        "tile_overlap": 0.1,
    }
    assert "niter" not in params
    assert "resample" not in params


def test_batch_entry_point_processes_images_independently(tmp_path):
    model = RecordingModel()
    items = [
        ("one", np.ones((5, 6), dtype=np.uint16)),
        ("two", np.ones((7, 8), dtype=np.uint16)),
    ]

    results = segment_nuclei_batch_2D(
        items,
        save_dir=tmp_path,
        model=model,
        min_area=0,
        max_area=100,
        save_mask_tiff=False,
        save_plot=False,
        pad_to_shape=(9, 10),
        batch_size="auto",
        return_names=True,
    )

    assert [name for name, _mask in results] == ["one", "two"]
    assert len(model.calls) == 2
    assert all(call_params["batch_size"] == 8 for _image, call_params in model.calls)
