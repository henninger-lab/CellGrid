"""Manual check of CellGrid's actual model and GPU; run in its Conda environment."""
import torch  # Import first, matching the Windows workstation launcher.
import sys
from importlib.metadata import version

from gpu_diagnostic import main as show_gpu_status


def main():
    show_gpu_status()
    print(f"Cellpose version: {version('cellpose')}")
    if sys.platform == 'win32' and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate cellgrid-win-gpu and check the NVIDIA driver before running CellGrid.")

    from automated_report.cellpose_nuclear_seg import get_model
    model = get_model(prefer_gpu=True)
    if sys.platform == 'win32' and model.device.type != 'cuda':
        raise RuntimeError(f"Expected the NVIDIA GPU, but Cellpose selected {model.device}.")
    print(f"Verified model: {model._henninger_model_name}")
    print(f"Weights: {model._henninger_loaded_model_path}")
    print(f"Device: {model._henninger_device_description}")


if __name__ == '__main__':
    main()
