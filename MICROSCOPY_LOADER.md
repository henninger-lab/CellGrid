# Microscopy Loader

`microscopy_loader.py` is the reusable raw-image reader in this repo. It is not tied to `CellGrid` internals.

Supported inputs:
- `.czi`
- `.ome.tif`
- `.ome.tiff`

Primary API:

```python
from microscopy_loader import MicroscopyImage, CziImage, find_supported_image_files

image = MicroscopyImage("/path/to/sample_global3d_aligned.ome.tiff")
channel_0_zyx = image.read_channel(0)
max_proj_cyx = image.zmax()
full_ctzyx = image.load_ctzyx()
pixel_size_x_um = image.scale["X"] * 1e6
channel_names = image.channel_names
```

Backwards compatibility:
- `automated_report.parse_czi.CziImage` now re-exports this shared loader.
- `CziImage` is kept as an alias of `MicroscopyImage` so older scripts can switch imports with minimal code changes.

Cross-project fit:
- `CellGrid` uses `read_channel`, `zmax`, `scale`, `channel_colors`, and `channel_wavelengths`.
- `ColocToolbox` utilities and radial distribution workflows use the same surface.
- `condensate_3D_segmentation` currently has a duplicate CZI-only reader and can be pointed at this module next.

Recommended migration in other repos:
- Replace `from parse_czi import CziImage` with `from microscopy_loader import CziImage`
- Keep downstream logic unchanged if it already uses `read_channel`, `get_channel_slice`, `image_data`, or `zmax`
