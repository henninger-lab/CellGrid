"""
Automated QC Report Generator.

Orchestrates the full pipeline:
1. Load/cache max projections
2. Segment nuclei with Cellpose cpsam_v2
3. Extract nucleus crops
4. Group by genotype
5. Compute global contrast
6. Generate multi-page PDF reports
"""

import os
import json
import uuid
import re
import numpy as np
import tifffile
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev

# sys.path.insert(0, str(Path(__file__).parent.parent))
from .parse_czi import CziImage, find_supported_image_files, microscopy_image_stem
from .prefetch import ImagePrefetcher
from .cellpose_nuclear_seg import DEFAULT_MODEL_NAME, segment_nuclei, save_qc_contact_sheet
from .figure_generator import ( 
    ColorScheme, 
    get_scheme,
    FigurePanel, 
    PanelConfig, 
    ScaleBarConfig,
    ChannelContrast,
)

from .nucleus_extractor import (
    NucleusCrop,
    extract_nuclei_from_mask,
    extract_crop_data,
    get_crop_bounds,
)
from .contrast_calculator import (
    compute_global_contrast,
    compute_per_file_contrast,
    apply_file_contrast_to_nuclei,
    apply_global_contrast,
    format_contrast_summary,
)

from .coloc_heatmap import (
    compute_coloc_max_projection,
    render_coloc_heatmap,
    render_coloc_scale_bar,
    VALID_COLOC_COLORMAPS,
)

from .color_schemes import scheme_custom, ColorScheme


import time
class PipelineTimer:
    """Simple timer for profiling pipeline stages."""
    
    def __init__(self):
        self.times = {}  # stage -> list of durations
        self._start = None
        self._stage = None
    
    def start(self, stage: str):
        """Start timing a stage."""
        self._stage = stage
        self._start = time.perf_counter()
    
    def stop(self):
        """Stop timing and record duration."""
        if self._start is None:
            return
        duration = time.perf_counter() - self._start
        if self._stage not in self.times:
            self.times[self._stage] = []
        self.times[self._stage].append(duration)
        self._start = None
        return duration
    
    def print_summary(self):
        """Print timing summary."""
        print("\n" + "=" * 60)
        print("PIPELINE TIMING SUMMARY")
        print("=" * 60)
        
        total_per_image = 0
        for stage, durations in self.times.items():
            mean = sum(durations) / len(durations)
            total_per_image += mean
            if len(durations) > 1:
                std = (sum((d - mean) ** 2 for d in durations) / len(durations)) ** 0.5
                print(f"  {stage:25s}: {mean:6.2f}s ± {std:.2f}s  (n={len(durations)})")
            else:
                print(f"  {stage:25s}: {mean:6.2f}s")
        
        print("-" * 60)
        print(f"  {'TOTAL PER IMAGE':25s}: {total_per_image:6.2f}s")
        print("=" * 60)
    

@dataclass
class ReportConfig:
    """Configuration for automated report generation."""
    # Input/Output
    input_dir: str
    output_dir: str
    genotypes: list[str]
    
    # Segmentation
    nuclear_channel: int = 0
    cellpose_diameter: int = 300
    cellpose_flow_threshold: float = 0.2
    cellpose_cellprob_threshold: float = 0.0
    min_area_pixels: int = 5000
    max_area_pixels: int = 500000
    nuclear_segmentation_blur_sigma: float = 0.0
    exclude_border: bool = True
    
    # Visualization
    box_size_um: float = 20.0
    scale_bar_um: float = 5.0
    scale_bar_fontsize: int = 8
    gap_pixels: int = 10
    border_pixels: int = 10
    rows_per_page: int = 10  # Rows per column
    nucleus_cols: int = 3    # Nucleus columns per page
    dpi: int = 300
    show_outline: bool = True  # Draw nucleus outlines
    show_channel_header_labels: bool = True  # Legacy field; page-1 channel labels are always shown
    
    # Contrast - list of (low, high) percentile tuples per channel
    # If None, uses (0.5, 99.5) for all channels
    # If fewer tuples than channels, last tuple is reused
    fish_contrast_setting: bool = False
    contrast_percentiles: Optional[list[tuple[float, float]]] = None
    
    # Channels
    visible_channels: Optional[list[int]] = None  # None = all
    channel_colors: Optional[list[str]] = None  # None = auto from wavelengths
    channel_labels: Optional[list[str]] = None  # Optional manual labels by channel index
    merged_label: str = "Merged"
    
    # Colocalization settings
    enable_coloc: bool = False  # Enable colocalization heatmap
    coloc_channels: Optional[tuple[int, int]] = None  # Channel pair for colocalization
    coloc_vmax: Optional[float] = None  # Max value for colormap scaling (None = auto)
    coloc_colormap: str = 'oslo'
    
    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ReportConfig':
        """Create from dictionary."""
        return cls(**d)


def get_cache_paths(config: ReportConfig) -> tuple[str, str]:
    """Get paths for cache and segmentation directories."""
    cache_dir = config.cache_dir or os.path.join(config.output_dir, "cache")
    seg_dir = os.path.join(config.output_dir, "segmentation")
    return cache_dir, seg_dir


def _is_valid_max_projection(arr: np.ndarray) -> bool:
    """Validate cached max projection shape before using it."""
    return arr.ndim == 3 and arr.shape[0] > 0 and arr.shape[1] > 0 and arr.shape[2] > 0


def _write_tiff_atomic(path: str, data: np.ndarray, zstd_level: int = 6):
    """Write TIFF atomically to avoid partial cache files."""
    tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"
    try:
        tifffile.imwrite(
            tmp_path,
            data,
            compression='zstd',
            compressionargs={'level': zstd_level},
        )
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _wrap_filename_header(
    filename: str,
    available_width_in: float,
    fontsize: float,
    max_lines: int = 3,
) -> str:
    """Wrap long filename headers so they stay inside a figure column."""
    avg_char_width_in = max((fontsize * 0.6) / 72.0, 0.01)
    max_chars = max(10, int(max(available_width_in, 0.25) / avg_char_width_in))
    tokens = re.findall(r'[^_\-\s]+(?:[_\-\s]+|$)', filename) or [filename]

    lines: list[str] = []
    current = ""
    token_idx = 0
    overflowed = False

    while token_idx < len(tokens):
        token = tokens[token_idx]

        if current and len(current) + len(token) <= max_chars:
            current += token
            token_idx += 1
            continue

        if not current and len(token) <= max_chars:
            current = token
            token_idx += 1
            continue

        if current:
            lines.append(current.rstrip())
            current = ""
            if len(lines) >= max_lines:
                overflowed = True
                break
            continue

        # No natural break point available, so hard-wrap the token.
        lines.append(token[:max_chars].rstrip())
        tokens[token_idx] = token[max_chars:]
        if len(lines) >= max_lines:
            overflowed = bool(tokens[token_idx].strip()) or token_idx < len(tokens) - 1
            break

    if not overflowed and current:
        lines.append(current.rstrip())

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        overflowed = True

    if overflowed and lines:
        truncated = lines[-1][:max(1, max_chars - 3)].rstrip()
        lines[-1] = f"{truncated}..."

    return "\n".join(lines)


def load_or_compute_max_projection(
    czi_path: str,
    cache_dir: str,
    use_cache: bool = True,
) -> tuple[np.ndarray, 'CziImage']:
    """
    Load or compute max projection for a microscopy file.
    
    Checks for cached file FIRST to avoid loading large CZI files unnecessarily.
    
    Returns:
        (max_proj_cyx, czi_image)
    """
    czi_name = microscopy_image_stem(czi_path)
    cache_file = os.path.join(cache_dir, f"{czi_name}_max.tif")
    
    # Check cache BEFORE loading CZI to save memory
    if use_cache and os.path.exists(cache_file):
        print(f"  Loading cached max projection: {cache_file}")
        try:
            max_proj = tifffile.imread(cache_file)
            if not _is_valid_max_projection(max_proj):
                raise ValueError(f"invalid cached projection shape: {max_proj.shape}")
            # Still need CZI for metadata, but don't load image data
            czi = CziImage(czi_path)
            return max_proj, czi
        except Exception as e:
            print(f"  WARNING: Corrupt max-projection cache, rebuilding: {e}")
            try:
                os.remove(cache_file)
            except OSError:
                pass
    
    # No cache - need to compute max projection
    print(f"  Loading microscopy file and computing max projection...")
    print(f"  (This may take a while for large files. Consider pre-caching with cache_max_projections.py)")
    
    czi = CziImage(czi_path)
    
    # Check if image is very large and warn user
    estimated_size_gb = (czi.C * czi.Z * czi.Y * czi.X * 2) / (1024**3)  # Assume 16-bit
    if estimated_size_gb > 2.0:
        print(f"  WARNING: Large image (~{estimated_size_gb:.1f} GB). If this crashes, run:")
        print(f"    python cache_max_projections.py {Path(czi_path).parent} --cache_dir {cache_dir}")
    
    try:
        max_proj = czi.zmax()
    except MemoryError:
        print(f"\n  ERROR: Out of memory! Pre-cache the max projection first:")
        print(f"    python cache_max_projections.py {Path(czi_path).parent} --cache_dir {cache_dir}")
        raise
    if not _is_valid_max_projection(max_proj):
        raise ValueError(f"Computed invalid max projection shape: {max_proj.shape}")
    
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        _write_tiff_atomic(cache_file, max_proj, zstd_level=6)
        print(f"  Cached to: {cache_file}")
    
    return max_proj, czi


def compute_colocalization_for_file(
    czi_path: str,
    cache_dir: str,
    coloc_channels: tuple[int, int],
    use_cache: bool = True,
) -> np.ndarray:
    """
    Compute colocalization heatmap for a CZI file.
    
    Reads only the two requested 3D channels and caches the result for
    subsequent runs.
    
    Args:
        czi_path: Path to CZI file
        cache_dir: Directory for caching results
        coloc_channels: Tuple of (channel1, channel2) for colocalization
        use_cache: Whether to use/create cached results
        
    Returns:
        coloc_2d: 2D max-projected colocalization heatmap (YX)
    """
    czi_name = microscopy_image_stem(czi_path)
    ch1, ch2 = coloc_channels
    cache_file = os.path.join(cache_dir, f"{czi_name}_coloc_ch{ch1}_ch{ch2}.tif")
    
    # Check cache first
    if use_cache and os.path.exists(cache_file):
        print(f"  Loading cached colocalization: {cache_file}")
        return tifffile.imread(cache_file)
    
    # Need the two requested 3D channels and compute colocalization
    print(f"  Computing colocalization for channels {ch1} and {ch2}...")
    
    czi = CziImage(czi_path)
    
    # Read only the channels needed for colocalization instead of the full volume.
    coloc_input = np.stack(
        [czi.read_channel(ch1), czi.read_channel(ch2)],
        axis=0,
    )
    coloc_2d = compute_coloc_max_projection(coloc_input, (0, 1))
    
    # Cache the result
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        # Convert to float32 for storage (preserves precision)
        tifffile.imwrite(
            cache_file, 
            coloc_2d.astype(np.float32),
            compression='zstd', 
            compressionargs={'level': 6}
        )
        print(f"  Cached colocalization to: {cache_file}")
    
    return coloc_2d


def should_rerun_segmentation(
    seg_dir: str,
    sample_name: str,
    config: ReportConfig,
) -> bool:
    """
    Check if segmentation should be re-run based on parameter changes.
    
    Compares the current segmentation settings against saved parameters.
    """
    params_file = os.path.join(seg_dir, f"{sample_name}_params.json")
    mask_file = os.path.join(seg_dir, f"{sample_name}_seg.tif")
    
    # If mask doesn't exist, need to segment
    if not os.path.exists(mask_file):
        return True
    
    # If params file doesn't exist, re-run to be safe
    if not os.path.exists(params_file):
        return True
    
    # Load saved parameters and compare segmentation settings
    try:
        with open(params_file, 'r') as f:
            saved_params = json.load(f)

        current_params = get_segmentation_params(config)
        for key, current_value in current_params.items():
            if saved_params.get(key) != current_value:
                print(
                    f"  Segmentation parameter '{key}' changed "
                    f"({saved_params.get(key)} -> {current_value}), re-segmenting"
                )
                return True

    except (json.JSONDecodeError, IOError):
        return True
    
    return False


def get_segmentation_params(config: ReportConfig) -> dict:
    """Return parameters that affect cached segmentation masks."""
    return {
        'cellpose_model': DEFAULT_MODEL_NAME,
        'nuclear_channel': config.nuclear_channel,
        'cellpose_diameter': config.cellpose_diameter,
        'cellpose_flow_threshold': config.cellpose_flow_threshold,
        'cellpose_cellprob_threshold': config.cellpose_cellprob_threshold,
        'min_area_pixels': config.min_area_pixels,
        'max_area_pixels': config.max_area_pixels,
        'nuclear_segmentation_blur_sigma': config.nuclear_segmentation_blur_sigma,
    }


def prepare_nuclear_segmentation_image(
    max_proj: np.ndarray,
    nuclear_channel: int,
    blur_sigma: float,
) -> np.ndarray:
    """Create the 2D nuclear image used only for segmentation."""
    nuclear_img = max_proj[nuclear_channel]
    if blur_sigma <= 0:
        return nuclear_img

    return gaussian_filter(
        nuclear_img.astype(np.float32, copy=False),
        sigma=(blur_sigma, blur_sigma),
        mode='nearest',
    )


def segment_image(
    max_proj: np.ndarray,
    sample_name: str,
    config: ReportConfig,
    seg_dir: str,
) -> np.ndarray:
    """
    Segment nuclei from max projection, with caching.
    
    Returns:
        Segmentation mask (H, W) with integer labels
    """
    os.makedirs(seg_dir, exist_ok=True)
    mask_file = os.path.join(seg_dir, f"{sample_name}_seg.tif")
    
    # Check if we need to re-run segmentation
    if not should_rerun_segmentation(seg_dir, sample_name, config):
        print(f"  Loading cached segmentation: {mask_file}")
        return tifffile.imread(mask_file)
    
    # Run Cellpose segmentation
    print(f"  Running Cellpose segmentation...")
    nuclear_img = prepare_nuclear_segmentation_image(
        max_proj,
        config.nuclear_channel,
        config.nuclear_segmentation_blur_sigma,
    )
    if config.nuclear_segmentation_blur_sigma > 0:
        print(
            "  Applying 2D Gaussian blur to segmentation input "
            f"(sigma={config.nuclear_segmentation_blur_sigma:g} px)"
        )
    
    mask = segment_nuclei(
        nuclear_img,
        sample_name,
        save_dir=seg_dir,
        diameter=config.cellpose_diameter,
        min_area=config.min_area_pixels,
        max_area=config.max_area_pixels,
        flow_threshold=config.cellpose_flow_threshold,
        cellprob_threshold=config.cellpose_cellprob_threshold,
        save_mask_tiff=True,
        verbose=True,
    )
    
    # Need for knowing when to use cached segmentation
    save_segmentation_params(seg_dir, sample_name, config)

    return mask


def match_genotype(filename: str, genotypes: list[str]) -> Optional[str]:
    """
    Match a filename to a genotype based on substring matching.
    
    Args:
        filename: Name of the file (without extension)
        genotypes: List of genotype patterns to match
        
    Returns:
        Matched genotype string, or None if no match
    """
    for genotype in genotypes:
        if genotype in filename:
            return genotype
    return None


def process_single_file(
    czi_path: str,
    config: ReportConfig,
    existing_ids: set[str],
) -> tuple[list[NucleusCrop], np.ndarray, CziImage]:
    """
    Process a single CZI file: load, segment, extract nuclei.
    
    Returns:
        (nucleus_crops, max_proj, czi_image)
    """
    cache_dir, seg_dir = get_cache_paths(config)
    
    # Load/compute max projection
    max_proj, czi = load_or_compute_max_projection(
        czi_path, cache_dir, config.use_cache
    )
    
    # Segment nuclei
    mask = segment_image(max_proj, czi.name, config, seg_dir)
    
    # Match genotype
    genotype = match_genotype(czi.name, config.genotypes)
    if genotype is None:
        print(f"  Warning: No genotype match for {czi.name}, skipping")
        return [], max_proj, czi
    
    # Get pixel sizes
    pixel_size_x_um = czi.scale.get('X', 1e-6) * 1e6
    pixel_size_y_um = czi.scale.get('Y', 1e-6) * 1e6
    
    # Extract nucleus crops
    nucleus_crops = extract_nuclei_from_mask(
        mask=mask,
        image_data=max_proj,
        source_filename=czi.name,
        genotype=genotype,
        box_size_um=config.box_size_um,
        pixel_size_um=(pixel_size_y_um, pixel_size_x_um),
        channel_wavelengths=czi.channel_wavelengths,
        channel_colors=czi.channel_colors,
        visible_channels=config.visible_channels,
        exclude_border=config.exclude_border,
        existing_ids=existing_ids,
    )
    
    print(f"  Extracted {len(nucleus_crops)} nuclei for genotype '{genotype}'")
    
    return nucleus_crops, max_proj, czi


def load_segmentation_mask(config: ReportConfig, source_filename: str) -> Optional[np.ndarray]:
    """Load the segmentation mask for a given source file."""
    _, seg_dir = get_cache_paths(config)
    mask_file = os.path.join(seg_dir, f"{source_filename}_seg.tif")
    
    if os.path.exists(mask_file):
        return tifffile.imread(mask_file)
    return None


def extract_nucleus_contour(
    mask: np.ndarray,
    nucleus_id: int,
    crop_meta,
) -> Optional[np.ndarray]:
    """
    Extract the contour of a nucleus within a crop region.
    
    Returns array of (y, x) coordinates for the contour, or None if not found.
    """
    from skimage import measure
    
    y_start, y_end, x_start, x_end = get_crop_bounds(mask.shape, crop_meta)
    
    # Extract mask crop
    mask_crop = mask[y_start:y_end, x_start:x_end]
    
    # Create binary mask for this nucleus
    binary_mask = (mask_crop == nucleus_id).astype(np.uint8)
    
    if binary_mask.sum() == 0:
        return None
    
    # Find contours on the raw binary mask
    contours = measure.find_contours(binary_mask, 0.5)
    
    if len(contours) == 0:
        return None
    
    # Return the largest contour
    largest_contour = max(contours, key=len)
    return largest_contour


def smooth_contour_bspline(contour: np.ndarray, num_points: int = 200, smoothing: float = 50.0) -> Optional[np.ndarray]:
    """
    Smooth a contour using B-spline interpolation.
    
    Args:
        contour: Array of (y, x) coordinates, shape (N, 2)
        num_points: Number of points in the output smooth curve
        smoothing: Smoothing factor for the spline (higher = smoother)
    
    Returns:
        Smoothed contour as (num_points, 2) array, or None if smoothing fails
    """
    from scipy.interpolate import splprep, splev
    
    if contour is None or len(contour) < 10:
        return None
    
    # Heavily downsample the contour first to remove pixel-level detail
    # This helps the spline focus on the overall shape
    target_points = min(80, len(contour) // 3)
    if len(contour) > target_points:
        indices = np.linspace(0, len(contour) - 1, target_points).astype(int)
        contour = contour[indices]
    
    y = contour[:, 0]
    x = contour[:, 1]
    
    try:
        # Fit periodic B-spline (per=True means closed curve)
        # s = smoothing factor - HIGH value for smooth curves
        # k = spline degree (cubic)
        tck, u = splprep([x, y], s=smoothing, per=True, k=3)
        
        # Evaluate spline at evenly spaced points
        u_new = np.linspace(0, 1, num_points)
        x_smooth, y_smooth = splev(u_new, tck)
        
        return np.column_stack([y_smooth, x_smooth])
    
    except Exception:
        # Fallback: return original contour if spline fails
        return contour


def draw_nucleus_outline(
    ax,
    contour: np.ndarray,
    color='white',
    linewidth=1.0,
    linestyle='dashed',
    smooth=True,
):
    """
    Draw a smooth dashed nucleus outline using B-spline interpolation.
    
    Args:
        ax: Matplotlib axes
        contour: Raw contour from extract_nucleus_contour (will be smoothed)
        color: Line color
        linewidth: Line width
        linestyle: 'dashed', 'solid', or custom dash tuple
    """
    if contour is None or len(contour) < 5:
        return
    
    if smooth:
        smoothed_contour = smooth_contour_bspline(contour, num_points=30, smoothing=200.0)
        if smoothed_contour is not None:
            contour = smoothed_contour
    
    # Close the contour
    contour_closed = np.vstack([contour, contour[0]])
    
    # Define dash pattern: (dash_length, gap_length) in points
    if linestyle == 'dashed':
        dash_pattern = (0, (5, 4))  # 5pt dash, 4pt gap
    else:
        dash_pattern = linestyle
    
    # Draw the smooth curve
    ax.plot(
        contour_closed[:, 1],  # x coordinates
        contour_closed[:, 0],  # y coordinates
        color=color,
        linewidth=linewidth,
        linestyle=dash_pattern,
        solid_capstyle='round',
        dash_capstyle='round',
        solid_joinstyle='round',
        dash_joinstyle='round',
        alpha=0.9,
        zorder=10,
    )


def compute_valid_crop_mask(crop_meta, full_shape: tuple[int, int]) -> np.ndarray:
    """
    Compute which pixels in a fixed-size crop come from real image data.

    Returns:
        Boolean mask of shape (box_h, box_w), True where pixels are from the source image.
    """
    full_h, full_w = full_shape
    center_y, center_x = crop_meta.center_pixel
    box_h, box_w = crop_meta.box_size_pixels

    y_start = center_y - box_h // 2
    y_end = y_start + box_h
    x_start = center_x - box_w // 2
    x_end = x_start + box_w

    src_y_start = max(0, y_start)
    src_y_end = min(full_h, y_end)
    src_x_start = max(0, x_start)
    src_x_end = min(full_w, x_end)

    dst_y_start = src_y_start - y_start
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = src_x_start - x_start
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    valid = np.zeros((box_h, box_w), dtype=bool)
    if src_y_end > src_y_start and src_x_end > src_x_start:
        valid[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = True
    return valid


def apply_padding_color(img: np.ndarray, valid_mask: np.ndarray, pad_color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Recolor padded pixels to a visible background color for clarity in reports.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    if valid_mask.shape != img.shape[:2]:
        return img
    if np.all(valid_mask):
        return img

    out = img.copy()
    out[~valid_mask] = np.array(pad_color, dtype=out.dtype)
    return out


def build_contour_cache(
    nucleus_crops: list[NucleusCrop],
    mask_dict: dict[str, np.ndarray],
) -> dict[str, Optional[np.ndarray]]:
    """Compute and smooth each nucleus contour once for PDF reuse."""
    contour_cache: dict[str, Optional[np.ndarray]] = {}

    for nuc_crop in nucleus_crops:
        mask = mask_dict.get(nuc_crop.source_filename)
        if mask is None:
            contour_cache[nuc_crop.unique_id] = None
            continue

        contour = extract_nucleus_contour(mask, nuc_crop.nucleus_id, nuc_crop.crop_meta)
        if contour is not None and len(contour) >= 10:
            smoothed_contour = smooth_contour_bspline(contour, num_points=30, smoothing=200.0)
            if smoothed_contour is not None:
                contour = smoothed_contour
        contour_cache[nuc_crop.unique_id] = contour

    return contour_cache

def generate_multipage_pdf(
    nucleus_crops: list[NucleusCrop],
    image_data_dict: dict[str, np.ndarray],
    czi_info_dict: dict[str, CziImage],
    genotype: str,
    output_path: str,
    config: ReportConfig,
    color_scheme: Optional[ColorScheme] = None,
    mask_dict: Optional[dict[str, np.ndarray]] = None,
    coloc_data_dict: Optional[dict[str, np.ndarray]] = None,
):
    """
    Generate a multi-page PDF report for a genotype.
    
    Layout: Organized by source file, with file headers.
    Each file section shows nuclei in a 3-column × N-row grid.
    
    Args:
        nucleus_crops: List of NucleusCrop objects for this genotype
        image_data_dict: Dict mapping source_filename -> max-projected CYX array
        czi_info_dict: Dict mapping source_filename -> CziImage for metadata
        genotype: Genotype label
        output_path: Output PDF path
        config: Report configuration
        color_scheme: Color scheme (auto-assigned if None)
        mask_dict: Dict mapping source_filename -> segmentation mask (for outlines)
        coloc_data_dict: Dict mapping source_filename -> 2D colocalization heatmap
    """
    if len(nucleus_crops) == 0:
        print(f"  No nuclei for genotype '{genotype}', skipping PDF")
        return
    
    # Group nuclei by source file, then sort by nucleus_id within each file
    nuclei_by_file: dict[str, list[NucleusCrop]] = {}
    for nuc in nucleus_crops:
        if nuc.source_filename not in nuclei_by_file:
            nuclei_by_file[nuc.source_filename] = []
        nuclei_by_file[nuc.source_filename].append(nuc)
    
    # Sort files alphabetically, and nuclei within each file by nucleus_id
    sorted_files = sorted(nuclei_by_file.keys())
    for filename in sorted_files:
        nuclei_by_file[filename].sort(key=lambda n: n.nucleus_id)
    
    # Build flat list in file-grouped order
    nucleus_crops_sorted = []
    for filename in sorted_files:
        nucleus_crops_sorted.extend(nuclei_by_file[filename])
    
    # Load segmentation masks if needed for outlines
    if mask_dict is None and config.show_outline:
        mask_dict = {}
        for filename in sorted_files:
            mask = load_segmentation_mask(config, filename)
            if mask is not None:
                mask_dict[filename] = mask
    elif mask_dict is None:
        mask_dict = {}

    contour_cache = {}
    if config.show_outline and mask_dict:
        contour_cache = build_contour_cache(nucleus_crops_sorted, mask_dict)
    
    # Layout configuration from config
    nucleus_cols = config.nucleus_cols
    rows_per_col = config.rows_per_page
    nuclei_per_page = nucleus_cols * rows_per_col
    
    # Get channel info from first nucleus
    first_nuc = nucleus_crops_sorted[0]
    active_channels = first_nuc.crop_meta.get_active_channels()
    include_merged = len(active_channels) > 1
    # channels + optional merge + coloc (if enabled)
    has_coloc = coloc_data_dict is not None and len(coloc_data_dict) > 0
    images_per_nucleus = len(active_channels) + (1 if include_merged else 0) + (1 if has_coloc else 0)
    
    # Get image dimensions
    first_image = image_data_dict[first_nuc.source_filename]
    crop_data = extract_crop_data(first_image, first_nuc.crop_meta)
    img_h, img_w = crop_data.shape[1], crop_data.shape[2]
    
    # Configure panels with tighter spacing
    panel_config = PanelConfig(
        show_channel_labels=False,
        merged_label=config.merged_label,
        gap_pixels=2,
        gap_color=(255, 255, 255),
        border_pixels=1,
        scale_bar=ScaleBarConfig(
            length_um=config.scale_bar_um,
            font_size=config.scale_bar_fontsize,
            position='bottom_right',
            show_label=True,
        )
    )
    
    # Build pages while accounting for file headers that also consume grid slots.
    # Without this, nuclei can be dropped when headers are inserted at render time.
    page_entries: list[list[tuple[NucleusCrop, bool]]] = []
    n_nuclei = len(nucleus_crops_sorted)
    nuc_idx = 0
    while nuc_idx < n_nuclei:
        slots_used = 0
        prev_file_on_page = None
        entries: list[tuple[NucleusCrop, bool]] = []

        while nuc_idx < n_nuclei:
            nuc = nucleus_crops_sorted[nuc_idx]
            is_new_file = (nuc.source_filename != prev_file_on_page)
            slots_needed = 1 + (1 if is_new_file else 0)

            # Normal case: header row + nucleus row both fit.
            if slots_used + slots_needed <= nuclei_per_page:
                entries.append((nuc, is_new_file))
                slots_used += slots_needed
                prev_file_on_page = nuc.source_filename
                nuc_idx += 1
                continue

            # Edge case: very small layouts (e.g., 1x1) cannot fit both header and nucleus.
            # Keep the nucleus and suppress the dedicated header row so content is never dropped.
            if slots_used == 0 and is_new_file and slots_used + 1 <= nuclei_per_page:
                entries.append((nuc, False))
                slots_used += 1
                prev_file_on_page = nuc.source_filename
                nuc_idx += 1
            break

        if len(entries) == 0:
            raise RuntimeError("Pagination failed to place any nuclei on page; check layout settings.")
        page_entries.append(entries)

    n_pages = len(page_entries)
    
    print(f"  Generating {n_pages} page(s) for {n_nuclei} nuclei from {len(sorted_files)} files...")
    
    # Create PDF
    with PdfPages(output_path) as pdf:
        for page_idx, page_nuclei in enumerate(page_entries):
            
            # Figure size calculation
            img_width_in = img_w / config.dpi
            img_height_in = img_h / config.dpi
            is_first_page = (page_idx == 0)
            
            # Add space for labels and gaps between nucleus groups
            label_width_in = 0.4  # Space for nucleus ID labels
            col_gap_in = 0.15  # Gap between nucleus groups
            coloc_scale_gap_in = 0.03 if has_coloc else 0.0
            coloc_scale_width_in = (img_width_in * 0.14) if has_coloc else 0.0
            page_top_margin = 0.08 if is_first_page else 0.04
            
            group_content_width_in = (
                img_width_in * images_per_nucleus
                + label_width_in
                + coloc_scale_gap_in
                + coloc_scale_width_in
            )
            fig_width = group_content_width_in * nucleus_cols + col_gap_in * (nucleus_cols - 1)
            fig_height = img_height_in * rows_per_col + 0.5  # Extra for title
            
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=config.dpi)
            fig.patch.set_facecolor('white')
            
            scale_bar_config = panel_config.scale_bar or ScaleBarConfig()
            page_labels_drawn = False
            
            # Render nuclei with file headers taking their own row
            slot_idx = 0  # Tracks position including headers
            prev_file = None
            
            for nuc_crop, should_draw_header in page_nuclei:
                current_file = nuc_crop.source_filename
                
                # If new file, draw header row first
                if should_draw_header:
                    col_group = slot_idx // rows_per_col
                    row_in_group = slot_idx % rows_per_col
                    
                    # Only draw if we have room on this page
                    if col_group < nucleus_cols:
                        # Calculate header position
                        group_width = group_content_width_in / fig_width
                        group_start_x = col_group * (group_width + col_gap_in / fig_width)
                        row_height = img_height_in / fig_height
                        row_start_y = 1.0 - page_top_margin - (row_in_group + 1) * row_height
                        
                        # Draw filename header
                        filename_stem = Path(current_file).stem
                        header_fontsize = 6
                        header_width_in = max(
                            group_content_width_in - label_width_in - 0.04,
                            img_width_in * 0.75,
                        )
                        header_label = filename_stem
                        if len(active_channels) == 1:
                            header_label = _wrap_filename_header(
                                filename_stem,
                                available_width_in=header_width_in,
                                fontsize=header_fontsize,
                                max_lines=3,
                            )
                        fig.text(
                            group_start_x + label_width_in / fig_width,
                            row_start_y + row_height * 0.5,
                            header_label,
                            fontsize=header_fontsize,
                            ha='left',
                            va='center',
                            fontweight='bold',
                            family='sans-serif',
                            color='black',
                            linespacing=1.0,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9),
                        )
                    
                    slot_idx += 1
                    prev_file = current_file
                elif prev_file is None:
                    prev_file = current_file
                
                # Now draw the nucleus
                col_group = slot_idx // rows_per_col
                row_in_group = slot_idx % rows_per_col
                
                # Defensive guard: pagination should prevent overflow.
                if col_group >= nucleus_cols:
                    break
                
                # Get image data and render
                image_data = image_data_dict[nuc_crop.source_filename]
                crop_data = extract_crop_data(image_data, nuc_crop.crop_meta)
                valid_crop_mask = compute_valid_crop_mask(
                    nuc_crop.crop_meta,
                    (image_data.shape[1], image_data.shape[2]),
                )
                
                # Get nucleus contour for outline
                contour = contour_cache.get(nuc_crop.unique_id)
                
                # Create FigurePanel
                panel = FigurePanel(
                    crop_data=crop_data,
                    crop_meta=nuc_crop.crop_meta,
                    color_scheme=color_scheme,
                    config=panel_config,
                    channel_names=config.channel_labels,
                )
                
                # Render images
                individual = panel.render_individual_channels()
                all_images = list(individual)
                if include_merged:
                    all_images.append(panel.render_merged())
                
                # Add colocalization panel if enabled
                coloc_scale_info = None
                if has_coloc and nuc_crop.source_filename in coloc_data_dict:
                    coloc_full = coloc_data_dict[nuc_crop.source_filename]
                    y_start, y_end, x_start, x_end = get_crop_bounds(
                        coloc_full.shape,
                        nuc_crop.crop_meta,
                    )
                    coloc_crop = coloc_full[y_start:y_end, x_start:x_end]
                    
                    # Render colocalization heatmap
                    coloc_rgb, coloc_scale_info = render_coloc_heatmap(
                        coloc_crop, 
                        vmax=config.coloc_vmax,
                        colormap=config.coloc_colormap,
                        return_scale=True,
                    )
                    all_images.append(coloc_rgb)

                # Make padded regions visually explicit in exported PDF panels.
                all_images = [
                    apply_padding_color(img, valid_crop_mask, pad_color=(255, 255, 255))
                    for img in all_images
                ]
                
                pixel_size_um = nuc_crop.crop_meta.pixel_size_um[1]
                
                # Calculate axes positions
                group_width = group_content_width_in / fig_width
                group_start_x = col_group * (group_width + col_gap_in / fig_width)
                
                row_height = img_height_in / fig_height
                row_start_y = 1.0 - page_top_margin - (row_in_group + 1) * row_height
                
                # Add nucleus ID label
                label_x = group_start_x + 0.005
                label_y = row_start_y + row_height / 2
                fig.text(
                    label_x, label_y, f"#{nuc_crop.nucleus_id}",
                    fontsize=5,
                    ha='left',
                    va='center',
                    fontweight='bold',
                    family='monospace',
                )
                
                # Plot each image
                img_start_x = group_start_x + label_width_in / fig_width
                single_img_width = img_width_in / fig_width
                label_texts = panel.channel_names + ([panel_config.merged_label] if include_merged else [])
                if has_coloc:
                    label_texts.append("Colocalization")
                
                for img_idx, img in enumerate(all_images):
                    ax_x = img_start_x + img_idx * single_img_width
                    ax_y = row_start_y
                    ax_w = single_img_width * 0.98
                    ax_h = row_height * 0.98
                    
                    ax = fig.add_axes([ax_x, ax_y, ax_w, ax_h])
                    ax.imshow(img)
                    ax.axis('off')
                    
                    # Draw nucleus outline on all images (if enabled)
                    if config.show_outline and contour is not None:
                        draw_nucleus_outline(ax, contour, color='white', linewidth=0.8, smooth=False)
                    
                    # Add scale bar (only on first nucleus of first page, first image)
                    if is_first_page and slot_idx == 1 and img_idx == 0:
                        _add_scale_bar(ax, pixel_size_um, (img_h, img_w), scale_bar_config, show_label=True)
                    else:
                        _add_scale_bar(ax, pixel_size_um, (img_h, img_w), scale_bar_config, show_label=False)

                    if (
                        is_first_page
                        and not page_labels_drawn
                        and col_group == 0
                        and img_idx < len(label_texts)
                    ):
                        label_y = row_start_y + row_height + min(
                            page_top_margin * 0.12,
                            row_height * 0.08,
                        )
                        fig.text(
                            ax_x + ax_w * 0.5,
                            label_y,
                            label_texts[img_idx],
                            fontsize=11,
                            ha='center',
                            va='center',
                            fontweight='bold',
                            color='black',
                        )
                        if img_idx == len(label_texts) - 1:
                            page_labels_drawn = True

                    if coloc_scale_info is not None and img_idx == len(all_images) - 1:
                        scale_x = ax_x + ax_w + (coloc_scale_gap_in / fig_width)
                        scale_w = coloc_scale_width_in / fig_width
                        scale_ax = fig.add_axes([scale_x, ax_y, scale_w, ax_h])
                        scale_ax.imshow(
                            render_coloc_scale_bar(
                                height=max(64, img.shape[0]),
                                width=max(8, int(img.shape[1] * 0.08)),
                                colormap=config.coloc_colormap,
                            )
                        )
                        scale_ax.axis('off')
                        scale_ax.text(
                            1.15,
                            1.0,
                            f"{coloc_scale_info['vmax']:.2f}",
                            transform=scale_ax.transAxes,
                            ha='left',
                            va='top',
                            fontsize=4,
                            color='black',
                        )
                        scale_ax.text(
                            1.15,
                            0.0,
                            f"{coloc_scale_info['vmin']:.2f}",
                            transform=scale_ax.transAxes,
                            ha='left',
                            va='bottom',
                            fontsize=4,
                            color='black',
                        )
                
                slot_idx += 1
            # Add title
            title = f"{genotype} - Page {page_idx + 1}/{n_pages}"
            fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
            
            # Save page
            pdf.savefig(fig, dpi=config.dpi, bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)
            
            print(f"    Page {page_idx + 1}/{n_pages} complete")
    
    print(f"  Saved: {output_path}")
    
def _add_scale_bar(
    ax: plt.Axes,
    pixel_size_um: float,
    image_shape: tuple[int, int],
    config: ScaleBarConfig,
    show_label: bool = True,
):
    """Add a scale bar to matplotlib axes."""
    from matplotlib.patches import Rectangle
    
    h, w = image_shape
    bar_length_px = config.length_um / pixel_size_um
    bar_height_px = h * config.height_fraction
    
    margin_x = w * config.margin_fraction
    margin_y = h * config.margin_fraction
    
    # Calculate position
    if 'right' in config.position:
        x = w - margin_x - bar_length_px
    else:
        x = margin_x
    
    if 'bottom' in config.position:
        y = h - margin_y - bar_height_px
    else:
        y = margin_y
    
    # Draw bar
    rect = Rectangle(
        (x, y), bar_length_px, bar_height_px,
        facecolor=config.color, edgecolor='none'
    )
    ax.add_patch(rect)
    
    # Add label if requested
    if show_label:
        label_x = x + bar_length_px / 2
        if 'bottom' in config.position:
            label_y = y - bar_height_px * 0.5
            va = 'bottom'
        else:
            label_y = y + bar_height_px * 1.5
            va = 'top'
        
        ax.text(
            label_x, label_y, f"{config.length_um:.0f} µm",
            color=config.label_color,
            fontsize=config.font_size,
            ha='center', va=va,
            fontweight='bold'
        )


def save_metadata_csv(
    all_nucleus_crops: dict[str, list[NucleusCrop]],
    image_data_dict: dict[str, np.ndarray],
    output_path: str,
):
    """
    Save nucleus metadata to CSV file.
    
    Includes: nucleus_id, genotype, source_file, centroid coordinates, area,
    mean intensity per channel, and contrast settings per channel.
    """
    import csv
    
    # Collect all rows
    rows = []
    
    for genotype, nucleus_crops in all_nucleus_crops.items():
        for nuc_crop in nucleus_crops:
            image_data = image_data_dict.get(nuc_crop.source_filename)
            
            row = {
                'unique_id': nuc_crop.unique_id,
                'nucleus_id': nuc_crop.nucleus_id,
                'genotype': nuc_crop.genotype,
                'source_file': nuc_crop.source_filename,
                'centroid_y_um': nuc_crop.centroid_um[0],
                'centroid_x_um': nuc_crop.centroid_um[1],
                'area_um2': nuc_crop.area_um2,
                'area_pixels': nuc_crop.area_pixels,
            }
            
            # Add contrast settings for each channel
            for ch, contrast in enumerate(nuc_crop.crop_meta.channel_contrasts):
                row[f'contrast_min_ch{ch}'] = contrast.min_val
                row[f'contrast_max_ch{ch}'] = contrast.max_val
            
            # Add mean intensities if image data available
            if nuc_crop.mean_intensities:
                for ch, mean_intensity in enumerate(nuc_crop.mean_intensities):
                    row[f'mean_intensity_ch{ch}'] = mean_intensity
            elif image_data is not None:
                crop_data = extract_crop_data(image_data, nuc_crop.crop_meta)
                for ch in range(crop_data.shape[0]):
                    row[f'mean_intensity_ch{ch}'] = float(np.mean(crop_data[ch]))
            
            rows.append(row)
    
    # Write CSV
    if len(rows) > 0:
        fieldnames = list(rows[0].keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Saved metadata: {output_path}")


def save_run_parameters(config: ReportConfig, output_dir: str, timestamp: Optional[str] = None):
    """Save all run parameters to JSON file."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    params_file = os.path.join(output_dir, f"run_parameters_{timestamp}.json")
    
    params = config.to_dict()
    params['run_timestamp'] = timestamp
    
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"Saved run parameters: {params_file}")

def generate_qc_report_with_progress(config: ReportConfig, progress_tracker=None):
    """
    Generate QC report with progress callbacks for UI updates.
    
    Args:
        config: ReportConfig with all settings
        progress_tracker: Optional object with update(file_idx, filename, stage), 
                         log(message), and set_stage(name) methods
    """
    def log(msg):
        print(msg)
        if progress_tracker:
            progress_tracker.log(msg)
    
    def update_progress(idx, name, stage):
        if progress_tracker:
            progress_tracker.update(idx, name, stage)
    
    def set_stage(name):
        if progress_tracker:
            progress_tracker.set_stage(name)
    
    log("=" * 60)
    log("AUTOMATED QC REPORT GENERATOR")
    log("=" * 60)
    log(f"Input directory: {config.input_dir}")
    log(f"Output directory: {config.output_dir}")
    log(f"Genotypes: {config.genotypes}")
    log(f"Nuclear channel: {config.nuclear_channel}")
    log(f"Box size: {config.box_size_um} µm")
    if config.enable_coloc and config.coloc_channels:
        log(f"Colocalization: channels {config.coloc_channels}")
    log("=" * 60)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_run_parameters(config, config.output_dir, timestamp=timestamp)
    
    # Find and filter CZI files
    czi_files = find_supported_image_files(config.input_dir)
    
    # Filter by genotype BEFORE processing
    czi_files = [f for f in czi_files 
                 if match_genotype(microscopy_image_stem(f), config.genotypes) is not None]
    
    if len(czi_files) == 0:
        log(f"No supported microscopy files match genotypes: {config.genotypes}")
        return
    
    log(f"Processing {len(czi_files)} files matching genotypes: {config.genotypes}")
    
    # Initialize data structures
    set_stage("🔬 Processing images...")
    all_nucleus_crops: dict[str, list[NucleusCrop]] = {g: [] for g in config.genotypes}
    image_data_dict: dict[str, np.ndarray] = {}
    coloc_data_dict: dict[str, np.ndarray] = {}  # Colocalization heatmaps per file
    czi_info_dict: dict[str, CziImage] = {}
    existing_ids: set[str] = set()
    segmentation_qc_paths: list[str] = []
    
    cache_dir, seg_dir = get_cache_paths(config)

    # Initialize prefetcher
    prefetcher = ImagePrefetcher(
        czi_files=czi_files,
        cache_dir=cache_dir,
        use_cache=config.use_cache,
        prefetch_count=2,
    )

    timer = PipelineTimer()

    for i, czi_path in enumerate(czi_files):
        filename = os.path.basename(czi_path)
        log(f"\n[{i+1}/{len(czi_files)}] {filename}")
        
        try:
            # Stage 1: Load/compute max projection
            update_progress(i, filename, "📂 Loading image...")
            timer.start("1. Load/projection")
            max_proj, czi = prefetcher.get(czi_path)
            timer.stop()
            log(f"  Loaded: {max_proj.shape}")
            
            # Stage 1.5: Compute colocalization (if enabled)
            coloc_data = None
            if config.enable_coloc and config.coloc_channels is not None:
                update_progress(i, filename, "🔬 Computing colocalization...")
                timer.start("1.5 Colocalization")
                coloc_data = compute_colocalization_for_file(
                    czi_path=czi_path,
                    cache_dir=cache_dir,
                    coloc_channels=config.coloc_channels,
                    use_cache=config.use_cache,
                )
                timer.stop()
                log(f"  Colocalization computed: {coloc_data.shape}")
            
            # Stage 2: Segment nuclei
            update_progress(i, filename, "🧠 Segmenting nuclei...")
            timer.start("2. Segment nuclei")
            mask = segment_image(max_proj, czi.name, config, seg_dir)
            timer.stop()
            n_nuclei = len(np.unique(mask)) - 1
            qc_path = os.path.join(seg_dir, f"{czi.name}_seg.png")
            if os.path.exists(qc_path):
                segmentation_qc_paths.append(qc_path)
            log(f"  Segmented: {n_nuclei} nuclei")
            
            # Stage 3: Extract crops
            update_progress(i, filename, "✂️ Extracting nucleus crops...")
            timer.start("3. Extract crops")
            genotype = match_genotype(czi.name, config.genotypes)
            
            pixel_size_x_um = czi.scale.get('X', 1e-6) * 1e6
            pixel_size_y_um = czi.scale.get('Y', 1e-6) * 1e6
            
            nucleus_crops = extract_nuclei_from_mask(
                mask=mask,
                image_data=max_proj,
                source_filename=czi.name,
                genotype=genotype,
                box_size_um=config.box_size_um,
                pixel_size_um=(pixel_size_y_um, pixel_size_x_um),
                channel_wavelengths=czi.channel_wavelengths,
                channel_colors=czi.channel_colors,
                visible_channels=config.visible_channels,
                exclude_border=config.exclude_border,
                existing_ids=existing_ids,
            )
            timer.stop()

            log(f"  Extracted: {len(nucleus_crops)} crops for '{genotype}'")
            
            all_nucleus_crops[genotype].extend(nucleus_crops)
            image_data_dict[czi.name] = max_proj
            if coloc_data is not None:
                coloc_data_dict[czi.name] = coloc_data
            czi_info_dict[czi.name] = czi
            
        except Exception as e:
            log(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    prefetcher.shutdown()
    if segmentation_qc_paths:
        contact_sheet_paths = save_qc_contact_sheet(
            os.path.join(seg_dir, "segmentation_qc_contact_sheet.png"),
            segmentation_qc_paths,
        )
        for contact_sheet_path in contact_sheet_paths:
            log(f"  Segmentation QC contact sheet: {contact_sheet_path}")
    prefetcher.print_stats()
    timer.print_summary()
    # Report summary
    log("\n" + "=" * 60)
    log("PROCESSING SUMMARY")
    log("=" * 60)
    for genotype in config.genotypes:
        n_nuclei = len(all_nucleus_crops[genotype])
        log(f"  {genotype}: {n_nuclei} nuclei")
    
    # Generate PDFs
    set_stage("📄 Generating PDF reports...")
    log("\n" + "=" * 60)
    log("GENERATING REPORTS")
    log("=" * 60)
    
    for genotype in config.genotypes:
        nucleus_crops = all_nucleus_crops[genotype]
        
        if len(nucleus_crops) == 0:
            log(f"\n[{genotype}] No nuclei, skipping")
            continue
        
        log(f"\n[{genotype}] Processing {len(nucleus_crops)} nuclei...")
        
        # Compute per-file contrast
        nuclei_by_file: dict[str, list[NucleusCrop]] = {}
        for nuc in nucleus_crops:
            if nuc.source_filename not in nuclei_by_file:
                nuclei_by_file[nuc.source_filename] = []
            nuclei_by_file[nuc.source_filename].append(nuc)
        
        # Compute and apply contrast per file
        file_contrast_settings: dict[str, dict[int, tuple[float, float]]] = {}
        
        for source_file, file_nuclei in nuclei_by_file.items():
            image_data = image_data_dict[source_file]
            
            contrast_limits = compute_per_file_contrast(
                image_data,
                percentiles=config.contrast_percentiles,
            )
            
            file_contrast_settings[source_file] = contrast_limits
            apply_file_contrast_to_nuclei(file_nuclei, contrast_limits)
            
            log(f"  {source_file}: {len(file_nuclei)} nuclei")
            for ch, (vmin, vmax) in contrast_limits.items():
                if config.contrast_percentiles and ch < len(config.contrast_percentiles):
                    pct = config.contrast_percentiles[ch]
                elif config.contrast_percentiles:
                    pct = config.contrast_percentiles[-1]
                else:
                    pct = (0.5, 99.5)
                log(f"    Ch{ch} ({pct[0]}-{pct[1]}%): [{vmin:.1f}, {vmax:.1f}]")
        
        # Determine color scheme
        color_scheme = None
        if config.channel_colors:
            try:
                color_scheme = scheme_custom(config.channel_colors, name="Custom_CLI")
                log(f"  Using custom colors: {config.channel_colors}")
                for i, lut in enumerate(color_scheme.channel_luts):
                    log(f"    Ch{i}: {config.channel_colors[i]} -> RGB {lut.color_rgb}")
            except ValueError as e:
                log(f"  Warning: Invalid color in {config.channel_colors}: {e}")
                log(f"  Falling back to auto colors")
                color_scheme = None
        
        # Generate PDF
        update_progress(len(czi_files) - 1, f"{genotype}_report.pdf", f"📄 Generating {genotype} PDF...")
        output_path = os.path.join(config.output_dir, f"{genotype}_report_{timestamp}.pdf")
        generate_multipage_pdf(
            nucleus_crops,
            image_data_dict,
            czi_info_dict,
            genotype,
            output_path,
            config,
            color_scheme=color_scheme,
            coloc_data_dict=coloc_data_dict if config.enable_coloc else None,
        )
    
    # Save metadata CSV
    set_stage("💾 Saving metadata...")
    csv_path = os.path.join(config.output_dir, "nucleus_metadata.csv")
    save_metadata_csv(all_nucleus_crops, image_data_dict, csv_path)
    
    log("\n" + "=" * 60)
    log("COMPLETE!")
    log("=" * 60)
    log(f"Reports saved to: {config.output_dir}")

def save_segmentation_params(
    seg_dir: str,
    sample_name: str,
    config: ReportConfig,
):
    """Save segmentation parameters for later comparison."""
    params_file = os.path.join(seg_dir, f"{sample_name}_params.json")
    
    params = get_segmentation_params(config)
    params['timestamp'] = datetime.now().isoformat()
    
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
