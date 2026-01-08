"""
Automated QC Report Generator.

Orchestrates the full pipeline:
1. Load/cache max projections
2. Segment nuclei with Cellpose CPSAM
3. Extract nucleus crops
4. Group by genotype
5. Compute global contrast
6. Generate multi-page PDF reports
"""

import os
import json
import glob
import numpy as np
import tifffile
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from scipy.interpolate import splprep, splev

# sys.path.insert(0, str(Path(__file__).parent.parent))
from .parse_czi import CziImage
from .cellpose_nuclear_seg import segment_nuclei
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
)
from .contrast_calculator import (
    compute_global_contrast,
    compute_per_file_contrast,
    apply_file_contrast_to_nuclei,
    apply_global_contrast,
    format_contrast_summary,
)

from .color_schemes import scheme_custom, ColorScheme

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
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    min_area_pixels: int = 5000
    max_area_pixels: int = 500000
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
    
    # Contrast - list of (low, high) percentile tuples per channel
    # If None, uses (0.5, 99.5) for all channels
    # If fewer tuples than channels, last tuple is reused
    contrast_percentiles: Optional[list[tuple[float, float]]] = None
    
    # Channels
    visible_channels: Optional[list[int]] = None  # None = all
    channel_colors: Optional[list[str]] = None  # None = auto from wavelengths
    
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


def load_or_compute_max_projection(
    czi_path: str,
    cache_dir: str,
    use_cache: bool = True,
) -> tuple[np.ndarray, 'CziImage']:
    """
    Load or compute max projection for a CZI file.
    
    Checks for cached file FIRST to avoid loading large CZI files unnecessarily.
    
    Returns:
        (max_proj_cyx, czi_image)
    """
    # Get the filename stem without loading the full CZI
    czi_name = Path(czi_path).stem
    cache_file = os.path.join(cache_dir, f"{czi_name}_max.tif")
    
    # Check cache BEFORE loading CZI to save memory
    if use_cache and os.path.exists(cache_file):
        print(f"  Loading cached max projection: {cache_file}")
        max_proj = tifffile.imread(cache_file)
        # Still need CZI for metadata, but don't load image data
        czi = CziImage(czi_path)
        return max_proj, czi
    
    # No cache - need to compute max projection
    print(f"  Loading CZI and computing max projection...")
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
    
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        tifffile.imwrite(cache_file, max_proj, compression='zstd', compressionargs={'level': 6})
        print(f"  Cached to: {cache_file}")
    
    return max_proj, czi


def should_rerun_segmentation(
    seg_dir: str,
    sample_name: str,
    config: ReportConfig,
) -> bool:
    """
    Check if segmentation should be re-run based on parameter changes.
    
    Compares the current nuclear_channel setting against saved parameters.
    """
    params_file = os.path.join(seg_dir, f"{sample_name}_params.json")
    mask_file = os.path.join(seg_dir, f"{sample_name}_seg.tif")
    
    # If mask doesn't exist, need to segment
    if not os.path.exists(mask_file):
        return True
    
    # If params file doesn't exist, re-run to be safe
    if not os.path.exists(params_file):
        return True
    
    # Load saved parameters and compare nuclear_channel
    try:
        with open(params_file, 'r') as f:
            saved_params = json.load(f)
        
        if saved_params.get('nuclear_channel') != config.nuclear_channel:
            print(f"  Nuclear channel changed ({saved_params.get('nuclear_channel')} -> {config.nuclear_channel}), re-segmenting")
            return True
        
        # Could add more parameter comparisons here if desired
        
    except (json.JSONDecodeError, IOError):
        return True
    
    return False


def save_segmentation_params(
    seg_dir: str,
    sample_name: str,
    config: ReportConfig,
):
    """Save segmentation parameters for later comparison."""
    params_file = os.path.join(seg_dir, f"{sample_name}_params.json")
    
    params = {
        'nuclear_channel': config.nuclear_channel,
        'cellpose_diameter': config.cellpose_diameter,
        'cellpose_flow_threshold': config.cellpose_flow_threshold,
        'cellpose_cellprob_threshold': config.cellpose_cellprob_threshold,
        'min_area_pixels': config.min_area_pixels,
        'max_area_pixels': config.max_area_pixels,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)


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
    nuclear_img = max_proj[config.nuclear_channel]
    
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
    
    # Save parameters for future comparison
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
    
    # Get crop bounds
    center_y, center_x = crop_meta.center_pixel
    box_h, box_w = crop_meta.box_size_pixels
    
    y_start = center_y - box_h // 2
    x_start = center_x - box_w // 2
    y_end = y_start + box_h
    x_end = x_start + box_w
    
    # Clip to image bounds
    y_start = max(0, y_start)
    x_start = max(0, x_start)
    y_end = min(mask.shape[0], y_end)
    x_end = min(mask.shape[1], x_end)
    
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


def draw_nucleus_outline(ax, contour: np.ndarray, color='white', linewidth=1.0, linestyle='dashed'):
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
    
    # Smooth the contour using B-spline with high smoothing
    smooth = smooth_contour_bspline(contour, num_points=30, smoothing=200.0)
    
    if smooth is None:
        smooth = contour  # Fallback to original
    
    # Close the contour
    contour_closed = np.vstack([smooth, smooth[0]])
    
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

def generate_multipage_pdf(
    nucleus_crops: list[NucleusCrop],
    image_data_dict: dict[str, np.ndarray],
    czi_info_dict: dict[str, CziImage],
    genotype: str,
    output_path: str,
    config: ReportConfig,
    color_scheme: Optional[ColorScheme] = None,
    mask_dict: Optional[dict[str, np.ndarray]] = None,
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
    
    # Layout configuration from config
    nucleus_cols = config.nucleus_cols
    rows_per_col = config.rows_per_page
    nuclei_per_page = nucleus_cols * rows_per_col
    
    # Get channel info from first nucleus
    first_nuc = nucleus_crops_sorted[0]
    active_channels = first_nuc.crop_meta.get_active_channels()
    images_per_nucleus = len(active_channels) + 1  # channels + merge
    
    # Get image dimensions
    first_image = image_data_dict[first_nuc.source_filename]
    crop_data = extract_crop_data(first_image, first_nuc.crop_meta)
    img_h, img_w = crop_data.shape[1], crop_data.shape[2]
    
    # Configure panels with tighter spacing
    panel_config = PanelConfig(
        show_channel_labels=False,
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
    
    # Calculate pagination
    n_nuclei = len(nucleus_crops_sorted)
    n_pages = (n_nuclei + nuclei_per_page - 1) // nuclei_per_page
    
    print(f"  Generating {n_pages} page(s) for {n_nuclei} nuclei from {len(sorted_files)} files...")
    
    # Track which file we're in for headers
    current_file_idx = 0
    nucleus_global_idx = 0
    
    # Create PDF
    with PdfPages(output_path) as pdf:
        for page_idx in range(n_pages):
            start_idx = page_idx * nuclei_per_page
            end_idx = min(start_idx + nuclei_per_page, n_nuclei)
            page_nuclei = nucleus_crops_sorted[start_idx:end_idx]
            
            # Figure size calculation
            img_width_in = img_w / config.dpi
            img_height_in = img_h / config.dpi
            
            # Add space for labels and gaps between nucleus groups
            label_width_in = 0.4  # Space for nucleus ID labels
            col_gap_in = 0.15  # Gap between nucleus groups
            
            fig_width = (img_width_in * images_per_nucleus + label_width_in) * nucleus_cols + col_gap_in * (nucleus_cols - 1)
            fig_height = img_height_in * rows_per_col + 0.5  # Extra for title
            
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=config.dpi)
            fig.patch.set_facecolor('white')
            
            scale_bar_config = panel_config.scale_bar or ScaleBarConfig()
            is_first_page = (page_idx == 0)
            
            # Render nuclei with file headers taking their own row
            slot_idx = 0  # Tracks position including headers
            prev_file = None
            
            for nuc_crop in page_nuclei:
                current_file = nuc_crop.source_filename
                is_new_file = (current_file != prev_file)
                
                # If new file, draw header row first
                if is_new_file:
                    col_group = slot_idx // rows_per_col
                    row_in_group = slot_idx % rows_per_col
                    
                    # Only draw if we have room on this page
                    if col_group < nucleus_cols:
                        # Calculate header position
                        group_width = (img_width_in * images_per_nucleus + label_width_in) / fig_width
                        group_start_x = col_group * (group_width + col_gap_in / fig_width)
                        row_height = img_height_in / fig_height
                        row_start_y = 1.0 - 0.04 - (row_in_group + 1) * row_height
                        
                        # Draw filename header
                        filename_stem = Path(current_file).stem
                        fig.text(
                            group_start_x + label_width_in / fig_width,
                            row_start_y + row_height * 0.5,
                            filename_stem,
                            fontsize=6,
                            ha='left',
                            va='center',
                            fontweight='bold',
                            family='sans-serif',
                            color='black',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9),
                        )
                    
                    slot_idx += 1
                    prev_file = current_file
                
                # Now draw the nucleus
                col_group = slot_idx // rows_per_col
                row_in_group = slot_idx % rows_per_col
                
                # Skip if we've run out of space on this page
                if col_group >= nucleus_cols:
                    continue
                
                # Get image data and render
                image_data = image_data_dict[nuc_crop.source_filename]
                crop_data = extract_crop_data(image_data, nuc_crop.crop_meta)
                
                # Get nucleus contour for outline
                contour = None
                if nuc_crop.source_filename in mask_dict:
                    contour = extract_nucleus_contour(
                        mask_dict[nuc_crop.source_filename],
                        nuc_crop.nucleus_id,
                        nuc_crop.crop_meta,
                    )
                
                # Create FigurePanel
                panel = FigurePanel(
                    crop_data=crop_data,
                    crop_meta=nuc_crop.crop_meta,
                    color_scheme=color_scheme,
                    config=panel_config,
                )
                
                # Render images
                individual = panel.render_individual_channels()
                merged = panel.render_merged()
                all_images = individual + [merged]
                
                pixel_size_um = nuc_crop.crop_meta.pixel_size_um[1]
                
                # Calculate axes positions
                group_width = (img_width_in * images_per_nucleus + label_width_in) / fig_width
                group_start_x = col_group * (group_width + col_gap_in / fig_width)
                
                row_height = img_height_in / fig_height
                row_start_y = 1.0 - 0.04 - (row_in_group + 1) * row_height
                
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
                        draw_nucleus_outline(ax, contour, color='white', linewidth=0.8)
                    
                    # Add scale bar (only on first nucleus of first page, first image)
                    if is_first_page and slot_idx == 1 and img_idx == 0:
                        _add_scale_bar(ax, pixel_size_um, (img_h, img_w), scale_bar_config, show_label=True)
                    else:
                        _add_scale_bar(ax, pixel_size_um, (img_h, img_w), scale_bar_config, show_label=False)
                
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
            if image_data is not None:
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


def save_run_parameters(config: ReportConfig, output_dir: str):
    """Save all run parameters to JSON file."""
    params_file = os.path.join(output_dir, "run_parameters.json")
    
    params = config.to_dict()
    params['run_timestamp'] = datetime.now().isoformat()
    
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"Saved run parameters: {params_file}")


def generate_qc_report(config: ReportConfig):
    """
    Main entry point: generate automated QC reports.
    
    Args:
        config: ReportConfig with all settings
    """
    print("=" * 60)
    print("AUTOMATED QC REPORT GENERATOR")
    print("=" * 60)
    print(f"Input directory: {config.input_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Genotypes: {config.genotypes}")
    print(f"Nuclear channel: {config.nuclear_channel}")
    print(f"Box size: {config.box_size_um} µm")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save run parameters
    save_run_parameters(config, config.output_dir)
    
    # Find all CZI files
    czi_files = glob.glob(os.path.join(config.input_dir, "*.czi"))
    czi_files.extend(glob.glob(os.path.join(config.input_dir, "*.CZI")))
    
    if len(czi_files) == 0:
        print(f"No CZI files found in {config.input_dir}")
        return
    
    print(f"\nFound {len(czi_files)} CZI files")
    
    # Process all files
    all_nucleus_crops: dict[str, list[NucleusCrop]] = {g: [] for g in config.genotypes}
    image_data_dict: dict[str, np.ndarray] = {}
    czi_info_dict: dict[str, CziImage] = {}
    existing_ids: set[str] = set()
    
    for i, czi_path in enumerate(czi_files):
        print(f"\n[{i+1}/{len(czi_files)}] Processing: {os.path.basename(czi_path)}")
        
        try:
            nucleus_crops, max_proj, czi = process_single_file(
                czi_path, config, existing_ids
            )
            
            if len(nucleus_crops) > 0:
                genotype = nucleus_crops[0].genotype
                all_nucleus_crops[genotype].extend(nucleus_crops)
                image_data_dict[czi.name] = max_proj
                czi_info_dict[czi.name] = czi
                
        except Exception as e:
            print(f"  Error processing {czi_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Report summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    for genotype in config.genotypes:
        n_nuclei = len(all_nucleus_crops[genotype])
        print(f"  {genotype}: {n_nuclei} nuclei")
    
    # Compute global contrast per genotype and generate PDFs
    print("\n" + "=" * 60)
    print("GENERATING REPORTS")
    print("=" * 60)
    
    for genotype in config.genotypes:
        nucleus_crops = all_nucleus_crops[genotype]
        
        if len(nucleus_crops) == 0:
            print(f"\n[{genotype}] No nuclei, skipping")
            continue
        
        print(f"\n[{genotype}] Processing {len(nucleus_crops)} nuclei...")
        
        # Compute per-file contrast (instead of global)
        # Group nuclei by source file
        nuclei_by_file: dict[str, list[NucleusCrop]] = {}
        for nuc in nucleus_crops:
            if nuc.source_filename not in nuclei_by_file:
                nuclei_by_file[nuc.source_filename] = []
            nuclei_by_file[nuc.source_filename].append(nuc)
        
        # Compute and apply contrast per file
        file_contrast_settings: dict[str, dict[int, tuple[float, float]]] = {}
        
        for source_file, file_nuclei in nuclei_by_file.items():
            image_data = image_data_dict[source_file]
            
            # Compute contrast from full image with per-channel percentiles
            contrast_limits = compute_per_file_contrast(
                image_data,
                percentiles=config.contrast_percentiles,
            )
            
            file_contrast_settings[source_file] = contrast_limits
            
            # Apply to all nuclei from this file
            apply_file_contrast_to_nuclei(file_nuclei, contrast_limits)
            
            # Print contrast info with percentile settings
            print(f"  {source_file}: {len(file_nuclei)} nuclei")
            for ch, (vmin, vmax) in contrast_limits.items():
                if config.contrast_percentiles and ch < len(config.contrast_percentiles):
                    pct = config.contrast_percentiles[ch]
                elif config.contrast_percentiles:
                    pct = config.contrast_percentiles[-1]
                else:
                    pct = (0.5, 99.5)
                print(f"    Ch{ch} ({pct[0]}-{pct[1]}%): [{vmin:.1f}, {vmax:.1f}]")
        
        # Determine color scheme
        color_scheme = None
        if config.channel_colors:
            try:
                # Pass string colors directly to scheme_custom() which has
                # microscopy-friendly color mapping (e.g., "green" = full brightness)
                # Do NOT pre-convert with mcolors.to_rgb() which gives dim colors!
                color_scheme = scheme_custom(config.channel_colors, name="Custom_CLI")
                print(f"  Using custom colors: {config.channel_colors}")
                # Debug: show actual RGB values being used
                for i, lut in enumerate(color_scheme.channel_luts):
                    print(f"    Ch{i}: {config.channel_colors[i]} -> RGB {lut.color_rgb}")
            except ValueError as e:
                print(f"  Warning: Invalid color in {config.channel_colors}: {e}")
                print(f"  Falling back to auto colors")
                color_scheme = None
        
        # Generate PDF
        output_path = os.path.join(config.output_dir, f"{genotype}_report.pdf")
        generate_multipage_pdf(
            nucleus_crops,
            image_data_dict,
            czi_info_dict,
            genotype,
            output_path,
            config,
            color_scheme=color_scheme,
        )
    
    # Save metadata CSV
    csv_path = os.path.join(config.output_dir, "nucleus_metadata.csv")
    save_metadata_csv(all_nucleus_crops, image_data_dict, csv_path)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Reports saved to: {config.output_dir}")