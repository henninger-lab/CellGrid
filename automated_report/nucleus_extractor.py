"""
Nucleus extraction from Cellpose segmentation masks.

Converts segmentation masks into CropMetadata objects compatible
with the existing figure generation workflow.
"""

import numpy as np
from skimage import measure
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from .utilities import CropMetadata, ChannelContrast


@dataclass
class NucleusCrop:
    """Extended crop metadata for automated nucleus extraction."""
    crop_meta: CropMetadata
    nucleus_id: int
    genotype: str
    source_filename: str
    centroid_um: tuple[float, float]
    area_um2: float
    area_pixels: int
    unique_id: str
    mean_intensities: list[float] = field(default_factory=list)
    
    def __post_init__(self):
        self.crop_meta.crop_id = self.unique_id


def is_border_touching(mask: np.ndarray, nucleus_id: int) -> bool:
    """
    Check if a nucleus touches the image boundary.
    
    Args:
        mask: Segmentation mask array (H, W)
        nucleus_id: Label ID to check
        
    Returns:
        True if nucleus touches any edge
    """
    nucleus_coords = np.where(mask == nucleus_id)
    if len(nucleus_coords[0]) == 0:
        return True  # Empty nucleus, consider invalid
    
    y_coords, x_coords = nucleus_coords
    h, w = mask.shape
    
    touches_edge = (
        np.any(y_coords == 0) or 
        np.any(y_coords == h - 1) or
        np.any(x_coords == 0) or 
        np.any(x_coords == w - 1)
    )
    return touches_edge


def compute_nucleus_centroid(mask: np.ndarray, nucleus_id: int) -> tuple[int, int]:
    """
    Compute the centroid of a nucleus in pixel coordinates.
    
    Args:
        mask: Segmentation mask
        nucleus_id: Label ID
        
    Returns:
        (y, x) centroid in pixels
    """
    nucleus_coords = np.where(mask == nucleus_id)
    centroid_y = int(np.mean(nucleus_coords[0]))
    centroid_x = int(np.mean(nucleus_coords[1]))
    return (centroid_y, centroid_x)


def compute_nucleus_area(mask: np.ndarray, nucleus_id: int) -> int:
    """
    Compute the area of a nucleus in pixels.
    
    Args:
        mask: Segmentation mask
        nucleus_id: Label ID
        
    Returns:
        Area in pixels
    """
    return int(np.sum(mask == nucleus_id))


def extract_nuclei_from_mask(
    mask: np.ndarray,
    image_data: np.ndarray,
    source_filename: str,
    genotype: str,
    box_size_um: float,
    pixel_size_um: tuple[float, float],
    channel_wavelengths: list[dict],
    channel_colors: list[str],
    visible_channels: Optional[list[int]] = None,
    exclude_border: bool = True,
    existing_ids: Optional[set[str]] = None,
) -> list[NucleusCrop]:
    """
    Extract NucleusCrop objects from a segmentation mask.
    
    OPTIMIZED: Uses regionprops for O(pixels) instead of O(n_nuclei × pixels).
    """
    if existing_ids is None:
        existing_ids = set()
    
    pixel_size_y, pixel_size_x = pixel_size_um
    h, w = mask.shape
    
    # Calculate box size in pixels
    box_height_px = int(np.ceil(box_size_um / pixel_size_y))
    box_width_px = int(np.ceil(box_size_um / pixel_size_x))
    half_h = box_height_px // 2
    half_w = box_width_px // 2
    
    # Determine visible channels
    n_channels = image_data.shape[0]
    if visible_channels is None:
        visible_channels = list(range(n_channels))
    
    # Placeholder contrast
    placeholder_contrasts = [
        ChannelContrast(min_val=0.0, max_val=65535.0)
        for _ in range(n_channels)
    ]
    
    # >>> OPTIMIZATION: Compute ALL region properties in ONE pass
    # This replaces the per-nucleus np.where() calls
    regions = measure.regionprops(mask)
    # <
    
    nucleus_crops = []
    
    for props in regions:
        nuc_id = props.label
        
        # Exclude nuclei that either touch the segmentation border or would
        # require a crop extending outside the image bounds.
        if exclude_border:
            min_row, min_col, max_row, max_col = props.bbox
            if min_row == 0 or min_col == 0 or max_row == h or max_col == w:
                continue

            centroid_y_px = int(props.centroid[0])
            centroid_x_px = int(props.centroid[1])
            crop_y_start = centroid_y_px - half_h
            crop_y_end = crop_y_start + box_height_px
            crop_x_start = centroid_x_px - half_w
            crop_x_end = crop_x_start + box_width_px
            if (
                crop_y_start < 0
                or crop_x_start < 0
                or crop_y_end > h
                or crop_x_end > w
            ):
                continue
        # <
        
        # >>> OPTIMIZATION: Centroid and area from pre-computed properties
        centroid_y_px = int(props.centroid[0])
        centroid_x_px = int(props.centroid[1])
        area_pixels = props.area
        # <
        
        # Convert to physical coordinates
        centroid_y_um = centroid_y_px * pixel_size_y
        centroid_x_um = centroid_x_px * pixel_size_x
        area_um2 = area_pixels * pixel_size_y * pixel_size_x
        
        # Generate unique ID
        base_id = f"{genotype}_{source_filename}_{nuc_id:04d}"
        unique_id = base_id
        counter = 1
        while unique_id in existing_ids:
            unique_id = f"{base_id}_{counter}"
            counter += 1
        existing_ids.add(unique_id)
        
        # Create CropMetadata
        crop_meta = CropMetadata(
            source_file=source_filename,
            source_name=source_filename,
            center_pixel=(centroid_y_px, centroid_x_px),
            box_size_pixels=(box_height_px, box_width_px),
            center_physical_um=(centroid_y_um, centroid_x_um),
            box_size_physical_um=box_size_um,
            pixel_size_um=(pixel_size_y, pixel_size_x),
            z_range=(0, 1),
            z_projection="max",
            channel_contrasts=placeholder_contrasts,
            channel_wavelengths=channel_wavelengths,
            channel_colors_original=channel_colors,
            visible_channels=visible_channels,
            timestamp=datetime.now().isoformat(),
            notes=f"Auto-extracted nucleus {nuc_id}",
            crop_id=unique_id,
        )

        crop_data = extract_crop_data(image_data, crop_meta)
        mean_intensities = [float(np.mean(crop_data[ch])) for ch in range(crop_data.shape[0])]
        
        nucleus_crop = NucleusCrop(
            crop_meta=crop_meta,
            nucleus_id=nuc_id,
            genotype=genotype,
            source_filename=source_filename,
            centroid_um=(centroid_y_um, centroid_x_um),
            area_um2=area_um2,
            area_pixels=area_pixels,
            unique_id=unique_id,
            mean_intensities=mean_intensities,
        )
        
        nucleus_crops.append(nucleus_crop)
    
    return nucleus_crops


def get_crop_bounds(
    image_shape: tuple[int, ...],
    crop_meta: CropMetadata,
) -> tuple[int, int, int, int]:
    """Return clipped YX crop bounds for a CropMetadata region."""
    center_y, center_x = crop_meta.center_pixel
    box_h, box_w = crop_meta.box_size_pixels

    y_start = center_y - box_h // 2
    y_end = y_start + box_h
    x_start = center_x - box_w // 2
    x_end = x_start + box_w

    y_start = max(0, y_start)
    y_end = min(image_shape[-2], y_end)
    x_start = max(0, x_start)
    x_end = min(image_shape[-1], x_end)

    return y_start, y_end, x_start, x_end


def extract_crop_data(
    image_data: np.ndarray,
    crop_meta: CropMetadata,
) -> np.ndarray:
    """
    Extract crop data from max-projected image using CropMetadata.
    
    Args:
        image_data: Max-projected image in CYX format
        crop_meta: Crop metadata with center and size info
        
    Returns:
        Cropped array in CYX format
    """
    center_y, center_x = crop_meta.center_pixel
    box_h, box_w = crop_meta.box_size_pixels
    
    # Calculate bounds
    y_start = center_y - box_h // 2
    y_end = y_start + box_h
    x_start = center_x - box_w // 2
    x_end = x_start + box_w
    
    # Build fixed-size crop and pad with zeros if the box extends beyond image bounds.
    # This preserves near-edge nuclei instead of dropping them during extraction.
    n_channels = image_data.shape[0]
    crop = np.zeros((n_channels, box_h, box_w), dtype=image_data.dtype)

    src_y_start = max(0, y_start)
    src_y_end = min(image_data.shape[1], y_end)
    src_x_start = max(0, x_start)
    src_x_end = min(image_data.shape[2], x_end)

    dst_y_start = src_y_start - y_start
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = src_x_start - x_start
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    if src_y_end > src_y_start and src_x_end > src_x_start:
        crop[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image_data[
            :, src_y_start:src_y_end, src_x_start:src_x_end
        ]

    return crop
