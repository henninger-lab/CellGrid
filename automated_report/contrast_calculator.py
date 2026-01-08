"""
Global contrast calculation for automated reports.

Computes consistent contrast limits across all nuclei within a genotype,
ensuring comparable visualization across the entire dataset.
"""

import numpy as np
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from .utilities import ChannelContrast

from .nucleus_extractor import NucleusCrop, extract_crop_data


def compute_per_file_contrast(
    image_data: np.ndarray,
    percentiles: Optional[list[tuple[float, float]]] = None,
) -> dict[int, tuple[float, float]]:
    """
    Compute contrast limits from a full max-projected image.
    
    This computes contrast from the entire field of view, not just nucleus crops,
    providing more consistent results when there are acquisition differences
    between files.
    
    Args:
        image_data: Max-projected image in CYX format
        percentiles: List of (low, high) percentile tuples per channel.
                    If None, uses (0.5, 99.5) for all.
                    If fewer tuples than channels, last tuple is reused.
        
    Returns:
        Dict mapping channel_index -> (vmin, vmax)
    """
    n_channels = image_data.shape[0]
    contrast_limits = {}
    
    # Default percentiles if not provided
    if percentiles is None:
        percentiles = [(0.5, 99.5)]
    
    for ch in range(n_channels):
        channel_data = image_data[ch].flatten()
        
        # Get percentile pair for this channel (reuse last if not enough)
        if ch < len(percentiles):
            pct_low, pct_high = percentiles[ch]
        else:
            pct_low, pct_high = percentiles[-1]
        
        vmin = float(np.percentile(channel_data, pct_low))
        vmax = float(np.percentile(channel_data, pct_high))
        
        # Ensure vmax > vmin
        if vmax <= vmin:
            vmax = vmin + 1.0
        
        contrast_limits[ch] = (vmin, vmax)
    
    return contrast_limits


def apply_file_contrast_to_nuclei(
    nucleus_crops: list[NucleusCrop],
    contrast_limits: dict[int, tuple[float, float]],
) -> list[NucleusCrop]:
    """
    Apply contrast limits to a list of NucleusCrop objects.
    
    Args:
        nucleus_crops: List of NucleusCrop objects (all from same source file)
        contrast_limits: Dict mapping channel_index -> (vmin, vmax)
        
    Returns:
        Same list with updated contrasts
    """
    for nuc_crop in nucleus_crops:
        n_channels = len(nuc_crop.crop_meta.channel_contrasts)
        
        new_contrasts = []
        for ch in range(n_channels):
            if ch in contrast_limits:
                vmin, vmax = contrast_limits[ch]
            else:
                vmin = nuc_crop.crop_meta.channel_contrasts[ch].min_val
                vmax = nuc_crop.crop_meta.channel_contrasts[ch].max_val
            
            new_contrasts.append(ChannelContrast(min_val=vmin, max_val=vmax))
        
        nuc_crop.crop_meta.channel_contrasts = new_contrasts
    
    return nucleus_crops


def compute_global_contrast(
    nucleus_crops: list[NucleusCrop],
    image_data_dict: dict[str, np.ndarray],
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
    n_channels: Optional[int] = None,
) -> dict[int, tuple[float, float]]:
    """
    Compute global contrast limits across all nuclei for each channel.
    
    Samples pixel values from all crops and computes percentile-based
    contrast limits that work well for the entire dataset.
    
    Args:
        nucleus_crops: List of NucleusCrop objects
        image_data_dict: Dict mapping source_filename -> max-projected CYX array
        percentile_low: Lower percentile for min value (default 0.5)
        percentile_high: Upper percentile for max value (default 99.5)
        n_channels: Number of channels (auto-detected if None)
        
    Returns:
        Dict mapping channel_index -> (vmin, vmax)
    """
    if len(nucleus_crops) == 0:
        raise ValueError("No nucleus crops provided")
    
    # Auto-detect number of channels from first available image
    if n_channels is None:
        first_filename = nucleus_crops[0].source_filename
        n_channels = image_data_dict[first_filename].shape[0]
    
    # Collect pixel values for each channel
    channel_values = {ch: [] for ch in range(n_channels)}
    
    for nuc_crop in nucleus_crops:
        image_data = image_data_dict.get(nuc_crop.source_filename)
        if image_data is None:
            print(f"  Warning: No image data for {nuc_crop.source_filename}, skipping")
            continue
        
        # Extract crop data
        crop_data = extract_crop_data(image_data, nuc_crop.crop_meta)
        
        # Collect values for each channel
        for ch in range(min(n_channels, crop_data.shape[0])):
            # Sample pixel values (flatten the crop)
            channel_values[ch].append(crop_data[ch].flatten())
    
    # Compute percentiles for each channel
    contrast_limits = {}
    
    for ch in range(n_channels):
        if len(channel_values[ch]) == 0:
            # No data for this channel, use defaults
            contrast_limits[ch] = (0.0, 65535.0)
            continue
        
        # Concatenate all values for this channel
        all_values = np.concatenate(channel_values[ch])
        
        vmin = float(np.percentile(all_values, percentile_low))
        vmax = float(np.percentile(all_values, percentile_high))
        
        # Ensure vmax > vmin
        if vmax <= vmin:
            vmax = vmin + 1.0
        
        contrast_limits[ch] = (vmin, vmax)
    
    return contrast_limits


def apply_global_contrast(
    nucleus_crops: list[NucleusCrop],
    contrast_limits: dict[int, tuple[float, float]],
) -> list[NucleusCrop]:
    """
    Apply global contrast limits to all NucleusCrop objects.
    
    Modifies the channel_contrasts in each crop_meta to use the
    global limits, ensuring consistent visualization.
    
    Args:
        nucleus_crops: List of NucleusCrop objects
        contrast_limits: Dict mapping channel_index -> (vmin, vmax)
        
    Returns:
        Same list of NucleusCrop objects with updated contrasts
    """
    for nuc_crop in nucleus_crops:
        n_channels = len(nuc_crop.crop_meta.channel_contrasts)
        
        new_contrasts = []
        for ch in range(n_channels):
            if ch in contrast_limits:
                vmin, vmax = contrast_limits[ch]
            else:
                # Fallback to existing or default
                vmin = nuc_crop.crop_meta.channel_contrasts[ch].min_val
                vmax = nuc_crop.crop_meta.channel_contrasts[ch].max_val
            
            new_contrasts.append(ChannelContrast(min_val=vmin, max_val=vmax))
        
        nuc_crop.crop_meta.channel_contrasts = new_contrasts
    
    return nucleus_crops


def compute_per_nucleus_contrast(
    image_data: np.ndarray,
    crop_meta,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
) -> list[ChannelContrast]:
    """
    Compute contrast limits for a single nucleus crop.
    
    Alternative to global contrast when per-nucleus optimization is desired.
    
    Args:
        image_data: Max-projected image in CYX format
        crop_meta: CropMetadata for this nucleus
        percentile_low: Lower percentile
        percentile_high: Upper percentile
        
    Returns:
        List of ChannelContrast objects, one per channel
    """
    crop_data = extract_crop_data(image_data, crop_meta)
    
    contrasts = []
    for ch in range(crop_data.shape[0]):
        channel_data = crop_data[ch]
        
        vmin = float(np.percentile(channel_data, percentile_low))
        vmax = float(np.percentile(channel_data, percentile_high))
        
        if vmax <= vmin:
            vmax = vmin + 1.0
        
        contrasts.append(ChannelContrast(min_val=vmin, max_val=vmax))
    
    return contrasts


def format_contrast_summary(
    contrast_limits: dict[int, tuple[float, float]],
    channel_wavelengths: Optional[list[dict]] = None,
) -> str:
    """
    Format contrast limits as a human-readable summary string.
    
    Args:
        contrast_limits: Dict mapping channel_index -> (vmin, vmax)
        channel_wavelengths: Optional wavelength info for channel names
        
    Returns:
        Formatted string summary
    """
    lines = ["Global Contrast Settings:"]
    
    for ch, (vmin, vmax) in sorted(contrast_limits.items()):
        if channel_wavelengths and ch < len(channel_wavelengths):
            wl = channel_wavelengths[ch]
            emission = wl.get('emission')
            if emission:
                name = f"{int(emission)}nm"
            else:
                name = f"Ch{ch}"
        else:
            name = f"Ch{ch}"
        
        lines.append(f"  {name}: [{vmin:.1f}, {vmax:.1f}]")
    
    return "\n".join(lines)