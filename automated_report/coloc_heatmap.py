"""
Colocalization heatmap computation for multi-channel microscopy images.

Computes pixel-wise colocalization scores based on z-score transformed
intensities, providing a measure of spatial correlation between channels.
"""

import numpy as np
from typing import Tuple, Optional
from numba import njit, prange

VALID_COLOC_COLORMAPS = ['oslo', 'white_to_black']


def get_coloc_colormap(colormap: str = 'oslo'):
    """Return a matplotlib colormap for colocalization rendering."""
    if colormap not in VALID_COLOC_COLORMAPS:
        raise ValueError(
            f"Unknown colocalization colormap '{colormap}'. Valid options: {VALID_COLOC_COLORMAPS}"
        )

    if colormap == 'white_to_black':
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list('white_to_black', ['white', 'black'])

    from palettable.scientific.sequential import Oslo_20
    return Oslo_20.mpl_colormap


def z_score_transform(image: np.ndarray) -> np.ndarray:
    """
    Transforms a float image as np array into z-score space.

    Parameters:
    image : np.ndarray
        Input float image.
    
    Returns:
    np.ndarray
        Z-score transformed image.
    """
    mean = np.mean(image)
    sigma = np.std(image)
    if sigma == 0:
        # Return zeros if no variance (avoid division by zero)
        return np.zeros_like(image, dtype=np.float32)
    z_image = (image - mean) / sigma
    return z_image


def coloc_heatmap(image: np.ndarray, channels: Tuple[int, int]) -> np.ndarray:
    """
    Returns a colocalization heatmap from a multi-channel image.
    
    Only shows colocalization where BOTH channels have above-mean intensity
    (i.e., actual fluorescent signal). Background regions are set to 0.

    Parameters:
    image : np.ndarray
        Input 4D multi-channel image (CZYX).
    channels : tuple of int (len 2)
        Tuple of channel indices to consider for colocalization.
    
    Returns:
    coloc_map : np.ndarray
        Colocalization heatmap as a 3D np array (ZYX).
    """
    channel_1 = image[channels[0], :, :, :].astype(np.float32)
    channel_2 = image[channels[1], :, :, :].astype(np.float32)
    z_channel_1 = z_score_transform(channel_1)
    z_channel_2 = z_score_transform(channel_2)
    
    # Compute colocalization
    coloc_map = np.multiply(z_channel_1, z_channel_2) / (abs(z_channel_1) + abs(z_channel_2) + 1)
    
    # Mask out regions where either channel is below mean (z < 0)
    # This ensures we only show colocalization of actual signal, not background
    background_mask = (z_channel_1 < 0) | (z_channel_2 < 0)
    coloc_map[background_mask] = 0
    
    return coloc_map


def compute_coloc_max_projection(
    image_czyx: np.ndarray,
    channels: Tuple[int, int],
) -> np.ndarray:
    """
    Compute colocalization heatmap and return max projection.
    
    This function is designed to be called BEFORE the regular max projection,
    as it requires the full 3D/4D image data.
    
    Parameters:
    -----------
    image_czyx : np.ndarray
        Input 4D multi-channel image in CZYX format.
        If 3D (CYX), a Z dimension will be added.
    channels : tuple of int
        Tuple of two channel indices for colocalization analysis.
    
    Returns:
    --------
    coloc_2d : np.ndarray
        2D max-projected colocalization heatmap (YX).
        Raw colocalization values (not normalized).
        Values represent colocalization strength:
        - Positive: co-localized (both channels high or both low)
        - Negative: anti-correlated (one high, one low)
        - Zero: no correlation
    """
    # Handle 3D input (CYX) by adding Z dimension
    if image_czyx.ndim == 3:
        raise ValueError("Input image must be 4D (CZYX), not 3D (CYX)")
    
    # Validate channels
    n_channels = image_czyx.shape[0]
    if channels[0] >= n_channels or channels[1] >= n_channels:
        raise ValueError(
            f"Invalid channel indices {channels} for image with {n_channels} channels"
        )
    
    if channels[0] == channels[1]:
        raise ValueError("Colocalization requires two different channels")
    
    # Compute 3D colocalization map
    coloc_3d = coloc_heatmap_optimized(image_czyx, channels) #DEBUG: trying optimized version of coloc_heatmap
    
    # Max projection along Z axis
    coloc_2d = np.max(coloc_3d, axis=0)
    
    return coloc_2d


def render_coloc_heatmap(
    coloc_data: np.ndarray,
    vmax: Optional[float] = None,
    colormap: str = 'oslo',
    return_scale: bool = False,
) -> np.ndarray:
    """
    Render colocalization heatmap as an RGB image.
    
    Zero and negative values use the low-end color: black for Oslo, white
    for white_to_black. Values at or above vmax use the high-end color.
    
    Parameters:
    -----------
    coloc_data : np.ndarray
        2D colocalization data (YX).
    vmax : float, optional
        Maximum value for colormap scaling. If None, uses data max.
    colormap : str
        Name of the colocalization colormap to use.
    return_scale : bool
        If True, also return scale metadata for legend rendering.
    
    Returns:
    --------
    rgb_image : np.ndarray
        RGB image (H, W, 3) as uint8.
    """
    cmap = get_coloc_colormap(colormap)
    
    # Clip negative values to 0
    coloc_clipped = np.clip(coloc_data, 0, None)
    
    # Determine vmax
    data_max = float(np.max(coloc_clipped)) if coloc_clipped.size else 0.0
    if vmax is None:
        vmax = data_max
    
    # Avoid division by zero
    scale_vmax = float(vmax)
    normalizer = scale_vmax if scale_vmax > 0 else 1.0
    
    # clip top end of the range to vmax
    normalized = np.clip(coloc_clipped / normalizer, 0, 1)
    
    # Apply colormap
    rgba = cmap(normalized)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    
    # Preserve Oslo's exact black background; white_to_black uses its white endpoint.
    if colormap == 'oslo':
        rgb[coloc_data <= 0] = 0
    
    if return_scale:
        return rgb, {
            'vmin': 0.0,
            'vmax': scale_vmax,
            'data_min': float(np.min(coloc_data)) if coloc_data.size else 0.0,
            'data_max': data_max,
            'colormap': colormap,
        }

    return rgb


def render_coloc_scale_bar(
    height: int = 128,
    width: int = 12,
    colormap: str = 'oslo',
) -> np.ndarray:
    """Render a vertical RGB scale bar image for a colocalization colormap."""
    cmap = get_coloc_colormap(colormap)
    gradient = np.linspace(0, 1, height, dtype=np.float32)[:, np.newaxis]
    gradient = np.repeat(gradient, width, axis=1)
    rgba = cmap(gradient[::-1])
    return (rgba[:, :, :3] * 255).astype(np.uint8)


@njit(parallel=True, fastmath=True)
def _compute_stats_and_heatmap(
    image: np.ndarray, 
    c1_idx: int, 
    c2_idx: int
) -> np.ndarray:
    """
    Fused kernel: Computes statistics and generates heatmap in a single pass 
    to minimize memory overhead.
    """
    # 1. Compute Mean and Std for both channels manually to avoid creating 
    # huge intermediate arrays.
    # We flatten the view conceptually to loop easier
    d, h, w = image.shape[1], image.shape[2], image.shape[3]
    size = d * h * w
    
    sum1, sum2 = 0.0, 0.0
    sq_sum1, sq_sum2 = 0.0, 0.0
    
    # First Pass: Statistics
    # We iterate linearly to be cache-friendly
    for z in prange(d):
        for y in range(h):
            for x in range(w):
                v1 = image[c1_idx, z, y, x]
                v2 = image[c2_idx, z, y, x]
                
                sum1 += v1
                sum2 += v2
                sq_sum1 += v1 * v1
                sq_sum2 += v2 * v2
                
    mean1 = sum1 / size
    mean2 = sum2 / size
    
    # Standard Deviation formula: sqrt(E[x^2] - (E[x])^2)
    std1 = np.sqrt(sq_sum1 / size - mean1 * mean1)
    std2 = np.sqrt(sq_sum2 / size - mean2 * mean2)
    
    # Avoid division by zero
    if std1 == 0: std1 = 1.0
    if std2 == 0: std2 = 1.0

    # Initialize output array
    coloc_map = np.zeros((d, h, w), dtype=np.float32)

    # Second Pass: Compute Z-scores and Heatmap
    # This fuses 5+ operations into one loop, keeping data in CPU cache.
    for z in prange(d):
        for y in range(h):
            for x in range(w):
                v1 = image[c1_idx, z, y, x]
                v2 = image[c2_idx, z, y, x]
                
                # Z-score transform on the fly
                z1 = (v1 - mean1) / std1
                z2 = (v2 - mean2) / std2
                
                # Check logic: Only show if BOTH are above mean (z > 0)
                # Note: Your original code used >= 0, keeping consistent
                if z1 >= 0 and z2 >= 0:
                    numerator = z1 * z2
                    denominator = abs(z1) + abs(z2) + 1
                    coloc_map[z, y, x] = numerator / denominator
                else:
                    coloc_map[z, y, x] = 0.0
                    
    return coloc_map

def coloc_heatmap_optimized(image: np.ndarray, channels: Tuple[int, int]) -> np.ndarray:
    """
    Wrapper to ensure inputs are correct before passing to Numba.
    """
    return _compute_stats_and_heatmap(image, channels[0], channels[1])
