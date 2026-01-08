"""
Improved nucleus contour drawing with smooth B-spline curves.

Replace your extract_nucleus_contour and draw_nucleus_outline functions with these.
"""

import numpy as np
from typing import Optional


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


def smooth_contour_bspline(contour: np.ndarray, num_points: int = 150, smoothing: float = 5.0) -> Optional[np.ndarray]:
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
    
    # Subsample if contour has too many points (speeds up spline fitting)
    if len(contour) > 500:
        indices = np.linspace(0, len(contour) - 1, 500).astype(int)
        contour = contour[indices]
    
    y = contour[:, 0]
    x = contour[:, 1]
    
    try:
        # Fit periodic B-spline (per=1 means closed curve)
        # s = smoothing factor, k = spline degree (cubic)
        tck, u = splprep([x, y], s=smoothing, per=True, k=3)
        
        # Evaluate spline at evenly spaced points
        u_new = np.linspace(0, 1, num_points)
        x_smooth, y_smooth = splev(u_new, tck)
        
        return np.column_stack([y_smooth, x_smooth])
    
    except Exception as e:
        # Fallback: return original contour if spline fails
        print(f"    Spline smoothing failed: {e}")
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
    
    # Smooth the contour using B-spline
    smooth = smooth_contour_bspline(contour, num_points=150, smoothing=3.0)
    
    if smooth is None:
        smooth = contour  # Fallback to original
    
    # Close the contour
    contour_closed = np.vstack([smooth, smooth[0]])
    
    # Define dash pattern: (dash_length, gap_length) in points
    # A nice balanced dash for microscopy images
    if linestyle == 'dashed':
        dash_pattern = (0, (4, 3))  # 4pt dash, 3pt gap
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


# --- TEST CODE ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a test contour (rough circle with noise)
    theta = np.linspace(0, 2*np.pi, 100)
    noise = np.random.randn(100) * 5
    y = 100 + 50 * np.sin(theta) + noise
    x = 100 + 50 * np.cos(theta) + noise
    test_contour = np.column_stack([y, x])
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Raw contour
    axes[0].set_title("Raw Contour")
    axes[0].plot(test_contour[:, 1], test_contour[:, 0], 'w-', linewidth=1)
    axes[0].set_facecolor('black')
    axes[0].set_aspect('equal')
    
    # Smoothed contour
    axes[1].set_title("B-Spline Smoothed")
    axes[1].set_facecolor('black')
    draw_nucleus_outline(axes[1], test_contour)
    axes[1].set_aspect('equal')
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(axes[0].get_ylim())
    
    plt.tight_layout()
    plt.savefig("contour_test.png", dpi=150)
    print("Saved contour_test.png")