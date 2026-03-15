import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['figure.dpi'] = 300

import os
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, plot, io
from scipy import ndimage
import tifffile

VALID_STRATEGIES = ['baseline', 'torch_compile', 'torch_compile_max']
_cellpose_models = {}


def get_cellpose_model(strategy='baseline'):
    """Get or create a cached Cellpose model for the requested strategy."""
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid strategies: {VALID_STRATEGIES}")

    if strategy not in _cellpose_models:
        print("  Loading Cellpose model (one-time)...")
        model = models.CellposeModel(pretrained_model='cpsam', gpu=True)
        io.logger_setup()
        _apply_strategy(model, strategy)
        _cellpose_models[strategy] = model

    return _cellpose_models[strategy]


def _apply_strategy(model, strategy):
    """Apply an optional optimization strategy to the Cellpose network."""
    if strategy == 'baseline':
        return

    import torch

    if strategy == 'torch_compile':
        print("  Compiling Cellpose model with torch.compile(mode='reduce-overhead')...")
        model.net = torch.compile(model.net, mode='reduce-overhead')
        return

    if strategy == 'torch_compile_max':
        print("  Compiling Cellpose model with torch.compile(mode='max-autotune', fullgraph=True)...")
        model.net = torch.compile(model.net, mode='max-autotune', fullgraph=True)
        return

    raise ValueError(f"Unknown strategy '{strategy}'")

def segment_nuclei(img, sample_name, save_dir='./nucleus_seg/', diameter=200,
                min_area=5000, max_area=500000, flow_threshold=0.9,
                cellprob_threshold=-0.3, save_mask_tiff=False, verbose=True,
                save_plot=True, model=None, strategy='baseline'):
    """
    Segment nuclei in a single image input as a numpy array. Returns the nuclei
    mask and saves the segmentation overlay as a PNG and optionally the labeled 
    mask as a TIFF. Increase flow_threshold to return more masks, decrease 
    cellprob_threshold to return more masks. Verbose mode will print additional 
    information.

    Mask structure:
    - For each nucleus, a unique integer label is assigned. The background is 0.
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Run Cellpose on the image
    masks = run_cellpose_2D(img, sample_name, SimpleNamespace(
        output_path=save_dir,
        diameter=diameter,
        min_area=min_area,
        max_area=max_area,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        save_mask_tiff=save_mask_tiff,
        verbose=verbose,
        save_plot=save_plot,
    ), model=model, strategy=strategy)
    return masks


def run_cellpose_2D(img, sample_name, params, model=None, strategy='baseline'):
    """
    Run Cellpose on a single image - updated for v4
    """
    if model is None:
        model = get_cellpose_model(strategy=strategy)
 
    
    
    eval_params = {
        'flow_threshold': params.flow_threshold,
        'cellprob_threshold': params.cellprob_threshold,
        'diameter': params.diameter,
    }

    mask, flow, diam = model.eval(img, **eval_params)
    
    # get rid of the nuclei that are too small or too big
    temp_mask = filter_nuclei_outside_size_thresholds(params, mask)
    final_mask = fix_mask_labels(temp_mask)

    # optionally save the mask as a TIFF file
    if params.save_mask_tiff:
        # Save the mask as a TIFF file
        output_name_tiff = os.path.join(params.output_path, f'{sample_name}_seg.tif')
        tifffile.imwrite(output_name_tiff, final_mask, compression='zstd')
    
    if params.save_plot:
        save_segmentation_plot(
            img,
            final_mask,
            os.path.join(params.output_path, sample_name + '_seg.png'),
        )

    if params.verbose:
        print(f"Processed {sample_name}: found {np.max(temp_mask)} nuclei")
    return final_mask


def save_segmentation_plot(img, mask, save_path):
    """
    Save a fast 2-panel Cellpose QC figure with centroid labels.

    This preserves the key QC signals from the prior utilities workflow:
    raw image context, colored mask overlay, and nucleus IDs.
    """
    img0 = img.copy()
    if img0.ndim == 3 and img0.shape[0] < 4:
        img0 = np.transpose(img0, (1, 2, 0))
    if img0.ndim < 3 or img0.shape[-1] < 3:
        img0 = plot.image_to_rgb(img0, channels=[0, 0])
    elif img0.max() <= 50.0:
        img0 = np.uint8(np.clip(img0, 0, 1) * 255)

    n_labels = int(mask.max())
    overlay = _fast_mask_overlay(img0, mask, n_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(img0)
    ax1.set_title("original image")
    ax1.axis('off')

    ax2.imshow(overlay)
    ax2.set_title("predicted masks")
    ax2.axis('off')

    if n_labels > 0:
        centroids = ndimage.center_of_mass(mask > 0, mask, range(1, n_labels + 1))
        for nucleus_id, (centroid_y, centroid_x) in enumerate(centroids, start=1):
            ax2.text(
                centroid_x,
                centroid_y,
                str(nucleus_id),
                color='white',
                fontsize=8,
                ha='center',
                va='center',
                fontweight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='black',
                    alpha=0.6,
                    edgecolor='none',
                ),
            )

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def _fast_mask_overlay(img0, mask, n_labels):
    """Vectorized HSV mask overlay compatible with Cellpose-style QC output."""
    from matplotlib.colors import hsv_to_rgb

    if img0.ndim > 2:
        gray = img0.astype(np.float32).mean(axis=-1)
    else:
        gray = img0.astype(np.float32)

    hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.float32)
    value = np.clip((gray / 255.0 if gray.max() > 1 else gray) * 1.5, 0, 1)
    foreground = mask > 0
    value[foreground] = np.maximum(value[foreground], 0.4)
    hsv[:, :, 2] = value

    hue_map = np.zeros(n_labels + 1, dtype=np.float32)
    if n_labels > 0:
        hue_map[1:] = np.linspace(0, 1, n_labels + 1)[np.random.permutation(n_labels)]

    hsv[:, :, 0] = hue_map[mask]
    hsv[:, :, 1] = foreground.astype(np.float32)

    return (hsv_to_rgb(hsv) * 255).astype(np.uint8)


def filter_nuclei_outside_size_thresholds(params, mask):
    """
    Filter out small and large objects - optimized version.
    
    Uses np.bincount for O(pixels) complexity instead of O(n_nuclei × pixels).
    """
    filtered_mask = mask.copy()
    
    # Compute area of all labels in ONE pass through the image
    # bincount[i] = number of pixels with value i
    areas = np.bincount(filtered_mask.ravel())
    
    # Find labels outside size thresholds (skip background at index 0)
    too_small = np.where((areas < params.min_area) & (np.arange(len(areas)) > 0))[0]
    too_big = np.where((areas > params.max_area) & (np.arange(len(areas)) > 0))[0]
    
    # Remove invalid labels
    labels_to_remove = np.concatenate([too_small, too_big])
    if len(labels_to_remove) > 0:
        # Create a lookup table: valid labels keep their value, invalid become 0
        lut = np.arange(len(areas), dtype=filtered_mask.dtype)
        lut[labels_to_remove] = 0
        filtered_mask = lut[filtered_mask]
    
    if params.verbose:
        print(f"Filtered out {len(labels_to_remove)} total puncta. "
              f"{len(too_big)} were too big and {len(too_small)} were too small.")
    
    return filtered_mask


def fix_mask_labels(mask):
    """
    Renumber labels to be contiguous
    """
    uniqueLabels = np.unique(mask)
    # Create mapping from old labels to new contiguous labels
    labelMap = np.zeros(np.max(mask) + 1, dtype=mask.dtype)
    labelMap[uniqueLabels] = np.arange(len(uniqueLabels))
    
    return labelMap[mask]
