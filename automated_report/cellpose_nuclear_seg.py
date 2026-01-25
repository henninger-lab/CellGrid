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
import tifffile

_cellpose_model = None

def get_cellpose_model():
    """Get or create the Cellpose model (singleton pattern)."""
    global _cellpose_model
    if _cellpose_model is None:
        print("  Loading Cellpose model (one-time)...")
        _cellpose_model = models.CellposeModel(pretrained_model='cpsam', gpu=True)
        io.logger_setup()
    return _cellpose_model


def segment_nuclei(img, sample_name, save_dir='./nucleus_seg/', diameter=200, 
                min_area=5000, max_area=500000, flow_threshold=0.9, 
                cellprob_threshold=-0.3, save_mask_tiff=False, verbose=True):
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
        verbose=verbose
    ))
    return masks

def run_cellpose_2D(img, sample_name, params):
    """
    Run Cellpose on a single image - updated for v4
    """
    model = get_cellpose_model()
 
    
    
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
    
    fig = plt.figure(figsize=(12, 3))

    # create a visualization of the segmentation and save as PNG
    plot.show_segmentation(fig, img, final_mask, flow[0])
    # Add nucleus ID labels at centroids
    
    ax = plt.gca()
    nucleus_ids = np.unique(final_mask)
    nucleus_ids = nucleus_ids[nucleus_ids != 0]  # Exclude background
    
    for nucleus_id in nucleus_ids:
        # Calculate centroid of this nucleus
        nucleus_coords = np.where(final_mask == nucleus_id)
        centroid_y = np.mean(nucleus_coords[0])
        centroid_x = np.mean(nucleus_coords[1])
        
        # Add text label at centroid
        ax.text(centroid_x, centroid_y, str(nucleus_id), 
                color='white', fontsize=8, ha='center', va='center',
                fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                         alpha=0.6, edgecolor='none'))
        

    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(params.output_path, sample_name + '_seg.png'), dpi=150)
    plt.close()
    if params.verbose:
        print(f"Processed {sample_name}: found {np.max(temp_mask)} nuclei")
    return final_mask


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