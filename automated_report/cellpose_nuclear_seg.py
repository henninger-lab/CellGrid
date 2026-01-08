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
    # Use the new CPSAM model for best performance
    model = models.CellposeModel(pretrained_model='cpsam', gpu=True)
    io.logger_setup()
    
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
    plt.savefig(os.path.join(params.output_path, sample_name + '_seg.png'), dpi=300)
    plt.close()
    if params.verbose:
        print(f"Processed {sample_name}: found {np.max(temp_mask)} nuclei")
    return final_mask


def filter_nuclei_outside_size_thresholds(params, mask):
    """
    Filter out small and large objects
    """
    filtered_mask = mask.copy()
    num_of_nuclei = np.max(filtered_mask)
    tooBig = 0
    tooSmall = 0
    for n in np.arange(1, num_of_nuclei + 1):
        nuc_area = np.sum(filtered_mask == n)
        if nuc_area < params.min_area:
            filtered_mask[filtered_mask == n] = 0
            tooSmall += 1
        elif nuc_area > params.max_area:
            filtered_mask[filtered_mask == n] = 0
            tooBig += 1
    if params.verbose:
        print(f"Filtered out {tooBig + tooSmall} total puncta. {tooBig} were too big and {tooSmall} were too small.")
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