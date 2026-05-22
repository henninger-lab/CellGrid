import os
from types import SimpleNamespace
import numpy as np
from cellpose import models, io
from scipy import ndimage
import tifffile

VALID_STRATEGIES = ['baseline', 'torch_compile', 'torch_compile_max']
_cellpose_models = {}


def install_fast_flow_qc():
    """Patch Cellpose's GPU flow-error diffusion to avoid slow fp64 scatter work."""
    import cellpose.dynamics as dynamics
    import torch

    if getattr(dynamics, '_henninger_fast_flow_qc', False):
        return

    def _extend_centers_gpu_flat_float32(neighbors, meds, isneighbor, shape, n_iter=200, device=torch.device('cpu')):
        shape_tuple = tuple(int(v) for v in shape)
        t_flat = torch.zeros(int(np.prod(shape_tuple)), dtype=torch.float32, device=device)
        ndim = len(shape_tuple)

        if ndim == 2:
            _ly, lx = shape_tuple
            flat_neighbors = (neighbors[0] * lx + neighbors[1]).long()
            flat_meds = (meds[:, 0] * lx + meds[:, 1]).long()
        else:
            _lz, ly, lx = shape_tuple
            flat_neighbors = (neighbors[0] * (ly * lx) + neighbors[1] * lx + neighbors[2]).long()
            flat_meds = (meds[:, 0] * (ly * lx) + meds[:, 1] * lx + meds[:, 2]).long()

        flat_center = flat_neighbors[0]
        n_neighbors = flat_neighbors.shape[0]
        isneighbor = isneighbor.to(device=device, dtype=torch.float32)

        for _ in range(n_iter):
            t_flat[flat_meds] += 1
            t_neighbor = t_flat[flat_neighbors]
            t_flat[flat_center] = (t_neighbor * isneighbor).sum(dim=0) / n_neighbors

        if ndim == 2:
            grads = t_flat[flat_neighbors[[2, 1, 4, 3]]]
            dy = grads[0] - grads[1]
            dx = grads[2] - grads[3]
            return np.stack((dy.cpu().numpy(), dx.cpu().numpy()), axis=0)

        grads = t_flat[flat_neighbors[1:]]
        dz = grads[0] - grads[1]
        dy = grads[2] - grads[3]
        dx = grads[4] - grads[5]
        return np.stack((dz.cpu().numpy(), dy.cpu().numpy(), dx.cpu().numpy()), axis=0)

    dynamics._extend_centers_gpu = _extend_centers_gpu_flat_float32
    dynamics._henninger_fast_flow_qc = True


def resize_label_mask_nearest(mask, shape):
    """Upsample/downsample an integer label mask without interpolating labels."""
    if mask.shape == shape:
        return mask
    y_index = np.minimum((np.arange(shape[0]) * mask.shape[0] / shape[0]).astype(np.intp), mask.shape[0] - 1)
    x_index = np.minimum((np.arange(shape[1]) * mask.shape[1] / shape[1]).astype(np.intp), mask.shape[1] - 1)
    return mask[y_index[:, None], x_index]


def smooth_label_boundaries(mask, sigma=8.0, threshold=0.5):
    """Smooth blocky nearest-upsampled label edges while preserving integer labels."""
    if sigma <= 0 or np.max(mask) == 0:
        return mask

    labels = np.unique(mask)
    labels = labels[labels != 0]
    output = np.zeros(mask.shape, dtype=mask.dtype)
    score = np.zeros(mask.shape, dtype=np.float32)
    object_slices = ndimage.find_objects(mask)
    pad = int(np.ceil(max(4, sigma * 4)))

    for label in labels:
        object_slice = object_slices[int(label) - 1]
        if object_slice is None:
            continue
        y_slice, x_slice = object_slice
        y0 = max(0, y_slice.start - pad)
        y1 = min(mask.shape[0], y_slice.stop + pad)
        x0 = max(0, x_slice.start - pad)
        x1 = min(mask.shape[1], x_slice.stop + pad)

        crop = mask[y0:y1, x0:x1] == label
        blurred = ndimage.gaussian_filter(crop.astype(np.float32), sigma=sigma, mode='nearest')
        candidate = blurred >= threshold
        local_score = score[y0:y1, x0:x1]
        update = candidate & (blurred > local_score)
        output_crop = output[y0:y1, x0:x1]
        output_crop[update] = label
        local_score[update] = blurred[update]

    return output


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
                min_area=5000, max_area=500000, flow_threshold=0.2,
                cellprob_threshold=0.0, save_mask_tiff=False, verbose=True,
                save_plot=True, model=None, strategy='baseline', niter=200,
                resample=False, smooth_upscale_sigma=8.0, fast_flow_qc=True):
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
        niter=niter,
        resample=resample,
        smooth_upscale_sigma=smooth_upscale_sigma,
        fast_flow_qc=fast_flow_qc,
    ), model=model, strategy=strategy)
    return masks


def run_cellpose_2D(img, sample_name, params, model=None, strategy='baseline'):
    """
    Run Cellpose on a single image - updated for v4.
    """
    if model is None:
        model = get_cellpose_model(strategy=strategy)

    eval_params = {
        'flow_threshold': params.flow_threshold,
        'cellprob_threshold': params.cellprob_threshold,
        'diameter': params.diameter,
        'niter': getattr(params, 'niter', 200),
        'resample': getattr(params, 'resample', False),
    }

    if getattr(params, 'fast_flow_qc', True) and params.flow_threshold > 0:
        install_fast_flow_qc()

    mask, flow, diam = model.eval(img, **eval_params)

    if mask.shape != img.shape:
        mask = resize_label_mask_nearest(mask, img.shape)

    # get rid of the nuclei that are too small or too big
    temp_mask = filter_nuclei_outside_size_thresholds(params, mask)
    final_mask = fix_mask_labels(temp_mask)

    smooth_sigma = getattr(params, 'smooth_upscale_sigma', 0.0)
    if smooth_sigma > 0 and not getattr(params, 'resample', False):
        final_mask = smooth_label_boundaries(final_mask, sigma=smooth_sigma)
        final_mask = filter_nuclei_outside_size_thresholds(params, final_mask)
        final_mask = fix_mask_labels(final_mask)

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
        print(f"Processed {sample_name}: found {np.max(final_mask)} nuclei")
    return final_mask


def save_segmentation_plot(img, mask, save_path, max_size=768):
    """
    Save a fast 3-panel OpenCV QC image with centroid labels.
    """
    import cv2

    title = os.path.basename(save_path)
    for suffix in ('_seg.png', '_seg.jpg', '_seg.jpeg', '.png', '.jpg', '.jpeg'):
        if title.endswith(suffix):
            title = title[: -len(suffix)]
            break

    gray = robust_qc_gray(img, max_size=max_size)
    mask_small = resize_for_qc(mask, max_size=max_size, interpolation=cv2.INTER_NEAREST)

    gray_rgb = np.repeat(gray[..., None], 3, axis=2)
    label_rgb = colorize_labels_for_qc(mask_small)
    label_rgb = overlay_label_numbers(label_rgb, mask_small)
    overlay = gray_rgb.copy()
    overlay[label_boundaries(mask_small)] = (255, 20, 20)

    panels = [
        add_qc_title(gray_rgb, "Channel max projection"),
        add_qc_title(label_rgb, f"Cellpose-SAM mask ({int(mask.max())} objects)"),
        add_qc_title(overlay, "Mask boundary overlay"),
    ]
    panel_height = max(panel.shape[0] for panel in panels)
    normalized_panels = []
    for panel in panels:
        if panel.shape[0] < panel_height:
            pad = np.full((panel_height - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
            panel = np.vstack([panel, pad])
        normalized_panels.append(panel)

    separator = np.full((panel_height, 32, 3), 255, dtype=np.uint8)
    body = np.hstack([normalized_panels[0], separator, normalized_panels[1], separator, normalized_panels[2]])
    title_panel = np.full((42, body.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        title_panel,
        title[:180],
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        2,
        lineType=cv2.LINE_AA,
    )
    canvas = np.vstack([title_panel, body])
    write_qc_image(save_path, canvas)


def resize_for_qc(image, max_size, interpolation):
    import cv2

    height, width = image.shape[:2]
    scale = min(1.0, max_size / float(max(height, width)))
    if scale == 1.0:
        return image
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=interpolation)


def colorize_labels_for_qc(mask):
    import cv2

    labels = mask.astype(np.uint32, copy=False)
    hue = ((labels * 47) % 180).astype(np.uint8)
    saturation = np.full(labels.shape, 235, dtype=np.uint8)
    value = np.full(labels.shape, 245, dtype=np.uint8)
    hsv = np.stack([hue, saturation, value], axis=2)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[labels == 0] = 0
    return rgb


def robust_qc_gray(image, max_size):
    import cv2

    preview = np.asarray(image)
    if preview.ndim == 3:
        if preview.shape[0] < 4 and preview.shape[-1] >= 4:
            preview = np.max(preview, axis=0)
        elif preview.shape[-1] in (3, 4):
            preview = preview[..., :3].mean(axis=2)
        else:
            preview = np.max(preview, axis=0)
    preview = resize_for_qc(preview, max_size=max_size, interpolation=cv2.INTER_AREA)
    preview = preview.astype(np.float32, copy=False)
    finite = preview[np.isfinite(preview)]
    if finite.size == 0:
        return np.zeros(preview.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1, 99.8))
    if hi <= lo:
        hi = float(finite.max(initial=lo + 1.0))
    if hi <= lo:
        return np.zeros(preview.shape, dtype=np.uint8)
    gray = np.clip((preview - lo) / (hi - lo), 0, 1)
    return (gray * 255).astype(np.uint8)


def label_boundaries(mask):
    labels = np.asarray(mask)
    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundaries[:-1, :] |= labels[1:, :] != labels[:-1, :]
    boundaries[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundaries[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    return boundaries & (labels > 0)


def draw_text_with_box(
    image,
    text,
    origin,
    *,
    font_scale,
    color=(255, 255, 255),
    thickness=1,
    box_alpha=0.55,
    padding=3,
):
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    x0 = max(0, x - padding)
    y0 = max(0, y - text_size[1] - padding)
    x1 = min(image.shape[1], x + text_size[0] + padding)
    y1 = min(image.shape[0], y + baseline + padding)
    if x1 <= x0 or y1 <= y0:
        return
    roi = image[y0:y1, x0:x1]
    black = np.zeros_like(roi)
    image[y0:y1, x0:x1] = cv2.addWeighted(black, box_alpha, roi, 1.0 - box_alpha, 0)
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)


def overlay_label_numbers(panel, mask):
    import cv2

    labels = mask.astype(np.int64, copy=False)
    max_label = int(labels.max()) if labels.size else 0
    if max_label == 0:
        return panel

    flat = labels.ravel()
    counts = np.bincount(flat, minlength=max_label + 1)
    foreground_labels = np.nonzero(counts[1:] > 0)[0] + 1
    y_coords, x_coords = np.indices(labels.shape)
    x_sums = np.bincount(flat, weights=x_coords.ravel(), minlength=max_label + 1)
    y_sums = np.bincount(flat, weights=y_coords.ravel(), minlength=max_label + 1)
    font_scale = max(0.34, min(0.54, panel.shape[1] / 1800.0))

    for label in foreground_labels:
        count = counts[label]
        if count == 0:
            continue
        text = str(int(label))
        text_size, _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        x = int(round(x_sums[label] / count - text_size[0] / 2))
        y = int(round(y_sums[label] / count + text_size[1] / 2))
        draw_text_with_box(panel, text, (max(0, x), max(text_size[1], y)), font_scale=font_scale, thickness=1)
    return panel


def add_qc_title(panel, title, height=38):
    import cv2

    titled = np.full((panel.shape[0] + height, panel.shape[1], 3), 255, dtype=np.uint8)
    titled[height:] = panel
    cv2.putText(
        titled,
        title,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        1,
        lineType=cv2.LINE_AA,
    )
    return titled


def write_qc_image(save_path, canvas):
    import cv2

    output = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    suffix = os.path.splitext(save_path)[1].lower()
    if suffix in {'.jpg', '.jpeg'}:
        cv2.imwrite(save_path, output, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        cv2.imwrite(save_path, output, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def _contact_sheet_part_path(path, part_index, total_parts):
    if total_parts == 1:
        return path
    stem, suffix = os.path.splitext(path)
    return f"{stem}_part{part_index:03d}{suffix}"


def save_qc_contact_sheet(output_path, qc_paths, max_cols=2, max_tile_width=1800, batch_size=24):
    """Combine per-image segmentation QC files into one or more OpenCV contact sheets."""
    import cv2

    qc_paths = [str(path) for path in qc_paths if path]
    if not qc_paths:
        return []

    if batch_size <= 0:
        batches = [qc_paths]
    else:
        batches = [qc_paths[index:index + batch_size] for index in range(0, len(qc_paths), batch_size)]

    written_paths = []
    total_parts = len(batches)
    for part_index, batch_paths in enumerate(batches, start=1):
        images = []
        for qc_path in batch_paths:
            image_bgr = cv2.imread(qc_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if image.shape[1] > max_tile_width:
                scale = max_tile_width / float(image.shape[1])
                image = cv2.resize(
                    image,
                    (max_tile_width, max(1, int(round(image.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            images.append(image)

        if not images:
            continue

        cols = min(max_cols, len(images))
        rows = int(np.ceil(len(images) / cols))
        tile_height = max(image.shape[0] for image in images)
        tile_width = max(image.shape[1] for image in images)
        row_gap = 28
        col_gap = 36
        canvas_height = rows * tile_height + max(0, rows - 1) * row_gap
        canvas_width = cols * tile_width + max(0, cols - 1) * col_gap
        canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

        for index, image in enumerate(images):
            row = index // cols
            col = index % cols
            y0 = row * (tile_height + row_gap)
            x0 = col * (tile_width + col_gap)
            canvas[y0:y0 + image.shape[0], x0:x0 + image.shape[1]] = image

        part_path = _contact_sheet_part_path(str(output_path), part_index, total_parts)
        cv2.imwrite(part_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        written_paths.append(part_path)

    return written_paths


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
