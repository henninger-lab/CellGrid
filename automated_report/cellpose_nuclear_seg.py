import os
from pathlib import Path

import numpy as np
import tifffile
from cellpose import io, models


DEFAULT_MODEL_NAME = "cpsam_v2"
DEFAULT_DIAMETER = 300
DEFAULT_PAD_TO_SHAPE = (2576, 2576)
SEGMENTATION_PIPELINE_VERSION = "radial_gold_v1"
_cached_models = {}


def get_model(model_name=DEFAULT_MODEL_NAME, prefer_gpu=True):
    """Load and cache a Cellpose model, preferring CUDA/MPS over CPU."""
    model_name = str(model_name)
    cache_key = (model_name, bool(prefer_gpu))
    if cache_key in _cached_models:
        return _cached_models[cache_key]

    model = models.CellposeModel(pretrained_model=model_name, gpu=prefer_gpu)
    try:
        io.logger_setup()
    except PermissionError as error:
        print(f"Cellpose logger setup skipped: {error}")

    loaded_model_path = Path(str(getattr(model, "pretrained_model", model_name))).expanduser()
    loaded_model = loaded_model_path.name
    if loaded_model != model_name:
        raise RuntimeError(f"Requested Cellpose model {model_name!r}, but loaded {loaded_model!r}.")

    device_type = getattr(getattr(model, "device", None), "type", None)
    if device_type in {"cuda", "mps"}:
        device_description = f"{device_type} GPU ({model.device})"
    else:
        device_description = f"CPU ({model.device})"

    model._henninger_model_name = model_name
    model._henninger_loaded_model_path = loaded_model_path
    model._henninger_device_description = device_description
    _cached_models[cache_key] = model
    print(f"Loaded Cellpose model {model_name!r} on {device_description}")
    return model


def segment_nuclei(
    img,
    sample_name,
    save_dir="./nucleus_seg/",
    diameter=DEFAULT_DIAMETER,
    min_area=5000,
    max_area=500000,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    save_mask_tiff=False,
    verbose=True,
    save_plot=True,
    model=None,
    model_name=DEFAULT_MODEL_NAME,
    prefer_gpu=True,
    niter=None,
    batch_size=8,
    tile_overlap=0.1,
    bsize=None,
    pad_to_shape=DEFAULT_PAD_TO_SHAPE,
    **_ignored_legacy_kwargs,
):
    """Segment one nuclear-channel image with Cellpose and local size filtering."""
    os.makedirs(save_dir, exist_ok=True)
    model = model or get_model(model_name=model_name, prefer_gpu=prefer_gpu)
    model_name = getattr(model, "_henninger_model_name", model_name)
    img = np.asarray(img)
    run_img = img
    crop_slices = None

    if pad_to_shape is not None:
        target_y, target_x = (int(pad_to_shape[0]), int(pad_to_shape[1]))
        if target_y <= 0 or target_x <= 0:
            raise ValueError(f"pad_to_shape must contain positive dimensions, got {pad_to_shape!r}")
        if img.shape[0] < target_y or img.shape[1] < target_x:
            padded_shape = (max(img.shape[0], target_y), max(img.shape[1], target_x))
            y0 = (padded_shape[0] - img.shape[0]) // 2
            x0 = (padded_shape[1] - img.shape[1]) // 2
            run_img = np.zeros(padded_shape, dtype=img.dtype)
            run_img[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
            crop_slices = (slice(y0, y0 + img.shape[0]), slice(x0, x0 + img.shape[1]))

    if isinstance(batch_size, str):
        batch_size = 8 if batch_size.strip().lower() == "auto" else int(batch_size)

    eval_params = {
        "diameter": diameter,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "batch_size": batch_size,
        "tile_overlap": tile_overlap,
    }
    if niter is not None:
        eval_params["niter"] = niter
    if bsize is not None:
        eval_params["bsize"] = bsize

    mask, _flow, _diam = model.eval(run_img, **eval_params)
    if mask.shape != run_img.shape:
        import cv2

        mask = cv2.resize(
            mask,
            (run_img.shape[1], run_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if crop_slices is not None:
        mask = mask[crop_slices]

    mask = filter_and_relabel_mask(mask, min_area=min_area, max_area=max_area, verbose=verbose)

    if save_mask_tiff:
        tifffile.imwrite(os.path.join(save_dir, f"{sample_name}_seg.tif"), mask, compression="zstd")

    if save_plot:
        save_segmentation_plot(
            img,
            mask,
            os.path.join(save_dir, f"{sample_name}_seg.png"),
            model_name=model_name,
        )

    if verbose:
        print(f"Processed {sample_name}: found {int(mask.max()) if mask.size else 0} nuclei")
    return mask


def segment_nuclei_batch_2D(
    image_items,
    save_dir="./nucleus_seg/",
    diameter=DEFAULT_DIAMETER,
    min_area=5000,
    max_area=500000,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    save_mask_tiff=True,
    verbose=True,
    save_plot=True,
    model=None,
    model_name=DEFAULT_MODEL_NAME,
    prefer_gpu=True,
    niter=None,
    batch_size=8,
    tile_overlap=0.1,
    bsize=None,
    pad_to_shape=DEFAULT_PAD_TO_SHAPE,
    return_names=False,
    **_ignored_legacy_kwargs,
):
    """Segment a sequence of (sample_name, image) records through segment_nuclei."""
    model = model or get_model(model_name=model_name, prefer_gpu=prefer_gpu)
    results = []
    for item in image_items:
        sample_name, img = item[0], item[1]
        mask = segment_nuclei(
            img,
            sample_name,
            save_dir=save_dir,
            diameter=diameter,
            min_area=min_area,
            max_area=max_area,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            save_mask_tiff=save_mask_tiff,
            verbose=verbose,
            save_plot=save_plot,
            model=model,
            model_name=model_name,
            prefer_gpu=prefer_gpu,
            niter=niter,
            batch_size=batch_size,
            tile_overlap=tile_overlap,
            bsize=bsize,
            pad_to_shape=pad_to_shape,
        )
        results.append((sample_name, mask))

    if return_names:
        return results
    return [mask for _sample_name, mask in results]


def segment_from_cache(
    cache_folder,
    output_folder="./results/",
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    diameter=DEFAULT_DIAMETER,
    nuclear_channel=0,
    save_plot=True,
    batch_size=8,
    model=None,
    pad_to_shape=DEFAULT_PAD_TO_SHAPE,
    **_ignored_legacy_kwargs,
):
    """Segment cached max-projection TIFF files with the standard nuclei path."""
    from tqdm import tqdm

    os.makedirs(output_folder, exist_ok=True)
    tif_files = sorted(Path(cache_folder).glob("*.tif"))
    if not tif_files:
        print(f"No .tif files found in {cache_folder}")
        print("Run cache_max_projections.py first to create cache files")
        return

    print(f"Found {len(tif_files)} cached images to segment")
    print("Parameters:")
    print(f"  Diameter: {diameter}")
    print(f"  Flow threshold: {flow_threshold}")
    print(f"  Cellprob threshold: {cellprob_threshold}")
    print(f"  Batch size: {batch_size}")
    print(f"  Pad to shape: {pad_to_shape}")
    print()

    model = model or get_model()
    qc_paths = []
    if isinstance(batch_size, str):
        if batch_size.strip().lower() == "auto":
            batch_size = 8
        elif "," in batch_size:
            batch_size = max(int(part.strip()) for part in batch_size.split(",") if part.strip())
        else:
            batch_size = int(batch_size)

    for tif_file in tqdm(tif_files, desc="Segmenting nuclei"):
        sample_name = Path(tif_file).stem
        try:
            try:
                img = tifffile.imread(tif_file, key=nuclear_channel)
                if img.ndim != 2:
                    raise ValueError("single-page read did not return a 2D image")
            except Exception:
                img = tifffile.imread(tif_file)[nuclear_channel, :, :]

            print(f"\nProcessing {sample_name} (shape: {img.shape})")
            mask = segment_nuclei(
                img,
                sample_name,
                save_dir=output_folder,
                save_mask_tiff=True,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                diameter=diameter,
                verbose=True,
                save_plot=save_plot,
                model=model,
                batch_size=batch_size,
                pad_to_shape=pad_to_shape,
            )

            print(f"  Found {int(mask.max()) if mask.size else 0} nuclei")
            qc_path = os.path.join(output_folder, f"{sample_name}_seg.png")
            if save_plot and os.path.exists(qc_path):
                qc_paths.append(qc_path)
        except Exception as error:
            print(f"\nError processing {sample_name}: {error}")

    if qc_paths:
        for contact_sheet_path in save_qc_contact_sheet(
            os.path.join(output_folder, "segmentation_qc_contact_sheet.png"),
            qc_paths,
        ):
            print(f"Segmentation QC contact sheet: {contact_sheet_path}")

    print(f"\nSegmentation complete! Results saved to {output_folder}")


def filter_and_relabel_mask(mask, min_area=5000, max_area=500000, verbose=True):
    """Filter labels by area and relabel the retained nuclei contiguously."""
    max_label = int(mask.max()) if mask.size else 0
    if max_label == 0:
        return np.zeros(mask.shape, dtype=mask.dtype)

    areas = np.bincount(mask.ravel(), minlength=max_label + 1)
    too_small = np.where(areas[1:] < min_area)[0] + 1
    too_big = np.where(areas[1:] > max_area)[0] + 1

    keep = np.ones(max_label + 1, dtype=bool)
    keep[0] = False
    keep[too_small] = False
    keep[too_big] = False

    mapping = np.zeros(max_label + 1, dtype=mask.dtype)
    kept_labels = np.nonzero(keep)[0]
    if kept_labels.size:
        mapping[kept_labels] = np.arange(1, kept_labels.size + 1, dtype=mask.dtype)

    if verbose:
        bad_count = len(too_small) + len(too_big)
        print(
            f"Filtered out {bad_count} nuclei. "
            f"{len(too_big)} were too big and {len(too_small)} were too small."
        )
    return mapping[mask]


def save_segmentation_plot(img, mask, save_path, max_size=768, model_name=DEFAULT_MODEL_NAME):
    """Save a fast 3-panel OpenCV QC image with centroid labels."""
    import cv2

    title = os.path.basename(save_path)
    for suffix in ("_seg.png", "_seg.jpg", "_seg.jpeg", ".png", ".jpg", ".jpeg"):
        if title.endswith(suffix):
            title = title[: -len(suffix)]
            break

    gray = robust_qc_gray(img, max_size=max_size)
    mask_small = resize_for_qc(mask, max_size=max_size, interpolation=cv2.INTER_NEAREST)

    gray_rgb = np.repeat(gray[..., None], 3, axis=2)
    label_rgb = overlay_label_numbers(colorize_labels_for_qc(mask_small), mask_small)
    overlay = gray_rgb.copy()
    overlay[label_boundaries(mask_small)] = (255, 20, 20)

    panels = [
        add_qc_title(gray_rgb, "Channel max projection"),
        add_qc_title(label_rgb, f"{model_name} mask ({int(mask.max())} objects)"),
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
    write_qc_image(save_path, np.vstack([title_panel, body]))


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
        draw_text_with_box(panel, text, (max(0, x), max(text_size[1], y)), font_scale=font_scale)
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
    if suffix in {".jpg", ".jpeg"}:
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
