"""Exact, field-balanced display thresholds from nuclear/background histograms."""

import numpy as np

DEFAULT_PERCENTILES = (50.0, 99.5)


def resolve_contrast_groups(conditions, groups=None):
    """Validate disjoint explicit groups; omitted conditions remain independent."""
    result, seen = [], set()
    for group in groups or []:
        if not group:
            raise ValueError("Contrast groups cannot be empty.")
        for condition in group:
            if condition not in conditions:
                raise ValueError(f"Unknown condition in contrast group: {condition}")
            if condition in seen:
                raise ValueError(f"Condition appears more than once in contrast groups: {condition}")
            seen.add(condition)
        result.append(list(group))
    return result + [[condition] for condition in conditions if condition not in seen]


def validate_percentiles(percentiles):
    # These percentiles describe different populations; low need not be < high.
    if percentiles is not None and not percentiles:
        raise ValueError("Provide at least one contrast percentile pair.")
    for pair in percentiles or [DEFAULT_PERCENTILES]:
        if len(pair) != 2 or not all(np.isfinite(p) and 0 <= p <= 100 for p in pair):
            raise ValueError("Contrast percentiles must be finite numbers between 0 and 100.")


def field_histograms(image, mask):
    """Count every finite pixel, with no random sampling or crop duplication."""
    if image.ndim != 3 or mask.shape != image.shape[1:]:
        raise ValueError("Expected a CYX image and matching YX nuclear mask.")
    nuclear = mask > 0
    result = []
    for channel in image:
        populations = []
        for region in (~nuclear, nuclear):
            pixels = channel[region]
            values, counts = np.unique(pixels[np.isfinite(pixels)], return_counts=True)
            populations.append((values, counts))
        result.append(tuple(populations))
    return result


def balanced_percentile(histograms, percentile):
    """Inverse empirical CDF, giving each nonempty field equal total weight."""
    histograms = [(v, n) for v, n in histograms if n.size]
    if not histograms:
        return None
    values = np.concatenate([v for v, _ in histograms])
    weights = np.concatenate([n.astype(float) / n.sum() for _, n in histograms])
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    index = np.searchsorted(cumulative, percentile / 100 * cumulative[-1], side="left")
    return float(values[min(index, len(values) - 1)])


def compute_shared_contrast(fields, percentiles=None):
    """Return per-channel limits shared by all fields in a condition group.

    Empty populations do not contribute. With no background anywhere in the
    group, use zero as the display floor. A group without finite nuclear pixels
    in any channel cannot establish nuclear contrast and raises an error.
    """
    validate_percentiles(percentiles)
    percentiles = percentiles or [DEFAULT_PERCENTILES]
    if not fields:
        raise ValueError("No fields available for shared contrast.")
    n_channels = len(fields[0])
    if any(len(field) != n_channels for field in fields):
        raise ValueError("Fields sharing contrast must have the same channel count and order.")
    limits = {}
    for ch in range(n_channels):
        low, high = percentiles[min(ch, len(percentiles) - 1)]
        vmin = balanced_percentile([field[ch][0] for field in fields], low)
        vmax = balanced_percentile([field[ch][1] for field in fields], high)
        if vmax is None:
            raise ValueError(f"No finite nuclear pixels for contrast channel {ch}.")
        vmin = 0.0 if vmin is None else vmin
        limits[ch] = (vmin, vmax if vmax > vmin else vmin + 1.0)
    return limits


def validate_contrast_settings(mode, percentiles=None, groups=None):
    """Reject ambiguous mode/group combinations before processing images."""
    if mode not in ("shared", "per_field"):
        raise ValueError("Contrast mode must be 'shared' or 'per_field'.")
    validate_percentiles(percentiles)
    if mode == "per_field":
        if groups:
            raise ValueError("Condition groups are only available with shared contrast.")
        for low, high in percentiles or [(0.5, 99.5)]:
            if low >= high:
                raise ValueError("Per-field contrast requires the low percentile to be less than the high percentile.")
