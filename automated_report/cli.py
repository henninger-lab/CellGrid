#!/usr/bin/env python3
"""
Command-line interface for automated QC report generation.

Usage:
    python -m automated_report --input_dir ./czi_files/ --genotypes WT KO --nuclear_channel 0
    
Or:
    python automated_report/cli.py --input_dir ./czi_files/ --genotypes WT KO --nuclear_channel 0
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .report_generator import (
    ReportConfig,
    generate_qc_report_with_progress,
    VALID_COLOC_COLORMAPS,
)


def validate_contrast_percentiles(
    contrast_percentiles: list[tuple[float, float]],
    *,
    error_fn,
    warning_fn,
):
    """Validate percentile pairs and emit warnings for narrow ranges."""
    for ch, (low, high) in enumerate(contrast_percentiles):
        if low >= high:
            error_fn(
                f"Invalid contrast percentile pair for channel {ch}: "
                f"low={low} must be less than high={high}."
            )
        if (high - low) < 10.0:
            warning_fn(
                f"Warning: channel {ch} contrast percentiles are only "
                f"{high - low:.1f} percentage points apart ({low}, {high}). "
                "This may produce very aggressive contrast stretching."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate automated QC reports from CZI or OME-TIFF microscopy files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python -m automated_report --input_dir ./images/ --genotypes WT KO Het --nuclear_channel 0

    # With custom colors
    python -m automated_report \\
        --input_dir ./images/ \\
        --genotypes WT KO \\
        --nuclear_channel 0 \\
        --channel_colors magenta green

    # Per-channel contrast percentiles (different for each channel)
    python -m automated_report \\
        --input_dir ./images/ \\
        --genotypes WT KO \\
        --nuclear_channel 0 \\
        --contrast_percentiles 0.5,95.0 10,99.95

    # Disable nucleus outlines
    python -m automated_report \\
        --input_dir ./images/ \\
        --genotypes WT KO \\
        --nuclear_channel 0 \\
        --no_outline

    # Specify visible channels (e.g., exclude DAPI at channel 0)
    python -m automated_report \\
        --input_dir ./images/ \\
        --genotypes WT KO \\
        --nuclear_channel 0 \\
        --visible_channels 1 2

    # Enable colocalization analysis between channels 0 and 1
    python -m automated_report \\
        --input_dir ./images/ \\
        --genotypes WT KO \\
        --nuclear_channel 0 \\
        --coloc_channels 0 1 \\
        --coloc_vmax 0.3
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        required=True,
        help='Directory containing supported microscopy files (.czi, .ome.tif, .ome.tiff)'
    )
    parser.add_argument(
        '--genotypes', '-g',
        nargs='+',
        required=True,
        help='Genotype patterns to match in filenames (e.g., WT KO Het)'
    )
    parser.add_argument(
        '--nuclear_channel', '-n',
        type=int,
        required=True,
        help='Channel index for nuclear stain (used for segmentation)'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='./qc_reports/',
        help='Output directory for reports (default: ./qc_reports/)'
    )
    
    # Segmentation options
    parser.add_argument(
        '--diameter',
        type=int,
        default=300,
        help='Expected nucleus diameter in pixels (default: 300)'
    )
    parser.add_argument(
        '--flow_threshold',
        type=float,
        default=0.2,
        help='Cellpose flow threshold (default: 0.2, higher = more masks)'
    )
    parser.add_argument(
        '--cellprob_threshold',
        type=float,
        default=0.0,
        help='Cellpose cell probability threshold (default: 0.0, lower = more masks)'
    )
    parser.add_argument(
        '--min_area',
        type=int,
        default=5000,
        help='Minimum nucleus area in pixels (default: 5000)'
    )
    parser.add_argument(
        '--max_area',
        type=int,
        default=500000,
        help='Maximum nucleus area in pixels (default: 500000)'
    )
    parser.add_argument(
        '--nuclear_segmentation_blur_sigma',
        type=float,
        default=8.0,
        help='2D Gaussian blur sigma in pixels applied only to the nuclear segmentation input (default: 8.0)'
    )
    parser.add_argument(
        '--include_border',
        action='store_true',
        help='Include nuclei touching image border (default: exclude)'
    )
    
    # Visualization options
    parser.add_argument(
        '--box_size_um',
        type=float,
        default=20.0,
        help='Crop box size in microns (default: 20.0)'
    )
    parser.add_argument(
        '--scale_bar_um',
        type=float,
        default=5.0,
        help='Scale bar length in microns (default: 5.0)'
    )
    parser.add_argument(
        '--scale_bar_fontsize',
        type=int,
        default=8,
        help='Scale bar label font size (default: 8)'
    )
    parser.add_argument(
        '--gap_pixels',
        type=int,
        default=10,
        help='Gap between panels in pixels (default: 10)'
    )
    parser.add_argument(
        '--border_pixels',
        type=int,
        default=10,
        help='Border around panels in pixels (default: 10)'
    )
    parser.add_argument(
        '--rows_per_page',
        type=int,
        default=10,
        help='Number of nucleus rows per column (default: 10)'
    )
    parser.add_argument(
        '--nucleus_cols',
        type=int,
        default=3,
        help='Number of nucleus columns per page (default: 3)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output DPI (default: 300)'
    )
    
    # Contrast options
    parser.add_argument(
        '--contrast_percentiles',
        nargs='+',
        type=str,
        default=None,
        help='Contrast percentile limits per channel as low,high pairs. '
             'E.g., --contrast_percentiles 0.5,95.0 10,99.95 for 2 channels. '
             'If fewer pairs than channels, last pair is reused. '
             'Default: 0.5,99.5 for all channels.'
    )
    
    # Outline options
    parser.add_argument(
        '--no_outline',
        action='store_true',
        help='Disable nucleus outline drawing (default: outlines enabled)'
    )
    
    # Channel options
    parser.add_argument(
        '--visible_channels',
        nargs='+',
        type=int,
        default=None,
        help='Channel indices to include in figures (default: all)'
    )
    parser.add_argument(
        '--channel_colors',
        nargs='+',
        type=str,
        default=None,
        help='Colors for each channel (e.g., --channel_colors gray green magenta red). '
             'Accepts color names (green, magenta, cyan, red, blue, yellow, gray, white) '
             'or hex codes (#00FF00). Order matches channel indices.'
    )
    
    # Caching options
    parser.add_argument(
        '--no_cache',
        action='store_true',
        help='Disable caching (recompute everything)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='Custom cache directory (default: output_dir/cache/)'
    )
    
    # Colocalization options
    parser.add_argument(
        '--coloc_channels',
        nargs=2,
        type=int,
        default=None,
        metavar=('CH1', 'CH2'),
        help='Enable colocalization analysis between two channels. '
             'E.g., --coloc_channels 0 1 to analyze channels 0 and 1.'
    )
    parser.add_argument(
        '--coloc_vmax',
        type=float,
        default=0.3,
        help='Max brightness for colocalization colormap (default: 0.3). '
             'Lower values = brighter signal.'
    )
    parser.add_argument(
        '--coloc_colormap',
        type=str,
        default='oslo',
        choices=VALID_COLOC_COLORMAPS,
        help='Colormap for colocalization panels.'
    )
    
    args = parser.parse_args()
    
    # Parse contrast percentiles
    contrast_percentiles = None
    if args.contrast_percentiles:
        contrast_percentiles = []
        for pair_str in args.contrast_percentiles:
            try:
                low, high = pair_str.split(',')
                contrast_percentiles.append((float(low), float(high)))
            except ValueError:
                print(f"Warning: Invalid contrast percentile pair '{pair_str}', expected format: low,high")
                print("Using default 0.5,99.5")
                contrast_percentiles.append((0.5, 99.5))

        validate_contrast_percentiles(
            contrast_percentiles,
            error_fn=parser.error,
            warning_fn=lambda msg: print(msg, file=sys.stderr),
        )
    
    # Create configuration
    config = ReportConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        genotypes=args.genotypes,
        nuclear_channel=args.nuclear_channel,
        cellpose_diameter=args.diameter,
        cellpose_flow_threshold=args.flow_threshold,
        cellpose_cellprob_threshold=args.cellprob_threshold,
        min_area_pixels=args.min_area,
        max_area_pixels=args.max_area,
        nuclear_segmentation_blur_sigma=args.nuclear_segmentation_blur_sigma,
        exclude_border=not args.include_border,
        box_size_um=args.box_size_um,
        scale_bar_um=args.scale_bar_um,
        scale_bar_fontsize=args.scale_bar_fontsize,
        gap_pixels=args.gap_pixels,
        border_pixels=args.border_pixels,
        rows_per_page=args.rows_per_page,
        nucleus_cols=args.nucleus_cols,
        dpi=args.dpi,
        contrast_percentiles=contrast_percentiles,
        visible_channels=args.visible_channels,
        channel_colors=args.channel_colors,
        show_outline=not args.no_outline,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        enable_coloc=args.coloc_channels is not None,
        coloc_channels=tuple(args.coloc_channels) if args.coloc_channels else None,
        coloc_vmax=args.coloc_vmax,
        coloc_colormap=args.coloc_colormap,
    )
    
    # Run report generation
    generate_qc_report_with_progress(config)


if __name__ == "__main__":
    main()
