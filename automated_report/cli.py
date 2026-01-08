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

from .report_generator import ReportConfig, generate_qc_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate automated QC reports from CZI microscopy files",
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
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        required=True,
        help='Directory containing CZI files'
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
        default=0.4,
        help='Cellpose flow threshold (default: 0.4, higher = more masks)'
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
    )
    
    # Run report generation
    generate_qc_report(config)


if __name__ == "__main__":
    main()