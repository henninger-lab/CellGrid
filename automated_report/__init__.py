"""
Automated QC Report Generator for Microscopy Images.

This package provides automated nucleus segmentation and figure generation
for fluorescence microscopy CZI and OME-TIFF files. It integrates with the existing
rep_image_workflow package for consistent figure styling.

Main entry points:
- generate_qc_report(config): Main function to generate reports
- ReportConfig: Configuration dataclass for all settings

Example usage:
    from automated_report import generate_qc_report, ReportConfig
    
    config = ReportConfig(
        input_dir='./czi_files/',
        output_dir='./reports/',
        genotypes=['WT', 'KO', 'Het'],
        nuclear_channel=0,
        box_size_um=20.0,
    )
    
    generate_qc_report(config)

Or use the command-line interface:
    python -m automated_report --input_dir ./czi_files/ --genotypes WT KO --nuclear_channel 0
"""

from .report_generator import (
    ReportConfig,
    generate_qc_report_with_progress,
    generate_multipage_pdf,
    save_metadata_csv,
)

from .nucleus_extractor import (
    NucleusCrop,
    extract_nuclei_from_mask,
    extract_crop_data,
    is_border_touching,
    compute_nucleus_centroid,
    compute_nucleus_area,
)

from .contrast_calculator import (
    compute_global_contrast,
    apply_global_contrast,
    compute_per_nucleus_contrast,
    format_contrast_summary,
)

from .coloc_heatmap import (
    compute_coloc_max_projection,
    render_coloc_heatmap,
    coloc_heatmap,
)


__all__ = [
    # Main entry points
    'ReportConfig',
    'generate_qc_report_with_progress',
    
    # PDF generation
    'generate_multipage_pdf',
    'save_metadata_csv',
    
    # Nucleus extraction
    'NucleusCrop',
    'extract_nuclei_from_mask',
    'extract_crop_data',
    'is_border_touching',
    'compute_nucleus_centroid',
    'compute_nucleus_area',
    
    # Contrast calculation
    'compute_global_contrast',
    'apply_global_contrast',
    'compute_per_nucleus_contrast',
    'format_contrast_summary',
    
    # Colocalization
    'compute_coloc_max_projection',
    'render_coloc_heatmap',
    'coloc_heatmap',
]

__version__ = '1.0.0'
