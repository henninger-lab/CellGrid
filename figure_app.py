#!/usr/bin/env python3
"""
Streamlit GUI for Automated QC Report Generator.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import os
import sys
import zipfile
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from automated_report.report_generator import (
    ReportConfig,
    VALID_COLOC_COLORMAPS,
    generate_qc_report_with_progress,
)
from automated_report.parse_czi import CziImage, find_supported_image_files

genotype_matches = None


def validate_contrast_percentiles(contrast_percentiles):
    """Return validation errors and warnings for percentile pairs."""
    errors = []
    warnings = []

    for ch, (low, high) in enumerate(contrast_percentiles):
        if low >= high:
            errors.append(
                f"Channel {ch}: low percentile ({low}) must be less than high percentile ({high})."
            )
        elif (high - low) < 10.0:
            warnings.append(
                f"Channel {ch}: low/high percentiles are only {high - low:.1f} points apart "
                f"({low}, {high}). This may produce very aggressive contrast stretching."
            )

    return errors, warnings

# Page config
st.set_page_config(
    page_title="CellGrid Figure Generator",
    page_icon="🔬",
    layout="wide",
)

# Title and description
st.title("🔬 CellGrid Figure Generator")
st.markdown("""
Generate quality single cell comparison figures from CZI or OME-TIFF microscopy files with automated
nucleus segmentation, contrast normalization, and PDF export.
""")

st.divider()

# Create two columns for the main layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("📁 Input/Output")
    
    # Input directory path
    input_dir = st.text_input(
        "Input directory (containing CZI or OME-TIFF files)",
        value="",
        placeholder="/path/to/your/czi_files/",
        help="Enter the full path to your folder containing supported microscopy files"
    )
    
    # Scan for supported microscopy files if directory exists
    czi_files = []
    if input_dir and os.path.isdir(input_dir):
        czi_files = find_supported_image_files(input_dir)
        
        if czi_files:
            st.success(f"✓ Found {len(czi_files)} supported microscopy file(s)")
            with st.expander("View detected files"):
                for f in czi_files:
                    # Show file size
                    size_mb = os.path.getsize(f) / (1024 * 1024)
                    st.text(f"  • {os.path.basename(f)} ({size_mb:.1f} MB)")
            with st.expander("Channel metadata (from first file)", expanded=True):
                try:
                    import io
                    import sys
                    
                    # Suppress print output during loading
                    old_stdout = sys.stdout
                    try:
                        sys.stdout = io.StringIO()
                        sample_czi = CziImage(czi_files[0])
                    finally:
                        sys.stdout = old_stdout
                    
                    # Display channel info as a table
                    channel_data = []
                    for i, (fluor, wl) in enumerate(zip(sample_czi.channel_fluorophores, sample_czi.channel_wavelengths)):
                        ex = f"{wl['excitation']:.0f} nm" if wl.get('excitation') else "N/A"
                        em = f"{wl['emission']:.0f} nm" if wl.get('emission') else "N/A"
                        channel_data.append({
                            "Channel": i,
                            "Fluorophore": fluor,
                            "Excitation": ex,
                            "Emission": em,
                        })
                    
                    import pandas as pd
                    st.dataframe(pd.DataFrame(channel_data), hide_index=True, width='stretch')
                except Exception as e:
                    st.warning(f"Could not read channel metadata: {e}")
        else:
            st.warning("No supported microscopy files found in this directory")
    elif input_dir:
        st.error("Directory not found")
    
    # Output directory
    default_output = os.path.join(input_dir, "auto_figure") if input_dir else ""
    output_dir = st.text_input(
        "Output directory",
        value=default_output,
        placeholder="/path/to/output/",
        help="Where to save the generated reports (will be created if it doesn't exist)"
    )
    
    st.divider()
    
    # Genotypes input
    st.header("🧬 Genotypes")
    st.markdown("""Enter genotype names (one per line). Files will be matched by substring.
                **Before proceeding, make sure the genotype preview correctly matched your files!**""")
    
    genotypes_text = st.text_area(
        "Genotypes",
        value="MED1\nBRD4",
        height=120,
        help="Each genotype on a new line. Files containing the genotype string will be grouped together.",
        label_visibility="collapsed",
    )
    
    # Parse genotypes
    genotypes = [g.strip() for g in genotypes_text.strip().split('\n') if g.strip()]
    
    if genotypes:
        st.info(f"Genotypes: {', '.join(genotypes)}")
        
        # Preview genotype matching
        if czi_files:
            st.markdown("**Preview genotype matching**")
            for g in genotypes:
                matches = [os.path.basename(f) for f in czi_files if g in os.path.basename(f)]
                if matches:
                    st.markdown(f"**{g}**: {len(matches)} file(s)")
                    for m in matches[:5]:  # Show first 5
                        st.text(f"    • {m}")
                    if len(matches) > 5:
                        st.text(f"    ... and {len(matches) - 5} more")
                    genotype_matches = True
                else:
                    st.markdown(f"**{g}**: ⚠️ No matches")
                    genotype_matches = False

with col_right:
    st.header("⚙️ Figure Settings")
    
    # Tabs for different setting categories
    # tab_seg, tab_vis, tab_contrast = st.tabs(["Segmentation", "Visualization", "Contrast"])
    
    st.markdown("### Segmentation")
    nuclear_channel = st.number_input(
        "Nuclear channel index (see fluorophore info to the left)",
        min_value=0,
        max_value=10,
        value=0,
        help="Channel used for nucleus segmentation (usually DAPI/Hoechst)"
    )
    
    cellpose_diameter = st.number_input(
        "Expected nucleus diameter (pixels)",
        min_value=10,
        max_value=1000,
        value=450,
        help="Approximate diameter of nuclei in pixels"
    )

    nuclear_segmentation_blur_sigma = st.number_input(
        "Nuclear segmentation blur sigma (pixels)",
        min_value=0.0,
        max_value=50.0,
        value=8.0,
        step=0.5,
        help="2D Gaussian blur applied only to the max-projected nuclear image before segmentation"
    )
    
    exclude_border = st.checkbox(
        "Exclude border-touching nuclei",
        value=True,
        help="Remove nuclei that touch the image edge"
    )
    
    st.markdown("### Visualization")
    col_v1, col_v2 = st.columns(2)
        
    with col_v1:
        box_size_um = st.number_input(
            "Crop box size (µm)",
            min_value=5.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
        )
        
        scale_bar_um = st.number_input(
            "Scale bar length (µm)",
            min_value=1.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            )
    
    with col_v2:
        rows_per_page = st.number_input(
            "Rows per column",
            min_value=1,
            max_value=20,
            value=10,
        )
        
        nucleus_cols = st.number_input(
            "Columns per page",
            min_value=1,
            max_value=5,
            value=3,
        )
    
    show_outline = st.checkbox(
        "Show nucleus outlines",
        value=True,
        help="Draw dashed outline around each nucleus"
    )

    st.markdown("### Contrast Normalization")
    st.markdown("**Per-channel contrast percentiles**")
    st.caption("Set low and high percentiles for each channel. Leave empty to use defaults (0.5, 99.5). \nHigher percentiles for the max will preserve more of the dynamic range but may produce dimmer images if there are very bright pixels. Lower percentiles will produce brighter images but may saturate some pixels. Typically a max between 98 and 99.95 works well.")

    fish_contrast_setting = st.checkbox(
        "FISH contrast setting",
        value=False,
        help="Override all channels to use min percentile 90 and max percentile 100."
    )
    
    # Dynamic channel contrast inputs
    n_contrast_channels = st.number_input(
        "Number of channels to configure",
        min_value=1,
        max_value=8,
        value=2,
        help="Number of channels in your images"
    )
    
    contrast_percentiles = []
    for ch in range(n_contrast_channels):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            low = st.number_input(
                f"Ch{ch} low %",
                min_value=0.0,
                max_value=100.0,
                value=0.5,
                step=0.1,
                key=f"contrast_low_{ch}"
            )
        with col_c2:
            high = st.number_input(
                f"Ch{ch} high %",
                min_value=0.0,
                max_value=100.0,
                value=99.8,
                step=0.1,
                key=f"contrast_high_{ch}"
            )
        contrast_percentiles.append((low, high))

    if fish_contrast_setting:
        contrast_percentiles = [(90.0, 100.0) for _ in range(n_contrast_channels)]
        st.info("FISH contrast setting enabled: all channels will use min percentile 90 and max percentile 100.")

contrast_errors, contrast_warnings = validate_contrast_percentiles(contrast_percentiles)
for warning in contrast_warnings:
    st.warning(f"⚠️ {warning}")

# Channel colors section
st.divider()
st.header("🎨 Channel Colors")

col_colors = st.columns(min(n_contrast_channels, 4))
color_options = ["gray", "green", "magenta", "cyan", "red", "blue", "yellow", "orange", "white"]

channel_colors = []
for ch in range(n_contrast_channels):
    with col_colors[ch % len(col_colors)]:
        color = st.selectbox(
            f"Channel {ch}",
            options=color_options,
            index=min(ch + 1, len(color_options) - 1),  # Default to different colors
            key=f"color_{ch}"
        )
        channel_colors.append(color)

# Visible channels
st.divider()
visible_channels_enabled = True

if visible_channels_enabled:
    visible_channels = st.multiselect(
        "Channels to display",
        options=list(range(n_contrast_channels)),
        default=list(range(n_contrast_channels)),
        format_func=lambda x: f"Channel {x}"
    )
else:
    visible_channels = None

# Manual channel labels
st.divider()
st.header("🏷️ Channel Labels")
st.caption("These labels will appear at the top of the first set of figure panels. Use this to provide meaningful names for your channels (e.g. 'Hoechst', 'MED1').")

channel_labels = []
label_cols = st.columns(min(n_contrast_channels, 4))
for ch in range(n_contrast_channels):
    with label_cols[ch % len(label_cols)]:
        label = st.text_input(
            f"Label for Channel {ch}",
            value=f"Ch{ch}",
            key=f"label_{ch}",
        )
        channel_labels.append(label.strip() or f"Ch{ch}")

merged_label = st.text_input(
    "Merged panel label",
    value="Merged",
    help="Manual label for the merged image panel."
).strip() or "Merged"

# Colocalization section
st.divider()
st.header("🔬 Colocalization Analysis")
st.caption("Compute pixel-wise colocalization heatmap between two channels.")

enable_coloc = st.checkbox(
    "Enable colocalization heatmap",
    value=False,
    help="Add a colocalization panel showing spatial correlation between two channels"
)

coloc_channels = None
coloc_vmax = 20.0
coloc_colormap = "oslo"

if enable_coloc:
    col_coloc1, col_coloc2, col_coloc3 = st.columns(3)
    
    with col_coloc1:
        coloc_ch1 = st.selectbox(
            "First channel",
            options=list(range(n_contrast_channels)),
            index=0,
            format_func=lambda x: f"Channel {x}",
            key="coloc_ch1"
        )
    
    with col_coloc2:
        coloc_ch2 = st.selectbox(
            "Second channel",
            options=list(range(n_contrast_channels)),
            index=min(1, n_contrast_channels - 1),
            format_func=lambda x: f"Channel {x}",
            key="coloc_ch2"
        )

    with col_coloc3:
        coloc_colormap = st.selectbox(
            "Colormap",
            options=VALID_COLOC_COLORMAPS,
            index=VALID_COLOC_COLORMAPS.index("oslo"),
            help="Colormap used for the colocalization heatmap and its scale bar.",
        )
    
    if coloc_ch1 == coloc_ch2:
        st.warning("⚠️ Please select two different channels for colocalization")
        coloc_channels = None
    else:
        coloc_channels = (coloc_ch1, coloc_ch2)
    
    coloc_vmax = st.number_input(
        "Max brightness",
        min_value=5.0,
        max_value=100.0,
        value=20.0,
        step=5.0,
        help="Maximum value for colormap scaling. Lower values = brighter colocalization signal."
    )
    
    st.info("ℹ️ Colocalization is computed from the full 3D image before max projection. "
            "Only positive colocalization is shown; the selected colormap controls the heatmap and scale bar.")

# Generate button
st.divider()

# Validation
ready_to_run = len(czi_files) > 0 and len(genotypes) > 0 and output_dir and not contrast_errors

if not input_dir:
    st.warning("⚠️ Please enter an input directory path")
elif not czi_files:
    st.warning("⚠️ No supported microscopy files found in the input directory")
if not genotypes or genotype_matches is False:
    st.warning("⚠️ Please enter at least one genotype that matches your files")
if not output_dir:
    st.warning("⚠️ Please enter an output directory path")
for error in contrast_errors:
    st.error(f"⚠️ {error}")

# Run button
if st.button("🚀 Generate Reports", type="primary", disabled=not ready_to_run, width="stretch"):
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create config
    config = ReportConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        genotypes=genotypes,
        nuclear_channel=nuclear_channel,
        cellpose_diameter=cellpose_diameter,
        nuclear_segmentation_blur_sigma=nuclear_segmentation_blur_sigma,
        exclude_border=exclude_border,
        box_size_um=box_size_um,
        scale_bar_um=scale_bar_um,
        rows_per_page=rows_per_page,
        nucleus_cols=nucleus_cols,
        show_outline=show_outline,
        fish_contrast_setting=fish_contrast_setting,
        contrast_percentiles=contrast_percentiles if contrast_percentiles else None,
        channel_colors=channel_colors if channel_colors else None,
        visible_channels=visible_channels,
        channel_labels=channel_labels,
        merged_label=merged_label,
        enable_coloc=enable_coloc,
        coloc_channels=coloc_channels,
        coloc_vmax=coloc_vmax,
        coloc_colormap=coloc_colormap,
    )
    
    # Run report generation with progress tracking
    status_container = st.container()
    
    with status_container:
        # Progress tracking UI
        overall_progress = st.progress(0, text="Starting...")
        stage_status = st.status("Initializing pipeline...", expanded=True)
        
        with stage_status:
            current_file_text = st.empty()
            stage_text = st.empty()
            log_area = st.empty()
        
        # Create a progress callback class
        class ProgressTracker:
            def __init__(self, total_files):
                self.total_files = max(total_files, 1)
                self.current_file = 0
                self.logs = []
            
            def update(self, file_idx, filename, stage):
                self.current_file = file_idx
                progress = min((file_idx + 1) / self.total_files, 1.0)
                overall_progress.progress(progress, text=f"File {file_idx + 1} of {self.total_files}")
                current_file_text.markdown(f"**Current file:** `{filename}`")
                stage_text.markdown(f"**Stage:** {stage}")
            
            def log(self, message):
                self.logs.append(message)
                # Show last 15 log entries
                log_area.code("\n".join(self.logs[-15:]), language=None)
            
            def set_stage(self, stage_name):
                stage_status.update(label=stage_name)
        
        try:
            # Count matching files
            matching_files = [f for f in czi_files 
                           if any(g in os.path.basename(f) for g in genotypes)]
            
            if not matching_files:
                st.error("No files match the specified genotypes!")
                st.stop()
            
            tracker = ProgressTracker(len(matching_files))
            tracker.set_stage("🔄 Processing files...")
            
            # Run with progress tracking
            generate_qc_report_with_progress(config, tracker)
            
            stage_status.update(label="✅ Complete!", state="complete", expanded=False)
            overall_progress.progress(1.0, text="Done!")
            
            st.success("✅ Reports generated successfully!")
            
            # Show processing log
            with st.expander("View processing log"):
                st.code("\n".join(tracker.logs))
            
            # List generated files
            output_files = list(Path(output_dir).glob("*"))
            pdf_files = [f for f in output_files if f.suffix == ".pdf"]
            
            if pdf_files:
                st.subheader("📄 Generated Reports")
                st.markdown(f"**Output location:** `{output_dir}`")
                
                # List files
                for pdf_file in pdf_files:
                    size_mb = pdf_file.stat().st_size / (1024 * 1024)
                    st.markdown(f"- 📄 **{pdf_file.name}** ({size_mb:.1f} MB)")
                
                csv_files = [f for f in output_files if f.suffix == ".csv"]
                for csv_file in csv_files:
                    st.markdown(f"- 📊 **{csv_file.name}**")
                
                json_files = [f for f in output_files if f.suffix == ".json"]
                for json_file in json_files:
                    st.markdown(f"- 📋 **{json_file.name}**")
                
                # Open folder button hint
                st.info(f"📂 Open your file browser to: `{output_dir}`")
                
            else:
                st.warning("No PDF reports were generated. Check the processing log for details.")
                
        except Exception as e:
            stage_status.update(label="❌ Error", state="error")
            st.error(f"❌ Error during processing: {str(e)}")
            with st.expander("View error details"):
                import traceback
                st.code(traceback.format_exc())
            with st.expander("View processing log"):
                if 'tracker' in dir():
                    st.code("\n".join(tracker.logs))

# Footer
st.divider()
st.caption("Happy imaging! 🔬 Please report any issues to the GitHub repository.")
