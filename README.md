# CellGrid
A tool for easily creating figures to visualize and compare many cells from 3D input microscopy images using a GUI interface.

## Quick Start
1. Clone this github repo using Github Desktop (easiest) or enter the following command in the terminal. If using the command, start in whatever directory (folder) you want this to live in (e.g. Desktop).
```
git clone https://github.com/henninger-lab/CellGrid.git
cd CellGrid
```

2. Create the virtual environment using conda. *If you don't have conda yet, get it from https://anaconda.com/download
```
conda env create -f environment.yml
```
3. Activate the virtual environment.
```
conda activate figure-gen
```

4. Launch the GUI interface for the app.
```
streamlit run figure_app.py
```

## Input/Output
- Input: folder containing z-stack `.czi`, `.ome.tif`, or `.ome.tiff` images
- Outputs:  
    - genotype1_report.pdf
        - figure combining images for each genotype
    - nucleus_metadata.csv
        - file tracking nulcear ID locations and source files
    - run_parameters.json
        - list of all parameters used to generate the figures
    - segmentation/
        - folder containing all segmentation masks and visualizations
    - cache/ 
        - folder containing compressed max z projections of the input images

## Ongoing usage
After setting up for the first time, all you need to do for future use is navigate to the CellGrid folder, activate the conda environment (Quick Start step 3), and launch the app (Quick Start step 4)

If a run reports `Requested Cellpose model 'cpsam_v2', but loaded 'cpsam'`, the active environment may still have an older Cellpose installation. Stop the app, update the existing environment, and restart using its Python:

```sh
conda activate figure-gen
python -m pip install cellpose==4.2.1.1
python -m streamlit run figure_app.py
```

Updating the repository does not update packages in an existing environment.

## Shared Loader
The repo includes a reusable raw-image reader in [MICROSCOPY_LOADER.md](/home/grant/Desktop/henninger/CellGrid/MICROSCOPY_LOADER.md). It supports `.czi` and aligned `.ome.tif` / `.ome.tiff` files behind one API so other Henninger tools can reuse the same loader instead of maintaining separate format-specific readers.

## How it works
CellGrid automates the process of converting raw microscopy data into organized figures.

1. Reads Data: The tool scans your input directory for supported microscopy files (`.czi`, `.ome.tif`, `.ome.tiff`) and groups them into experimental conditions (genotypes) based on text patterns you provide (e.g., "MED1", "BRD4").

2. Finds Cells (Segmentation): It uses Cellpose, a deep-learning algorithm, to automatically detect nuclei in your specified channel index.

3. Extracts Single Cells: For every detected nucleus, the tool creates a "crop" (a small, focused image). It automatically filters out "bad" crops, such as nuclei touching the image border or those that are too small/large (debris or artifacts).

4. Visualization: All fields within a condition use the same brightness/contrast limits per channel. By default the lower limit is the median background intensity (pixels outside segmented nuclei), and the upper limit is the 99.5th percentile inside nuclei. Every finite pixel is counted, with each field given equal weight separately for background and nuclear pixels. You can adjust both percentiles per channel. Colors are applied using your chosen lookup tables, and the actual intensity limits are saved in the nuclear metadata file.

   To share contrast between conditions, enter one comma-separated set per line in **Conditions sharing brightness/contrast**, for example `WT, KO` on one line and `Drug A, Drug B` on another. Each set is thresholded independently; unlisted conditions retain their own thresholds. Conditions may appear in only one set. Reports remain separate for each condition. Shared conditions should have matching channel order and comparable acquisition settings.

   From the command line, use `--contrast_group WT KO --contrast_group "Drug A" "Drug B"`. In Python, set `contrast_groups=[["WT", "KO"], ["Drug A", "Drug B"]]` in `ReportConfig`. The default `contrast_percentiles` is `(50, 99.5)` per channel: background percentile followed by nuclear percentile. These refer to different pixel populations, so they need not be numerically ordered.

   Fields with no nuclear pixels do not contribute to the nuclear threshold; fields with no background pixels do not contribute to the background threshold. If an entire group has no background, the display floor is zero. Constant or inverted intensity limits use a one-intensity-unit display range above the lower limit. To restore the previous behavior, select **Per-field automatic contrast — each field scaled independently**, or pass `--contrast_mode per_field` (Python: `contrast_mode="per_field"`). This optional mode uses full-image percentiles independently for each field, defaulting to 0.5–99.5%. Displayed brightness is not comparable between fields in this mode, and condition groups are unavailable. Shared normalization remains the default.

5. Generates Reports:
    - PDF Reports: Creates a multi-page PDF for each genotype, displaying every single cell side-by-side with individual channels and merged views. Visualization is entirely done using matplotlib.

    - Metadata: Exports a CSV file containing measurements (position, size, intensity) for every cell.

    - Caching: It saves the heavy computational work (finding cells) in a cache/ folder. This means you can change visualization settings (like colors or scale bars) and regenerate the report almost instantly without re-processing the raw images.

    - Segmentation visualizations: If you are unsure about the quality of segmentation, you can inspects how the masks look in the segmentation folder.

## Colocalization colormaps
The 3D colocalization score offers **Oslo** (default) and **White to black**. With White to black, zero and negative scores appear white, scores at or above the upper brightness threshold appear black, and intermediate scores appear gray. The PDF scale bar uses the same colors. Select it in the colocalization settings or use `--coloc_colormap white_to_black`.
