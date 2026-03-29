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

## Shared Loader
The repo includes a reusable raw-image reader in [MICROSCOPY_LOADER.md](/home/grant/Desktop/henninger/CellGrid/MICROSCOPY_LOADER.md). It supports `.czi` and aligned `.ome.tif` / `.ome.tiff` files behind one API so other Henninger tools can reuse the same loader instead of maintaining separate format-specific readers.

## How it works
CellGrid automates the process of converting raw microscopy data into organized figures.

1. Reads Data: The tool scans your input directory for supported microscopy files (`.czi`, `.ome.tif`, `.ome.tiff`) and groups them into experimental conditions (genotypes) based on text patterns you provide (e.g., "MED1", "BRD4").

2. Finds Cells (Segmentation): It uses Cellpose, a deep-learning algorithm, to automatically detect nuclei in your specified channel index.

3. Extracts Single Cells: For every detected nucleus, the tool creates a "crop" (a small, focused image). It automatically filters out "bad" crops, such as nuclei touching the image border or those that are too small/large (debris or artifacts).

4. Visualization: It calculates brightness/contrast values using percentiles that you can change. It calculates contrast per file, meaning it should work even if you change laser or exposure settings during the experiment. It then applies colors via a lookup table with colors that you can specify by name. * Note you can see the exact brightness/contrast settings applied in the nuclear metadata file.

5. Generates Reports:
    - PDF Reports: Creates a multi-page PDF for each genotype, displaying every single cell side-by-side with individual channels and merged views. Visualization is entirely done using matplotlib.

    - Metadata: Exports a CSV file containing measurements (position, size, intensity) for every cell.

    - Caching: It saves the heavy computational work (finding cells) in a cache/ folder. This means you can change visualization settings (like colors or scale bars) and regenerate the report almost instantly without re-processing the raw images.

    - Segmentation visualizations: If you are unsure about the quality of segmentation, you can inspects how the masks look in the segmentation folder.
