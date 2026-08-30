# SRS–Spontaneous Raman Comparison: Analysis Code

This repository contains the Python code used for data processing, image registration, quantitative comparison, and figure generation for the associated study comparing hyperspectral stimulated Raman scattering (SRS) and spontaneous confocal Raman microscopy.

## Repository structure

```text
.
├── breast/
├── colon/
├── skin_section_1/
├── skin_section_2/
├── Script_for_Figures/
├── utils_registration.py
├── requirements.txt
└── README.md
```

The four sample folders contain the sample-specific processing and registration scripts. Their organization and workflow are similar across breast, colon, and the two skin sections.

`Script_for_Figures/` contains the scripts used for the quantitative analyses and for generating the main and supplementary figures, together with the corresponding utility modules.

## Main components

### Sample processing and registration

The scripts inside the individual sample folders perform the main preprocessing steps for each dataset, including:

- loading SRS and spontaneous Raman hyperspectral data;
- spectral calibration and preprocessing;
- SRS baseline correction;
- spatial registration of the paired acquisitions;
- generation of registered hyperspectral cubes used in the subsequent analyses.

Shared registration functions are provided in:

```text
utils_registration.py
```

### Figure and analysis scripts

The `Script_for_Figures/` directory contains scripts for the analyses reported in the manuscript, including:

- direct SRS–Raman image and spectral comparisons;
- mean and region-of-interest spectral comparisons;
- pixel-wise spectral similarity analysis;
- band-wise spatial agreement analysis;
- representative high- and low-agreement pixel spectra;
- permutation-based null-model analysis;
- spectral-continuity / system-dependent performance analysis.

The corresponding `utils_*.py` files contain reusable functions used by these scripts.

## Installation

The required Python packages are listed in:

```text
requirements.txt
```

A new environment can be created and the dependencies installed with:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

## Running the code

The scripts were used as analysis scripts rather than as a packaged Python application.

Before running them, update the data paths defined near the beginning of each script so that they point to the corresponding local dataset folders.

A typical workflow is:

1. Run the sample-specific preprocessing scripts.
2. Perform spatial registration of the paired SRS and spontaneous Raman datasets.
3. Run the scripts in `Script_for_Figures/` to reproduce the quantitative analyses and figures.

The scripts assume that the associated data repository has been downloaded and is accessible locally.

## Data

The corresponding SRS and spontaneous Raman datasets are available separately at:

**Data repository:** [ZENODO DOI / URL]

The directory names in the data repository correspond to the sample names used here:

```text
breast
colon
skin_section_1
skin_section_2
```

## Citation

If you use this code or the associated dataset, please cite the corresponding publication and archived repository.
