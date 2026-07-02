# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
EMI-Mesh is an automated pipeline for generating high-quality tetrahedral meshes of brain tissue (both extracellular and intracellular spaces) from imaging data, suitable for numerical simulations. It uses [fTetWild](https://github.com/wildmeshing/fTetWild) for volumetric meshing.

### Core Architecture
The pipeline follows a sequential flow: **Data Download $\to$ Image Processing $\to$ Surface Extraction $\to$ Volumetric Meshing**.

- **Pipeline Orchestration**: Driven by `Snakemake`. Configuration is YAML-based (found in `config_files/` or `configfiles/`).
- **Processing Engine**: Core logic in `src/emimesh/`.
  - **Configuration Blocks**:
    - `raw`: Controls data source (CloudVolume/WebKnossos), resolution (`mip`), and volume extent.
    - `processing`: Defines resampling (`dx`) and a sequence of image processing `operation` strings.
    - `meshing`: Controls `fTetWild` parameters (e.g., `envelopsize`, `stopquality`).
  - **Image Operations**: Supports `merge`, `ncells`, `dilate`, `erode`, `smooth`, `removeislands`, and ROI operations (`roigenerate`, `roiapply`, `roi<op>`).
- **Interfaces**: CLI entry points in `pyproject.toml` (e.g., `emi-generate-mesh`, `emi-process-image-data`).

## Common Tasks

### Testing & Linting
- **Run all tests**: `pytest`
- **Run specific test**: `pytest tests/test_process_image_data.py`
- **Lint code**: `ruff check .`

### Pipeline Execution
- **Run full pipeline**: `snakemake --configfile config_files/your_config.yml --use-conda --cores 8`

### CLI Tools
Available commands (defined in `pyproject.toml`):
- `emi-download-data`: Downloads segmentation data.
- `emi-process-image-data`: Applies processing operations.
- `emi-extract-surfaces`: Extracts cell surfaces.
- `emi-generate-mesh`: Generates the final volumetric mesh.
- `emi-evaluate-mesh`: Evaluates mesh quality.
- `emi-plot-analysis`: Plots analysis results.
- `emi-generate-screenshot`: Generates screenshots.

## Key Implementation Notes
- **Authentication**: If using `caveclient`, authentication must be set up via `CAVEclient().auth.get_new_token()`.
- **Dependencies**: Relies on `fTetWild` (via `pytetwild`), `nbmorph` (for morphological operations), and `cloud-volume`.
- **Output Structure**:
  - `raw/`: Downloaded segmentation (`.vti`).
  - `processed/`: Processed image (`.vti`) and `imagestatistics.yml`.
  - `surfaces/`: Extracted cell surfaces (`.ply`).
  - `meshes/`: Volumetric meshes (`.xdmf`).
