# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
EMI-Mesh provides an automated pipeline for generating high-quality tetrahedral meshes of brain tissue (extracellular and intracellular spaces) from imaging data. It takes segmented image data as input, processes it, extracts surfaces, and generates volumetric meshes using [fTetWild](https://github.com/wildmeshing/fTetWild).

### Core Architecture
- **Pipeline Orchestration**: The pipeline is driven by `Snakemake` (see `readme.md` for execution commands).
- **Processing Engine**: The `src/emimesh/` directory contains the core logic for:
  - Downloading data (via `cloud-volume` or `webknossos`)
  - Image processing (smoothing, dilating, eroding, merging labels, ROI operations)
  - Surface extraction
  - Volumetric meshing
- **Interfaces**: CLI entry points for specific pipeline stages are defined in `pyproject.toml` (e.g., `emi-generate-mesh`, `emi-process-image-data`).

## Common Tasks

### Testing
This project uses `pytest`. Run tests with:
```bash
pytest
```
To run a single test file:
```bash
pytest tests/test_process_image_data.py
```

### Linting
This project uses `ruff`.
```bash
ruff check .
```

### Pipeline Execution
As described in `readme.md`, use Snakemake to execute the full pipeline:
```bash
snakemake --configfile configfiles/your_config.yml --use-conda --cores 8
```
Configuration files reside in `configfiles/` (or `config_files/` as referenced in the README).

## Key Implementation Notes
- **Configuration**: The workflow is heavily data-driven via YAML files. Changes to processing logic are often made by modifying the operations list in the `processing` block of these YAMLs.
- **Dependencies**: Key dependencies include `fTetWild`, `nbmorph` (for morphological operations), and `cloud-volume`.
