# Plan: Add CAVEclient neuron/cell download support

## Context
The goal is to extend `emimesh` to allow downloading specific cell types (e.g., neurons, astrocytes) using the MICrONS data infrastructure via `CAVEclient`. This will complement the existing `cloud-volume` and `webknossos` downloaders.

## Proposed Changes
1.  **Dependencies**: Add `caveclient` to `pyproject.toml`.
2.  **Implementation**: Create a new module `src/emimesh/clients/cave_download.py` that utilizes `CAVEclient` to query the `aibs_metamodel_celltypes_v661` table and retrieve cell segment IDs.
3.  **CLI Entry Point**: Register a new command `emi-download-cell` in `pyproject.toml` pointing to the new module.
4.  **Integration**: Ensure the new client follows the existing pattern established in `src/emimesh/clients/download_data.py`.

## Critical Files
- `pyproject.toml` (Add dependency and entry point)
- `src/emimesh/clients/cave_download.py` (New file)

## Verification
- Run `uv pip install .` to update dependencies.
- Verify `emi-download-cell --help` works.
- Perform a dry-run or mock test downloading a single cell ID to confirm `CAVEclient` integration.
