import os
import yaml
import pandas as pd
import argparse
from pathlib import Path
from emimesh.cave_query import get_cell_type_table

def process_folder(folder, cell_type_lookup):
    """
    Processes a single result folder to create cell_type_mapping.yml.
    """
    stats_path = folder / "processed" / "imagestatistic.yml"

    if not stats_path.exists():
        print(f"Skipping {folder.name}: {stats_path} not found.")
        return False

    print(f"Processing {folder.name}...")

    # Read the mapping from imagestatistic.yml
    with open(stats_path, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in {stats_path}: {e}")
            return False

    if not data or 'mapping' not in data:
        print(f"Skipping {folder.name}: No 'mapping' found in {stats_path}.")
        return False

    original_mapping = data['mapping']
    # original_mapping is { original_cell_id: mesh_label }

    # Create the mesh_label -> cell_type mapping
    label_to_type = {}
    for cell_id, mesh_label in original_mapping.items():
        if mesh_label == 0:
            label_to_type[0] = "extracellular"
            continue

        # Convert cell_id to string for lookup
        sid = str(cell_id)
        cell_type = cell_type_lookup.get(sid, "unknown")
        label_to_type[mesh_label] = cell_type

    # Save the mapping to cell_type_mapping.yml
    output_path = folder / "processed" / "cell_type_mapping.yml"
    with open(output_path, 'w') as f:
        yaml.dump(label_to_type, f, default_flow_style=False)

    print(f"  Saved mapping to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", 
        type=str,
        help="Folder to process.")
    args = parser.parse_args()

    # 1. Load the global cell type table from CAVE or cache
    cache_path = Path(".cache/filtered_cell_table.pickle")
    if cache_path.exists():
        print(f"Loading cell type table from cache: {cache_path}")
        cell_df = pd.read_pickle(cache_path)
    else:
        print("Fetching cell type table from CAVE... This may take a moment.")
        cell_df = get_cell_type_table()

        if cell_df is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cell_df.to_pickle(cache_path)
            print(f"Saved cell type table to cache: {cache_path}")

    if cell_df is None:
        print("Error: Could not retrieve cell type table.")
        return

    # Set pt_root_id as index for fast lookup
    cell_df['pt_root_id'] = cell_df['pt_root_id'].astype(str)
    cell_type_lookup = cell_df.set_index('pt_root_id')['cell_type_basic'].to_dict()
    print(f"Loaded {len(cell_type_lookup)} cell type mappings.")

    target_path = Path(args.folder)
    if target_path.is_dir():
        # If we are pointing exactly at the results directory, process all its subfolders
        if target_path.name == "results" and target_path.parent == Path("."):
            for folder in target_path.iterdir():
                if folder.is_dir():
                    process_folder(folder, cell_type_lookup)
        else:
            # Otherwise, treat the provided path as a single result folder to process
            process_folder(target_path, cell_type_lookup)
    else:
        print(f"Error: {args.folder} is not a valid directory.")

    print("\nPost-processing complete.")

if __name__ == "__main__":
    main()
