import os
import yaml
import pandas as pd
import argparse
from pathlib import Path
from emimesh.cave_query import get_cell_type_table

def process_folder(stats_path, cell_type_lookup, output_path):
    """
    Processes a single result folder to create cell_type_mapping.yml.
    """

    # Read the mapping from imagestatistic.yml
    with open(stats_path, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in {stats_path}: {e}")
            return False

    if not data or 'mapping' not in data:
        print(f"Skipping {stats_path}: No 'mapping' found in {stats_path}.")
        return False

    original_mapping = data['mapping']

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

    # Save the mapping
    with open(output_path, 'w') as f:
        yaml.dump(label_to_type, f, default_flow_style=False)

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagestatistic_path",
        type=str,
        help="Path to the imagestatistic.yml file.")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output file.")
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

    process_folder(Path(args.imagestatistic_path), cell_type_lookup, output_path=Path(args.output_path))

    print("\nPost-processing complete.")

if __name__ == "__main__":
    main()
