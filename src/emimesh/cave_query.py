import json
import os

import numpy as np
import pandas as pd
from pathlib import Path
from caveclient import CAVEclient
from cloudvolume import CloudVolume, Bbox
import urllib

def get_cell_type_table(table_name="aibs_metamodel_celltypes_v661"):
    """
    Query cell types using CAVEclient and add is_neuron column.
    """
    # Initialize the client (using standard MICrONS datastack)
    client = CAVEclient("minnie65_public")

    # Query the specified table
    try:
        df = client.materialize.query_table(
            table_name, 
            select_columns=['pt_root_id', 'pt_position', 'classification_system', 'cell_type']
        )

        # Create a new column with neuron, astrocyte, ...
        def classify(row):
            if row['classification_system'] != 'nonneuron':
                return 'neuron'
            else:
                return row['cell_type']

        df['cell_type_basic'] = df.apply(classify, axis=1)

        return df.drop_duplicates('pt_root_id', keep=False)
    except Exception as e:
        print(f"Error querying table {table_name}: {e}")
        return None

def filter_df_by_bbox(df, cloud_path="precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"):
    """
    Filters a CAVEclient dataframe, removing rows where 'pt_position'
    is outside the specified bounding box of the cloud volume.
    Because segmentation data is a subset of the full volume.

    Args:
        df (pd.DataFrame): The dataframe containing a 'pt_position' column.
        cloud_path (str): The path to the cloud volume.

    Returns:
        pd.DataFrame: A new dataframe containing only the rows within the bounding box.
    """
    if 'pt_position' not in df.columns:
        raise ValueError("Dataframe must contain a 'pt_position' column.")

    # Convert the list of coordinates in 'pt_position' into a 2D numpy array
    # Shape will be (N, 3) where N is the number of rows
    coords = np.array(df['pt_position'].tolist())

    # Extract minimum and maximum points from the Bbox
    bbox = CloudVolume(cloud_path, use_https=True, mip=0, bounded=True).bounds
    min_pt = bbox.minpt + 1000  # Add padding to avoid edge cases
    max_pt = bbox.maxpt - 1000

    # Correct for different subvolume and reslution
    min_pt[:2] *= 2
    max_pt[:2] *= 2

    # Create boolean masks for X, Y, and Z axes
    in_x = (coords[:, 0] >= min_pt[0]) & (coords[:, 0] <= max_pt[0])
    in_y = (coords[:, 1] >= min_pt[1]) & (coords[:, 1] <= max_pt[1])
    in_z = (coords[:, 2] >= min_pt[2]) & (coords[:, 2] <= max_pt[2])

    # Combine masks to find points that satisfy all three conditions
    inside_mask = in_x & in_y & in_z

    # Return the filtered dataframe
    return df[inside_mask].reset_index(drop=True)

def filter_df_by_proofread_cells(df, proofread_status=["axon_fully_extended"], cloud_path="minnie65_public"):
    """
    Filters the dataframe:
    - Neurons are kept only if they are found in the proofreading_status_and_strategy table.
    - Non-neurons are always kept.

    Args:
        df (pd.DataFrame): The dataframe to filter. Must contain 'pt_root_id' and 'cell_type_basic'.
        proofread_status (list): List of statuses to keep (e.g., ["axon_fully_extended"]).
        cloud_path (str): CAVE client path.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    if 'pt_root_id' not in df.columns:
        raise ValueError("Dataframe must contain a 'pt_root_id' column.")
    if 'cell_type_basic' not in df.columns:
        raise ValueError("Dataframe must contain a 'cell_type_basic' column.")

    client = CAVEclient(cloud_path)

    try:
        # Query the proofreading status table for matching root IDs
        proofread_df = client.materialize.query_table(
            "proofreading_status_and_strategy",
            select_columns=['pt_root_id'],
            filter_in_dict={"strategy_axon": proofread_status}
        )
        proofread_root_ids = set(proofread_df['pt_root_id'].tolist())
    except Exception as e:
        print(f"Error querying proofreading table: {e}")
        return df

    # Keep if (not a neuron) OR (is a neuron AND is proofread)
    is_neuron = (df['cell_type_basic'] == 'neuron')
    is_proofread = df['pt_root_id'].isin(proofread_root_ids)

    mask = (~is_neuron) | (is_neuron & is_proofread)

    return df[mask].reset_index(drop=True)
    
def skeleton_bounding_box(cell_id, cloud_path="minnie65_public", padding_voxels=20):
    """
    Download the skeleton for a given cell ID and create a bounding box.
    See: https://tutorial.microns-explorer.org/quickstart_notebooks/07-cave-download-skeleton.html
    """

    # Download the skeleton for the given cell ID
    client = CAVEclient(cloud_path)
    # print("Downloading skeleton for cell ID:", cell_id)
    dict = client.skeleton.get_skeleton(cell_id, output_format='dict') 
    vertices = dict["vertices"]

    # nm to voxel
    resolution = np.array(client.info.viewer_resolution())
    vertices_voxels = vertices / resolution
    
    # Find the minimum and maximum coordinates across all X, Y, Z vertices
    min_coords = np.floor(np.min(vertices_voxels, axis=0)).astype(int)
    max_coords = np.ceil(np.max(vertices_voxels, axis=0)).astype(int)
    
    # Add padding to capture the whole cell volume safely
    min_coords -= padding_voxels
    max_coords += padding_voxels

    # Create the Bounding Box
    bbox = Bbox(min_coords, max_coords, unit="vx")
    
    return bbox

def get_valid_bbox(
        cell_id, 
        cell_center, 
        padding_voxels=20, 
        max_size_nm=100,
        cloud_path="precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300",
        verbose=False
    ):

    # Org res (4,4,40) nm
    bbox_skeleton = skeleton_bounding_box(cell_id, padding_voxels=padding_voxels)
    if verbose:
        print(f"Skeleton bounding box volume size (x,y,z): {bbox_skeleton.size3()} or {bbox_skeleton.size3() * np.array([4,4,40])} nm^3. Not used!")

    # Bbox max size around cell center
    # Org res (4,4,40) nm for minnie65
    # max size nm to voxel conversion
    max_size_voxels = np.array([max_size_nm, max_size_nm, max_size_nm]) / np.array([4,4,40])
    bbox_centered_max = Bbox(cell_center - max_size_voxels // 2, cell_center + max_size_voxels // 2, unit="vx")

    # Bbox for segmentation data
    # CV seg res (8,8,40) nm for
    bbox_seg = CloudVolume(cloud_path, use_https=True, mip=0, bounded=True).bounds
    
    # Fix resolution in x and y directions for segmentation data
    # CV res (4,4,40) nm
    bbox_seg.minpt[:2] *= 2
    bbox_seg.maxpt[:2] *= 2

    # Intersection of the three bounding boxes to ensure we stay within the segmentation data bounds
    # Print volume loss from intersection
    if verbose:
        print("Starting from bbox from skeleton with volume size (x,y,z):", bbox_skeleton.size3())
    
    bbox = Bbox.intersection(bbox_skeleton, bbox_centered_max)
    if verbose:
        print(f"Volume after intersection with centered max bbox: {bbox.size3()}")
    
    bbox = Bbox.intersection(bbox, bbox_seg)
    if verbose:
        print(f"Final volume after intersection with segmentation bbox: {bbox.size3()}")

    print(f"Bounding box {bbox}\n With volume {bbox.size3()} ({bbox.size3() * np.array([4,4,40]) * 1e-3} mum^3)")

    return bbox

def cell_url(x, y, z, cell_id, output_folder=None):
    """Generate url to neuroglancer for the specific neuron

    Args:
        x (int): x in voxel
        y (int): y in voxel
        z (int): z in voxel
        cell_id (uint64): cell_id
        output_folder (str, optional): Save url to file in folder. Defaults to None.
    """
    state_json = {
        "dimensions": {
            "x": [4e-09,"m"],
            "y": [4e-09,"m"],
            "z": [4e-08,"m"]
        },
        "position": [int(x), int(y), int(z)],
        "crossSectionScale": 7.0,
        "projectionOrientation": [-0.9,0.1,0.3,-0.1],
        "projectionScale": 40000,
        "layers": [
            {
            "type": "image",
            "source": {
                "url": "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em",
                "subsources": {"default": True},
                "enableDefaultSubsources": False
            },
            "tab": "source",
            "shaderControls": {"normalized": {"range": [86,172]}},
            "name": "img65"
            },
            {
            "type": "segmentation",
            "source": "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300",
            "tab": "segments",
            "annotationColor": "#8f8f8a",
            "selectedAlpha": 0.41,
            "notSelectedAlpha": 0.06,
            "segments": [f"{cell_id}"],
            "segmentQuery": f"{cell_id}",
            "colorSeed": 1689220695,
            "name": "seg65"
            },
        ],
        "showAxisLines": False,
        "showSlices": False,
        "selectedLayer": {
            "visible": True,
            "layer": "seg65"
        },
        "layout": {
            "type": "xy-3d",
            "orthographicProjection": True
        }
    }

    json_str = json.dumps(state_json, separators=(',', ':'))
    encoded_json = urllib.parse.quote(json_str)
    full_url = f"https://neuroglancer-demo.appspot.com/#!{encoded_json}"
    api_url = f"https://da.gd/s?url={urllib.parse.quote(full_url)}"
    
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            url = response.read().decode('utf-8').strip()
            
    except Exception as e:
        print(f"Shortener failed: {e}")
        url = full_url  # Fallback to the long URL if the API fails

    print(f"Neuroglancer link for cell {cell_id}: {url}")

    if output_folder is not None:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_folder, f"neuroglancer_link.txt")
        with open(output_file, 'w') as f:
            f.write(f"Cell ID: {cell_id}\n")
            f.write(f"Neuroglancer link: {url}\n")

def get_cell_info(
        table_name="aibs_metamodel_celltypes_v661",
        cloud_path="precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300",
        cell_type="neuron", 
        cell_neuron_type="",
        idx=0, 
        padding_voxels=100, 
        max_size_nm=100_000,
        output="data.xdmf"
    ):
    """
    Get information about a specific cell from the specified table.
    """
    # Get the cell type table
    if os.path.exists(".cache/filtered_cell_table.pickle"):
        df = pd.read_pickle(".cache/filtered_cell_table.pickle")
    else:
        # Filter the dataframe for out of bounds of cloud volume bounds
        df = get_cell_type_table(table_name)
        df = filter_df_by_bbox(df, cloud_path)
        
        Path(".cache").mkdir(parents=True, exist_ok=True)
        df.to_pickle(".cache/filtered_cell_table.pickle")

    # Extract one cell
    if cell_neuron_type != "":
        print(f"Filtering for specific neuron type: {cell_neuron_type}")
        cell_df = df[df['cell_type'] == cell_neuron_type]
    else:
        print(f"Filtering for cell type: {cell_type}")
        cell_df = df[df['cell_type_basic'] == cell_type]
    if idx >= len(cell_df):
        raise IndexError(f"Index {idx} is out of bounds for cell type '{cell_type}' with {len(cell_df)} entries.")

    # Get info for one cell
    cell_id = cell_df["pt_root_id"].values[idx].astype(np.uint64)
    cell_type = cell_df["cell_type"].values[idx]

    print(f"Cell ID: {cell_id} with type {cell_type} at index {idx} from table '{table_name}'.")
    print(f"Cell position (pt_position): {cell_df['pt_position'].values[idx]}")

    # Print and save neuroglancer link for the cell
    x, y, z = cell_df['pt_position'].values[idx]
    cell_url(x, y, z, cell_id, Path(output).parent)

    # bbox in (4,4,40) nm resolution
    bbox = get_valid_bbox(
        cell_id, 
        cell_df["pt_position"].values[idx], 
        padding_voxels=padding_voxels, 
        max_size_nm=max_size_nm,
        cloud_path=cloud_path
    )

    print(f'Final bounding box coordinates: {bbox.minpt} to {bbox.maxpt} in voxels.')
    
    return cell_id, bbox


