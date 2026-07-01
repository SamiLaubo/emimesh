import json
import os

import numpy as np
import pandas as pd
from pathlib import Path
from caveclient import CAVEclient
from cloudvolume import CloudVolume, Bbox
import urllib

from emimesh.utils import np2pv
from emimesh.download_data import download_webknossos, download_cloudvolume

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
            # select_column_map={
            #     table_name: ['pt_root_id', 'pt_position', 'classification_system', 'cell_type']
            # }
        )

        # Create a new column with neuron, astrocyte, ...
        def classify(row):
            if row['classification_system'] != 'nonneuron':
                return 'neuron'
            else:
                return row['cell_type']

        df['cell_type_basic'] = df.apply(classify, axis=1)

        return df
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
    min_pt = bbox.minpt
    max_pt = bbox.maxpt

    # Correct for different subvolume and reslution
    min_pt[:2] *= 2
    max_pt[:2] *= 2
    
    # Create boolean masks for X, Y, and Z axes
    in_x = (coords[:, 0] >= min_pt[0]) & (coords[:, 0] <= max_pt[0])
    in_y = (coords[:, 1] >= min_pt[1]) & (coords[:, 1] <= max_pt[1])
    in_z = (coords[:, 2] >= min_pt[2]) & (coords[:, 2] <= max_pt[2])
    
    # Combine masks to find points that satisfy all three conditions
    inside_mask = in_x & in_y & in_z
    
    # Return the filtered dataframe, resetting the index for cleanliness
    return df[inside_mask].reset_index(drop=True)
    
def skeleton_bounding_box(cell_id, cloud_path="minnie65_public", padding_voxels=10):
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
        padding_voxels=10, 
        max_size_voxels=1000,
        cloud_path="precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"
    ):

    # res?
    bbox_skeleton = skeleton_bounding_box(cell_id, padding_voxels=padding_voxels)
    # print(f"Skeleton bounding box volume size (x,y,z): {bbox_skeleton.size3()}")

    # Bbox max size around cell center
    # Org res (4,4,40) nm for minnie65
    bbox_centered_max = Bbox(cell_center - max_size_voxels // 2, cell_center + max_size_voxels // 2, unit="vx")
    # print(f"Centered max bounding box volume size: {bbox_centered_max.size3()}")

    # Bbox for segmentation data
    # CV seg res (8,8,40) nm for
    bbox_seg = CloudVolume(cloud_path, use_https=True, mip=0, bounded=True).bounds
    
    # Fix resolution in x and y directions for segmentation data
    # CV res (4,4,40) nm
    bbox_seg.minpt[:2] *= 2
    bbox_seg.maxpt[:2] *= 2

    # Intersection of the three bounding boxes to ensure we stay within the segmentation data bounds
    # Print volume loss from intersection
    print("Starting from bbox from skeleton with volume size (x,y,z):", bbox_skeleton.size3())
    bbox = Bbox.intersection(bbox_skeleton, bbox_centered_max)
    print(f"Volume after intersection with centered max bbox: {bbox.size3()}")
    
    bbox = Bbox.intersection(bbox, bbox_seg)
    print(f"Final volume after intersection with segmentation bbox: {bbox.size3()}")

    # Make each side divisible by 8 for parallel downloading
    bbox = Bbox((np.ceil(bbox.minpt / 10) * 10).astype(np.int32), (np.floor(bbox.maxpt / 10) * 10).astype(np.int32), unit="vx")

    print(f"Final volume after making it divisible by 8: {bbox.size3()} ({bbox.size3() * np.array([4,4,40]) * 1e-3} mum^3)")

    return bbox

# def convert_label_by_coords():

def get_cell_url(x, y, z, cell_id):
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

    # client = CAVEclient('minnie65_public')
    # state_id = client.state.upload_state_json(state_json)
    # # return client.state.build_neuroglancer_url(state_id, 'https://ngl.microns-explorer.org')
    # return client.state.build_neuroglancer_url(state_id, 'https://neuroglancer-demo.appspot.com')

    # 1. Convert the Python dictionary back to a minified JSON string
    json_str = json.dumps(state_json, separators=(',', ':'))
    
    # 2. Safely URL-encode the JSON string
    encoded_json = urllib.parse.quote(json_str)
    
    # 3. Attach it to the public Google viewer
    full_url = f"https://neuroglancer-demo.appspot.com/#!{encoded_json}"
    
    api_url = f"https://da.gd/s?url={urllib.parse.quote(full_url)}"
    
    try:
        # User-Agent added to prevent standard bot blocking
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            # .strip() removes the invisible newline character at the end of the response
            return response.read().decode('utf-8').strip()
            
    except Exception as e:
        print(f"Shortener failed: {e}")
        return full_url  # Fallback to the long URL if the API fails

def get_cell_info(
        table_name="aibs_metamodel_celltypes_v661",
        cloud_path="precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300",
        cell_type="neuron", 
        idx=0, 
        padding_voxels=100, 
        max_size_voxels=1000,
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
    cell_df = df[df['cell_type_basic'] == cell_type]
    if idx >= len(cell_df):
        raise IndexError(f"Index {idx} is out of bounds for cell type '{cell_type}' with {len(cell_df)} entries.")

    # Get info for one cell
    cell_id = cell_df["pt_root_id"].values[idx].astype(np.uint64)

    print(f"Cell ID: {cell_id} with tyepe '{cell_type}' at index {idx} from table '{table_name}'.")
    print(f"Cell position (pt_position): {cell_df['pt_position'].values[idx]}")

    x, y, z = cell_df['pt_position'].values[idx]
    print(f"Neuroglancer link: {get_cell_url(x, y, z, cell_id)}")
    # print(f"Neuroglancer link:\n-----------------------------------\n\
    # https://ngl.microns-explorer.org/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B{x}%2C{y}%2C{z}%5D%2C%22crossSectionScale%22:7.096617776349856%2C%22projectionOrientation%22:%5B-0.9350257515907288%2C0.13611161708831787%2C0.29464977979660034%2C-0.14276540279388428%5D%2C%22projectionScale%22:489587.6696286937%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em%22%2C%22subsources%22:%7B%22default%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22shaderControls%22:%7B%22normalized%22:%7B%22range%22:%5B86%2C172%5D%7D%7D%2C%22name%22:%22img65%22%7D%2C%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie35/em%22%2C%22subsources%22:%7B%22default%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22shaderControls%22:%7B%22normalized%22:%7B%22range%22:%5B112%2C172%5D%7D%7D%2C%22name%22:%22img35%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300%22%2C%22tab%22:%22segments%22%2C%22annotationColor%22:%22#8f8f8a%22%2C%22selectedAlpha%22:0.41%2C%22notSelectedAlpha%22:0.06%2C%22segments%22:%5B%22864691134064155671%22%2C%22864691136144674612%22%2C%22864691135307555142%22%2C%22864691135937286404%22%2C%22864691136812081779%22%2C%22864691135067270468%22%2C%22864691135346954143%22%2C%22864691135356428751%22%2C%22864691135375633481%22%2C%22864691135394307317%22%2C%22864691135465381701%22%2C%22864691135492697695%22%2C%22864691135591944203%22%2C%22864691135617551721%22%2C%22864691135655610562%22%2C%22864691135777697453%22%2C%22864691135778484669%22%2C%22864691135808982045%22%2C%22864691135851839687%22%2C%22864691135865240702%22%2C%22864691136040742142%22%2C%22864691136210344892%22%2C%22864691136210699964%22%2C%22864691136662432990%22%2C%22864691136674556295%22%2C%22864691136912943345%22%5D%2C%22segmentQuery%22:%22864691136662432990%2C%20864691136144674612%2C%20864691135465381701%2C%20864691135375633481%2C%20864691136210344892%2C%20864691135808982045%2C%20864691135067270468%2C%20864691136040742142%2C%20864691135778484669%2C%20864691135777697453%2C%20864691135394307317%2C%20864691135865240702%2C%20864691135655610562%2C%20864691136674556295%2C%20864691135591944203%2C%20864691136210699964%2C%20864691135492697695%2C%20864691135346954143%2C%20864691136912943345%2C%20864691135937286404%2C%20864691135617551721%2C%20864691136812081779%2C%20864691135851839687%2C%20864691135356428751%22%2C%22colorSeed%22:1689220695%2C%22name%22:%22seg65%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://iarpa_microns/minnie/minnie35/seg%22%2C%22tab%22:%22segments%22%2C%22annotationColor%22:%22#8a8a8a%22%2C%22segments%22:%5B%22864691137827278437%22%2C%22864691138020403235%22%2C%22864691138081021535%22%2C%22864691138134948293%22%2C%22864691138142870469%22%2C%22%21864691138153699060%22%2C%22864691138178964470%22%2C%22864691138345166401%22%5D%2C%22name%22:%22seg35%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22seg65%22%7D%2C%22layout%22:%7B%22type%22:%22xy-3d%22%2C%22orthographicProjection%22:true%7D%7D \
    # \n-----------------------------------\n")



    # bbox in (4,4,40) nm resolution
    bbox = get_valid_bbox(
        cell_id, 
        cell_df["pt_position"].values[idx], 
        padding_voxels=padding_voxels, 
        max_size_voxels=max_size_voxels,
        cloud_path=cloud_path
    )

    print(f'Final bounding box coordinates: {bbox.minpt} to {bbox.maxpt} in voxels.')
    
    return cell_id, bbox
    

if __name__ == "__main__":
    df = get_cell_type_table()
    print(df.shape)
    df = filter_df_by_bbox(df)
    print(df.shape)
    if df is not None:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        print(df.head())
        print(df["cell_type_basic"].value_counts())

    print(get_cell_info(table_name="aibs_metamodel_celltypes_v661", cell_type="neuron", idx=0, padding_voxels=100, max_size_voxels=1000), mip=2)


### Cell types. https://tutorial.microns-explorer.org/annotation-tables.html#cell-type-tables
# Cell Type	Subclass	    Description
# 23P	    Excitatory	    Layer 2/3 cells
# 4P	    Excitatory	    Layer 4 cells
# 5P-IT	    Excitatory	    Layer 5 intratelencephalic cells
# 5P-ET	    Excitatory	    Layer 5 extratelencephalic cells
# 5P-NP	    Excitatory	    Layer 5 near-projecting cells
# 6P-IT	    Excitatory	    Layer 6 intratelencephalic cells
# 6P-CT	    Excitatory	    Layer 6 corticothalamic cells
#
# BC	    Inhibitory	    Basket cell
# BPC	    Inhibitory	    Bipolar cell. In practice, this was used for all cells thought to be VIP cell, not only those with a bipolar dendrite
# MC	    Inhibitory	    Martinotti cell. In practice, this label was used for all inhibitory neurons that appeared to be Somatostatin cell, not only those with a Martinotti cell morphology
# Unsure	Inhibitory	    Unsure. In practice, this label also is used for all likely-inhibitory neurons that did not match other types
#
# OPC	    Non-neuronal	Oligodendrocyte precursor cell
# astrocyte	Non-neuronal	Astrocyte
# microglia	Non-neuronal	Microglia
# pericyte	Non-neuronal	Pericyte
# oligo	    Non-neuronal	Oligodendrocyte



# bodor_pt_target_proofread
# 201 - "Limited query to 1 rows
# Index(['id', 'created', 'valid', 'target_id', 'classification_system',
#        'cell_type', 'id_ref', 'created_ref', 'valid_ref', 'volume',
#        'pt_supervoxel_id', 'pt_root_id', 'pt_position', 'bb_start_position',
#        'bb_end_position'],
#       dtype='object')

# bodor_pt_cells
# 201 - "Limited query to 1 rows
# Index(['id', 'created', 'superceded_id', 'valid', 'classification_system',
#        'cell_type', 'pt_supervoxel_id', 'pt_root_id', 'pt_position'],
#       dtype='object')

# baylor_gnn_cell_type_fine_model_v2
# 201 - "Limited query to 1 rows
# Index(['id_ref', 'created_ref', 'valid_ref', 'volume', 'pt_supervoxel_id',
#        'pt_root_id', 'id', 'created', 'valid', 'target_id',
#        'classification_system', 'cell_type', 'pt_position',
#        'bb_start_position', 'bb_end_position'],
#       dtype='object')

# allen_column_mtypes_v2
# 201 - "Limited query to 1 rows
# Index(['id_ref', 'created_ref', 'valid_ref', 'volume', 'pt_supervoxel_id',
#        'pt_root_id', 'id', 'created', 'valid', 'target_id',
#        'classification_system', 'cell_type', 'pt_position',
#        'bb_start_position', 'bb_end_position'],
#       dtype='object')


# aibs_metamodel_mtypes_v661_v2
# 201 - "Limited query to 1 rows
# Index(['id', 'created', 'valid', 'target_id', 'classification_system',
#        'cell_type', 'id_ref', 'created_ref', 'valid_ref', 'volume',
#        'pt_supervoxel_id', 'pt_root_id', 'pt_position', 'bb_start_position',
#        'bb_end_position'],
#       dtype='object')

# allen_v1_column_types_slanted_ref
# 201 - "Limited query to 1 rows
# Index(['id', 'created', 'valid', 'target_id', 'classification_system',
#        'cell_type', 'id_ref', 'created_ref', 'valid_ref', 'volume',
#        'pt_supervoxel_id', 'pt_root_id', 'pt_position', 'bb_start_position',
#        'bb_end_position'],
#       dtype='object')

# nucleus_ref_neuron_svm
# Table Owner Notice on nucleus_ref_neuron_svm: Please cite https://doi.org/10.1101/2022.07.20.499976 when using this table., 201 - "Limited query to 1 rows
# Index(['id', 'created', 'valid', 'target_id', 'classification_system',
#        'cell_type', 'id_ref', 'created_ref', 'valid_ref', 'volume',
#        'pt_supervoxel_id', 'pt_root_id', 'pt_position', 'bb_start_position',
#        'bb_end_position'],
#       dtype='object')

# cell_type_multifeature_combo
# 201 - "Limited query to 1 rows
# Index(['id', 'created', 'valid', 'target_id', 'classification_system',
#        'cell_type', 'id_ref', 'created_ref', 'valid_ref', 'volume',
#        'pt_supervoxel_id', 'pt_root_id', 'pt_position', 'bb_start_position',
#        'bb_end_position'],
#       dtype='object')

# aibs_metamodel_celltypes_v661
# 201 - "Limited query to 1 rows
# Index(['id', 'created', 'valid', 'target_id', 'classification_system',
#        'cell_type', 'id_ref', 'created_ref', 'valid_ref', 'volume',
#        'pt_supervoxel_id', 'pt_root_id', 'pt_position', 'bb_start_position',
#        'bb_end_position'],
#       dtype='object')
