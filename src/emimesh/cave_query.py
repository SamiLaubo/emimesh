import os

import numpy as np
import pandas as pd
from pathlib import Path
from caveclient import CAVEclient
from cloudvolume import CloudVolume, Bbox

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

def filter_df_by_bbox(df, cloud_path="precomputed://gs://iarpa_microns/minnie/minnie65/seg"):
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
    print(f"Cloud volume bounding box: {bbox}")
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
    print("Downloading skeleton for cell ID:", cell_id)
    dict = client.skeleton.get_skeleton(cell_id, output_format='dict') 
    vertices = dict["vertices"]

    # print resolution of the client and bounds
    print(f"Client resolution: {client.info.viewer_resolution()}")
    print(f"Client bounds: {client.info.bounds()}")

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
        cloud_path="precomputed://gs://iarpa_microns/minnie/minnie65/seg"
    ):

    # res?
    # bbox_skeleton = skeleton_bounding_box(cell_id, padding_voxels=padding_voxels)

    # Bbox max size around cell center
    # Org res
    bbox_centered_max = Bbox(cell_center - max_size_voxels // 2, cell_center + max_size_voxels // 2, unit="vx")

    # Bbox for segmentation data
    # CV seg res
    bbox_seg = CloudVolume(cloud_path, use_https=True, mip=0, bounded=True).bounds
    
    # Fix resolution in x and y directions for segmentation data
    # CV res
    bbox_seg.minpt[:2] *= 2
    bbox_seg.maxpt[:2] *= 2

    # Intersection of the three bounding boxes to ensure we stay within the segmentation data bounds
    # bbox = Bbox.intersection(bbox_skeleton, bbox_centered_max)
    bbox = bbox_centered_max
    bbox = Bbox.intersection(bbox, bbox_seg)

    print(f"\nCalculated Bounding Box (mip=0 voxels): {bbox}")
    print(f"Bounding Box Volume Size (x,y,z): {bbox.size3()}")

    return bbox


def get_cell_info(
        table_name="aibs_metamodel_celltypes_v661",
        cloud_path="precomputed://gs://iarpa_microns/minnie/minnie65/seg",
        mip=0,
        cell_type="neuron", 
        idx=0, 
        padding_voxels=100, 
        max_size_voxels=1000,
    ):
    """
    Get information about a specific cell from the specified table.
    """
    # Get the cell type table
    # if os.path.exists(".cache/filtered_cell_table.csv"):
        # df = pd.read_csv(".cache/filtered_cell_table.csv")
    # if os.path.exists(".cache/filtered_cell_table.hdf5"):
        # df = pd.read_hdf(".cache/filtered_cell_table.hdf5", key="cell_table")
    # if os.path.exists(".cache/filtered_cell_table.pickle"):
        # df = pd.read_pickle(".cache/filtered_cell_table.pickle")
    # else:
        # Filter the dataframe for out of bounds of cloud volume bounds
    df = get_cell_type_table(table_name)
    df = filter_df_by_bbox(df, cloud_path)
        
        # Path(".cache").mkdir(parents=True, exist_ok=True)
        # df.to_pickle(".cache/filtered_cell_table.pickle")
        # df.to_csv(".cache/filtered_cell_table.csv", index=False)

    # Extract one cell
    cell_df = df[df['cell_type_basic'] == cell_type]
    if idx >= len(cell_df):
        raise IndexError(f"Index {idx} is out of bounds for cell type '{cell_type}' with {len(cell_df)} entries.")

    # Get info for one cell
    cell_id = cell_df["pt_root_id"].values[idx]

    print("Cell ID: ", cell_id)

    bbox = get_valid_bbox(
        cell_id, 
        cell_df["pt_position"].values[idx], 
        padding_voxels=padding_voxels, 
        max_size_voxels=max_size_voxels,
        cloud_path=cloud_path
    )


    # bbox = skeleton_bounding_box(cell_id, padding_voxels=padding_voxels, max_size_voxels=max_size_voxels)

    # Convert bounds to correct mip
    cv = CloudVolume(cloud_path, use_https=True, mip=0, bounded=True)
    print(bbox)
    bbox = cv.bbox_to_mip(bbox, mip=0, to_mip=mip)
    print(bbox)

    # Divide x and y by 2, not in z direction
    # bbox = Bbox((bbox.minpt[0] // 2, bbox.minpt[1] // 2, bbox.minpt[2]), (bbox.maxpt[0] // 2, bbox.maxpt[1] // 2, bbox.maxpt[2]), unit="vx")

    
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
