from caveclient import CAVEclient
import pandas as pd

def get_cell_types(table_name="aibs_metamodel_celltypes_v661"):
    """
    Query cell types using CAVEclient and add is_neuron column.
    """
    # Initialize the client (using standard MICrONS datastack)
    client = CAVEclient("minnie65_public")

    # Query the specified table
    try:
        df = client.materialize.query_table(table_name, select_columns=['id', 'pt_position', 'classification_system', 'cell_type'])

        # Create a new column 'cell_type_basic'
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

if __name__ == "__main__":
    # Example usage
    df = get_cell_types()
    if df is not None:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        print(df.head())
        # Example: count by cell type
        print(df["cell_type"].value_counts())


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
