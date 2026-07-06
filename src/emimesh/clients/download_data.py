import argparse
from pathlib import Path

from fastremap import fastremap
from emimesh.utils import np2pv, remap_cloudvolume
from pathlib import Path
import argparse
from emimesh.download_data import download_webknossos, download_cloudvolume
from emimesh.cave_query import get_cell_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cloudpath",
        help="path to cloud data",
        type=str,
        default="precomputed://gs://iarpa_microns/minnie/minnie65/seg",
    )
    parser.add_argument("--mip", help="resolution (0 is highest)", type=int, default=0)
    parser.add_argument(
        "--position",
        help="point position in x-y-z integer pixel position \
              (can be copied from neuroglancer)",
        type=str,
    )
    parser.add_argument(
        "--size",
        help="cube side length of the volume to be downloaded (in nm)",
        type=str,
        default=1000,
    )
    parser.add_argument(
        "--output", help="output filename", type=str, default="data.xdmf"
    )
    parser.add_argument(
        "--cell_table_name",
        help="name of the table to query for cell types",
        type=str,
        default="aibs_metamodel_celltypes_v661",
    )
    parser.add_argument(
        "--cell_type",
        help="cell type to download (optional, will download a single cell of this type)\nAvailable types: neuron, astrocyte, microglia, oligo, pericyte, OPC",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--cell_neuron_type",
        help="specific neuron type to download (optional)",
        type=str,
        default="",
    )
    parser.add_argument(
        "--cell_idx",
        help="index of the cell type from table to download (optional) ",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cell_padding",
        help="padding around the cell skeleton bounding box in voxels (optional, default=100)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--cell_keep_surrounding",
        help="keep surrounding cells in the bounding box (optional, default=False)",
        type=str,
        default="False",
        # action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()

    if args.cell_type is not None:
        # Check if cloudvolume is available
        try:
            import cloudvolume
        except ImportError:
            raise ImportError(
                "cloudvolume is required for downloading specific cell types. Please install it with 'pip install cloud-volume'."
            )
        
        cell_id, bbox = get_cell_info(
            table_name=args.cell_table_name,
            cloud_path=args.cloudpath,
            cell_type=args.cell_type,
            cell_neuron_type=args.cell_neuron_type,
            idx=args.cell_idx,
            padding_voxels=args.cell_padding,
            max_size_nm=int(args.size),
            output=args.output
        )

        img, res = download_cloudvolume(args.cloudpath, args.mip, None, None, cell_id_bbox_surrounding=(cell_id, bbox, args.cell_keep_surrounding.lower() == "true"))

    else:
        position = args.position.split("-")
        try:
            size = [int(args.size)] * 3
        except ValueError:
            size = [int(s) for s in args.size.split("-")]

        try:
            img,res = download_cloudvolume(args.cloudpath, args.mip, position, size)
        except:
            img,res = download_webknossos(args.cloudpath, args.mip, position, size)
        
    print(res)

    data = np2pv(img, res)
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    data.save(args.output)


if __name__ == "__main__":
    main()