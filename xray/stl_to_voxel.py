import argparse
import math
import multiprocessing as mp
import os
import sys
from glob import glob
from itertools import repeat
from p_tqdm import p_map
from functools import partial
import numpy as np

from xray.util import dir_path, read_stl, get_voxels


def stl_to_voxel(stl_file, args):
    voxel_file = os.path.join(
        os.path.join(args.output, f"{os.path.split(stl_file)[1][:-4]}_{args.vres}_{str(args.rotate_mesh).lower()}"))

    mesh = read_stl(stl_file)
    # TODO: scale the mesh so that the image has reasonable dimension (xray/perimeter.py?)
    # Random rotation over x and y axis (rotation over z axis is done at image level)
    if args.rotate_mesh:
        mesh.rotate([0.5, 0., 0.0], math.radians(np.random.randint(30, 210)))
        mesh.rotate([0., 0.5, 0.0], math.radians(np.random.randint(30, 210)))
    voxels, _ = get_voxels(mesh, args.vres)
    np.save(voxel_file, voxels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert STL files to voxels')
    parser.add_argument('--input', type=dir_path, required=True, action='store',
                        help="Input directory containing .stl files.")
    parser.add_argument('--vres', type=int, default=20, action='store', help="Voxel resolution (default: 20)")
    parser.add_argument('--rotate-mesh', default=False, action='store_true', help="Rotate mesh")
    parser.add_argument('--output', type=str, default="./output", action='store',
                        help="Output directory (default: output)")
    parser.add_argument('--nproc', type=int, default=4, action='store', help="Number of CPUs to use. (default: 12)")
    args = parser.parse_args()

    # Load .stl files
    stl_files = glob(os.path.join(args.input, "*.stl"))
    if len(stl_files) == 0:
        print("ERROR: No .STL files found.")
        sys.exit(1)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # stl_to_voxel(stl_files[0], args)

    print("LOG: Converting .stl files...")
    pool = mp.Pool(args.nproc)
    # images = pool.starmap(stl_to_voxel, zip(stl_files, repeat(args)))
    p_map(partial(stl_to_voxel, args=args), stl_files)
    pool.close()
    print(f"Voxels files saved to {args.output}")
