import argparse
import math
import multiprocessing as mp
import os
import sys
from functools import partial
from glob import glob

import numpy as np
from p_tqdm import p_map

from xray.util import dir_path, read_stl, get_voxels, crop_model


def stl_to_voxel(stl_file, args):
    voxel_file = os.path.join(
        os.path.join(args.output, f"{os.path.split(stl_file)[1][:-4]}_{args.scale}_{str(args.rotate_mesh).lower()}"))

    if os.path.isfile(voxel_file + ".npy"):
        return
    mesh = read_stl(stl_file)
    # Random rotation over x and y axis (rotation over z axis is done at image level)
    if args.rotate_mesh:
        max_deviation = 20
        xrot = np.random.choice(range(-max_deviation, max_deviation, 1))
        yrot = np.random.choice(range(-max_deviation, max_deviation, 1))
        zrot = np.random.choice(range(0, 361, 90)) + np.random.choice(range(-max_deviation, max_deviation, 1))
        print(f"{stl_file} is rotated by ({xrot}, {yrot}, {zrot})")
        mesh.rotate([0.5, 0., 0.], math.radians(xrot))
        mesh.rotate([0., 0.5, 0.], math.radians(yrot))
        mesh.rotate([0., 0., 0.5], math.radians(zrot))
    voxels, _ = get_voxels(mesh, args.scale)
    voxels = crop_model(voxels)
    np.save(voxel_file, voxels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert STL files to voxels')
    parser.add_argument('--input', type=dir_path, required=True, action='store',
                        help="Input directory containing .stl files.")
    parser.add_argument('--scale', type=float, default=2.0, action='store', help="Scale of the objects (default: 2.0)")
    parser.add_argument('--rotate-mesh', default=False, action='store_true', help="Rotate mesh")
    parser.add_argument('--output', type=str, default="./output", action='store',
                        help="Output directory (default: output)")
    parser.add_argument('--nproc', type=int, default=4, action='store', help="Number of CPUs to use. (default: 4)")
    args = parser.parse_args()

    # Load .stl files
    stl_files = glob(os.path.join(args.input, "*.stl"))
    if len(stl_files) == 0:
        print("ERROR: No .STL files found.")
        sys.exit(1)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    print("LOG: Converting .stl files...")
    pool = mp.Pool(args.nproc)
    p_map(partial(stl_to_voxel, args=args), stl_files)
    pool.close()
    print(f"Voxels files saved to {args.output}")
