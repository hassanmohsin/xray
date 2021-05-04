import json
import math
import multiprocessing as mp
import os
from argparse import ArgumentParser
from functools import partial
from glob import glob
from time import time

import numpy as np
from p_tqdm import p_map

from xray.util import read_stl, get_voxels, crop_model


def stl_to_voxel(stl_file, args):
    voxel_file = os.path.join(
        os.path.join(args['voxel_dir'],
                     f"{os.path.split(stl_file)[1][:-4]}_{args['scale']}_{str(args['rotated']).lower()}"))

    if os.path.isfile(voxel_file + ".npy"):
        return
    mesh = read_stl(stl_file)
    # Random rotation over x and y axis (rotation over z axis is done at image level)
    if args['rotated']:
        xrot = np.random.normal(0., 5.0)
        yrot = np.random.normal(0., 5.0)
        zrot = np.random.choice([0, 90, 180, 270]) + np.random.normal(0., 5.0)
        print(f"{stl_file} is rotated by ({xrot}, {yrot}, {zrot})")
        rot_mat = np.matmul(
            mesh.rotation_matrix([1, 0, 0], math.radians(xrot)),
            np.matmul(
                mesh.rotation_matrix([0, 1, 0], math.radians(yrot)),
                mesh.rotation_matrix([0, 0, 1], math.radians(zrot))
            )
        )
        mesh.rotate_using_matrix(rot_mat)
    voxels, _ = get_voxels(mesh, args['scale'])
    voxels = crop_model(voxels)
    np.save(voxel_file, voxels)
    del voxels


if __name__ == '__main__':
    parser = ArgumentParser(description='Convert 3D models to false-color xray images')
    parser.add_argument('--input', type=str, required=True, action='store',
                        help="JSON input")
    args = parser.parse_args()
    tic = time()

    # Load arguments
    if not os.path.isfile(args.input):
        raise FileNotFoundError("Input {args.input} not found.")

    with open(args.input) as f:
        args = json.load(f)

    # Load .stl files
    if os.path.isdir(args['stl_dir']):
        stl_files = glob(os.path.join(args['stl_dir'], "*.stl"))
    else:
        raise NotADirectoryError(f"{args['stl_dir']} is not a directory/doesn't exist.")

    if len(stl_files) == 0:
        raise FileNotFoundError(f"ERROR: No .stl file found in {args['stl_dir']}.")

    if not os.path.isdir(args['voxel_dir']):
        os.makedirs(args['voxel_dir'])

    print("LOG: Converting .stl files...")
    pool = mp.Pool(mp.cpu_count() if args['nproc'] == -1 else min(mp.cpu_count(), args['nproc']))
    p_map(partial(stl_to_voxel, args=args), stl_files)
    pool.close()
    print(f"Voxels files saved to {args['voxel_dir']}")
    toc = time() - tic
    print(f"Execution time: {toc} seconds.")
