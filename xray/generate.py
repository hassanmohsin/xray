import argparse
import math
import multiprocessing as mp
import os
import sys
from glob import glob
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rotate
from tqdm import tqdm

from .config import decay_constant, material_constant
from .poisson_disc import poissonDisc
from .util import dir_path, read_stl, get_voxels, get_material


def get_image_array(voxels, material, axis=2):
    assert material is not None
    if material not in material_constant.keys() or not material_constant[material]:
        raise NotImplementedError(f"Available objects are {list(material_constant.keys())}")
    mat_const = -np.log(np.array(material_constant[material]))
    mat_const = (mat_const / np.sqrt(np.sum(mat_const ** 2)))
    depth = np.expand_dims(voxels.sum(axis=axis) / 255, axis=2) * mat_const
    img = np.exp(-decay_constant * depth)
    return img


def stl_to_image(stl_file, args):
    print(f"LOG: {stl_file}...")
    material = get_material(stl_file)
    voxel_file = os.path.join("./temp", f"{os.path.split(stl_file)[1]}_{args.vres}_{args.rotate_mesh}.npy")
    if args.caching and os.path.isfile(voxel_file):
        voxels = np.load(voxel_file)
        return get_image_array(voxels, material)

    mesh = read_stl(stl_file)
    # TODO: scale the mesh so that the image has reasonable dimension (xray/perimeter.py?)
    # Random rotation over x and y axis (rotation over z axis is done at image level)
    if args.rotate_mesh:
        mesh.rotate([0.5, 0., 0.0], math.radians(np.random.randint(30, 210)))
        mesh.rotate([0., 0.5, 0.0], math.radians(np.random.randint(30, 210)))
    voxels, _ = get_voxels(mesh, args.vres)
    if args.caching:
        np.save(voxel_file[:-4], voxels)
    return get_image_array(voxels, material)


# TODO: Remove for loop, use Pillow.
def remove_background(image):
    # Transparency
    new_image = []
    for item in image.getdata():
        if item[:3] == (255, 255, 255):
            new_image.append((255, 255, 255, 0))
        else:
            new_image.append(item[:3] + (128,))

    image.putdata(new_image)


def draw_canvas(id, args, images):
    canvas = np.ones((args.height, args.width, 3))
    center_points = poissonDisc(args.width,
                                args.height,
                                250,  # TODO: Remove this Hardcoded min-threshold
                                50)  # poissonDisc(width, height, min_distance, iter)
    drawn_centers = []
    for center, image in zip(center_points, images):
        # Choose one of the images of the same object randomly and rotate
        if args.rotate_object:
            image = rotate(image,
                           angle=np.random.randint(0, 360),
                           resize=True,
                           cval=1,
                           mode='constant')
        h, w = image.shape[:2]
        hpos, wpos = int(center[0]), int(center[1])
        if hpos + h >= args.height:
            hpos = args.height - h
        if wpos + w >= args.width:
            wpos = args.width - w
        drawn_centers.append([hpos, wpos])
        canvas[hpos:hpos + h, wpos:wpos + w] = canvas[hpos:hpos + h, wpos:wpos + w] * image
    plt.figure()
    plt.tight_layout()
    plt.imshow(canvas)
    # plt.axis('off')
    plt.savefig(f"{args.output}/sample_{id}.png", dpi=300)
    del canvas
    plt.figure()
    plt.gca().invert_yaxis()
    plt.title(f"Centers for {id}-th image")
    plt.scatter(*zip(*center_points), marker='o')
    plt.scatter(*zip(*drawn_centers), marker='*')
    plt.show()


def main(args):
    # Load .stl files
    stl_files = glob(os.path.join(args.input, "*.stl"))
    if len(stl_files) == 0:
        print("ERROR: No .STL files found.")
        sys.exit(1)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    if args.caching:
        if not os.path.isdir("./temp"):
            os.makedirs("./temp")

    # Get object images
    print("LOG: Converting .stl files...")
    pool = mp.Pool(args.nproc)
    images = pool.starmap(stl_to_image, zip(stl_files, repeat(args)))
    pool.close()

    # Draw canvas
    print("LOG: Generating false-color images...")
    pool = mp.Pool(args.nproc)
    pool.starmap(draw_canvas, tqdm(zip(range(args.count), repeat(args), repeat(images)), total=args.count))
    pool.close()


def argument_parser():
    parser = argparse.ArgumentParser(description='Convert STL files to false-color xray images')
    parser.add_argument('--input', type=dir_path, required=True, action='store',
                        help="Input directory containing .stl files.")
    parser.add_argument('--vres', type=int, default=20, action='store', help="Voxel resolution (default: 20)")
    parser.add_argument('--rotate-mesh', default=False, action='store_true', help="Rotate mesh")
    parser.add_argument('--rotate-object', default=False, action='store_true', help='Rotate objects')
    parser.add_argument('--width', type=int, default=1920, action='store', help="Image width  (default: 1920)")
    parser.add_argument('--height', type=int, default=1080, action='store', help="Image height (default: 1080)")
    parser.add_argument('--count', type=int, default=10, action='store',
                        help='Number of samples to generate (default: 10)')
    parser.add_argument('--output', type=str, default="./output", action='store',
                        help="Output directory (default: output)")
    parser.add_argument('--nproc', type=int, default=12, action='store', help="Number of CPUs to use. (default: 12)")
    parser.add_argument('--caching', type=int, default=0, action='store',
                        help="Enable (1) or disable (0) caching. (default: 0)")
    args = parser.parse_args()
    main(args)
