import argparse
import math
import multiprocessing as mp
import os
import random
import sys
from glob import glob
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as Im
from matplotlib import colors
from skimage.transform import rotate
from tqdm import tqdm

from .config import material_constant
from .poisson_disc import poissonDisc
from .util import dir_path, read_stl, get_voxels, get_material


def get_image_array(voxels, material):
    assert material is not None
    if material not in material_constant.keys() or not material_constant[material]:
        raise NotImplementedError(f"Available objects are {list(material_constant.keys())}")

    material_const = np.array(material_constant[material])
    # make 2 more by shifting the region to have variety in color
    scale_shift = 0.03
    material_consts = [material_const - scale_shift, material_const, material_const + scale_shift]
    image_arrays = []
    for const in material_consts:
        layer_im = np.zeros(voxels.shape + (3,))
        hue_map = np.interp(voxels,
                            np.linspace(voxels.min(), voxels.max(), 100),
                            np.linspace(*const, 100))
        layer_im[..., 0] = hue_map
        layer_im[..., 1] = np.interp(voxels,
                                     np.linspace(voxels.min(), voxels.max(), 100),
                                     np.linspace(0, 1, 100))
        layer_im[..., 2] = 1.
        # layer_im[..., 2] = np.interp(intensity,
        #                              np.linspace(intensity.min(), intensity.max(), 100),
        #                              np.linspace(0, 1, 100))

        layer_im[..., 1][voxels == 0.] = 0.  # Make background white
        layer_im[..., 2][voxels == 0.] = 1.

        image_arrays.append(colors.hsv_to_rgb(layer_im))

    return image_arrays


def stl_to_image(stl_file, args, rotate=True):
    print(f"LOG: {stl_file}...")
    material = get_material(stl_file)
    voxel_file = os.path.join("./temp", f"{os.path.split(stl_file)[1]}_{args.vres}.npy")
    if args.caching and os.path.isfile(voxel_file):
        voxels = np.load(voxel_file)
        return get_image_array(voxels.sum(axis=2), material)

    mesh = read_stl(stl_file)
    # TODO: scale the mesh so that the image has reasonable dimension (xray/perimeter.py?)
    # Random rotation over x and y axis (rotation over z axis is done at image level)
    if rotate:
        mesh.rotate([0.5, 0., 0.0], math.radians(np.random.randint(30, 210)))
        mesh.rotate([0., 0.5, 0.0], math.radians(np.random.randint(30, 210)))
    voxels, _ = get_voxels(mesh, args.vres)
    if args.caching:
        np.save(voxel_file[:-4], voxels)
    return get_image_array(voxels.sum(axis=2), material)


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
    canvas = Im.new("RGBA", (args.width, args.height), color=(255, 255, 255))
    center_points = poissonDisc(args.width - 200,
                                args.height - 200,
                                250,  # TODO: Remove this Hardcoded min-threshold
                                50)  # poissonDisc(width, height, min_distance, iter)
    drawn_centers = []
    for center, image in zip(center_points, images):
        # Choose one of the images of the same object randomly and rotate
        image = rotate(image[random.randint(0, 2)],
                       angle=np.random.randint(0, 360),
                       resize=True,
                       cval=1,
                       mode='constant')
        w, h = image.shape[:2]
        image = Im.fromarray((image * 255.).astype(np.uint8)).convert("RGBA")
        remove_background(image)
        xpos, ypos = center[0], center[1]
        if xpos + w >= args.width:
            xpos = args.width - w
        if ypos + h >= args.height:
            ypos = args.height - h
        drawn_centers.append([xpos, ypos])
        canvas.paste(image, (int(xpos), int(ypos)), mask=image)
    canvas.putalpha(255)
    canvas.save(f"{args.output}/sample_{id}.png", tranparency=0)
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
    parser.add_argument('--vres', type=int, default=20, action='store', help="Voxel resolution (default: 100)")
    parser.add_argument('--width', type=int, default=2048, action='store', help="Image width  (default: 512)")
    parser.add_argument('--height', type=int, default=2048, action='store', help="Image height (default: 512)")
    parser.add_argument('--count', type=int, default=100, action='store',
                        help='Number of samples to generate (default: 100)')
    parser.add_argument('--output', type=str, default="./output", action='store',
                        help="Output directory (default: output)")
    parser.add_argument('--nproc', type=int, default=12, action='store', help="Number of CPUs to use. (default: 12)")
    parser.add_argument('--caching', type=int, default=0, action='store',
                        help="Enable (1) or disable (0) caching. (default: 0)")
    args = parser.parse_args()
    main(args)
