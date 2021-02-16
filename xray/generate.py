import argparse
import multiprocessing as mp
import os
import sys
from glob import glob
from itertools import repeat

import numpy as np
from PIL import Image as Im
from matplotlib import colors

from .config import material_constant, alpha
from .util import dir_path, read_stl, get_voxels, get_material


def get_image_array(voxels, material):
    assert material is not None
    if material not in material_constant.keys() or not material_constant[material]:
        raise NotImplementedError(f"Available objects are {list(material_constant.keys())}")
    layer_im = np.zeros(voxels.shape + (3,))

    hue_map = np.interp(voxels, np.linspace(voxels.min(), voxels.max(), 100),
                        np.linspace(*material_constant[material], 100))
    layer_im[..., 0] = hue_map
    layer_im[..., 1] = 1.
    layer_im[..., 2] = 1 - np.exp(-1 * voxels * 1e3)

    layer_im[..., 1][voxels == 0.] = 0.  # Make background white
    layer_im[..., 2][voxels == 0.] = 1.

    return colors.hsv_to_rgb(layer_im)


def stl_to_image(stl_file, vres, output_dir):
    mesh = read_stl(stl_file)
    material = get_material(stl_file)
    # Random rotation of the mesh
    mesh.rotate(np.random.random((3,)), np.random.uniform(30., 60.))
    voxels, _ = get_voxels(mesh, vres)
    image_array = get_image_array(voxels.sum(axis=2), material)
    return image_array


def draw_canvas(images, canvas):
    canvas_height, canvas_width = canvas.shape[:2]
    for image in images:
        # TODO: add random rotation to the image
        # image = rotate(image, np.random.random() * 360, resize=True)
        h, w = image.shape[:2]
        ri, ci = np.random.randint(canvas_height - h), np.random.randint(canvas_width - w)
        # TODO: Find less crowded area of the canvas and place the image
        canvas[ri:ri + h, ci:ci + w] = image * alpha + canvas[ri:ri + h, ci:ci + w] * (1 - alpha)


def main(args):
    # Load .stl files
    stl_files = glob(os.path.join(args.input, "*.stl"))
    if len(stl_files) == 0:
        print("ERROR: No .STL files found.")
        sys.exit(1)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # Get object images
    pool = mp.Pool(args.nproc)
    images = pool.starmap(stl_to_image, zip(stl_files, repeat(args.vres), repeat(args.output)))
    pool.close()

    # Draw canvas
    canvas = np.ones((args.height, args.width, 3), dtype=np.float32)
    draw_canvas(images, canvas)
    canvas_image = Im.fromarray((canvas * 255.).astype(np.uint8))
    canvas_image.save(f"{args.output}/canvas.png", tranparency=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert STL files to false colored 2D image')
    parser.add_argument('--input', type=dir_path, required=True, action='store',
                        help="Input directory containing .stl files.")
    parser.add_argument('--vres', type=int, default=100, action='store', help="Voxel resolution")
    parser.add_argument('--width', type=int, default=512, action='store', help="Image width.")
    parser.add_argument('--height', type=int, default=512, action='store', help="Image height.")
    parser.add_argument('--count', type=int, default=1, action='store', help='Number of images.')
    parser.add_argument('--output', type=str, default="./output", action='store', help="Output directory.")
    parser.add_argument('--nproc', type=int, default=6, action='store', help="Number of CPUs to use.")
    args = parser.parse_args()
    main(args)
