import argparse
import multiprocessing as mp
import os
import sys
from glob import glob
from itertools import repeat

import numpy as np
from PIL import Image as Im
from matplotlib import colors
from skimage.transform import rotate
from tqdm import tqdm

from .config import material_constant
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


def stl_to_image(stl_file, args):
    print(f"LOG: {stl_file}...")
    material = get_material(stl_file)
    voxel_file = os.path.join("./temp", f"{os.path.split(stl_file)[1]}_{args.vres}.npy")
    if os.path.isfile(voxel_file):
        voxels = np.load(voxel_file)
        return get_image_array(voxels.sum(axis=2), material)
    else:
        mesh = read_stl(stl_file)
        # Random rotation of the mesh
        mesh.rotate(np.random.random((3,)), np.random.uniform(30., 60.))
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
    # canvas = np.ones((args.height, args.width, 3), dtype=np.float32)
    canvas = Im.new("RGBA", (args.width, args.height), color=(255, 255, 255))
    # canvas_height, canvas_width = canvas.shape[:2]
    for image in images:
        h, w = image.shape[:2]
        image = rotate(image, angle=np.random.randint(0, 360), resize=True, cval=1, mode='constant')
        # image = rescale(image, scale=1.5, anti_aliasing=False)
        image = Im.fromarray((image * 255.).astype(np.uint8)).convert("RGBA")
        remove_background(image)

        # TODO: add random rotation to the image
        try:
            ri, ci = np.random.randint(args.height - h), np.random.randint(args.width - w)
        except:
            print(f"Object is larger than the canvas. Increase the canvas size. Object size: ({h}, {w})")
            continue
        canvas.paste(image, (ri, ci), mask=image)
    canvas.putalpha(255)
    canvas.save(f"{args.output}/sample_{id}.png", tranparency=0)


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
    parser.add_argument('--vres', type=int, default=100, action='store', help="Voxel resolution (default: 100)")
    parser.add_argument('--width', type=int, default=512, action='store', help="Image width  (default: 512)")
    parser.add_argument('--height', type=int, default=512, action='store', help="Image height (default: 512)")
    parser.add_argument('--count', type=int, default=100, action='store',
                        help='Number of samples to generate (default: 100)')
    parser.add_argument('--output', type=str, default="./output", action='store',
                        help="Output directory (default: output)")
    parser.add_argument('--nproc', type=int, default=12, action='store', help="Number of CPUs to use. (default: 12)")
    parser.add_argument('--caching', type=int, default=1, action='store',
                        help="Enable (1) or disable (0) caching. (default: 1)")
    args = parser.parse_args()
    main(args)
