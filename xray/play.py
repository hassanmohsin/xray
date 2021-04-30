import json
import multiprocessing as mp
import os
import random
from argparse import ArgumentParser
from glob import glob
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np

from .config import Material
from .generate import get_image_array
from .util import get_background


def get_material(s):
    mat = Material()
    for const in mat.material_constant.keys():
        if const in s:
            return const

    return ''


def generate(args, id):
    # Shuffle the files (object) to change the order they are put in the box
    random.shuffle(args['voxels'])
    box_height, box_length, box_width = args['height'], args['length'], args['width']
    box = np.zeros((box_height, box_length, box_width), dtype=np.bool)  # Box to put the objects in
    # Xray images along 3 different axes (x, y, z)
    canvases = [np.ones((box_height, box_length, 3)),  # From longer side
                np.ones((box_height, box_width, 3)),  # From wider side
                np.ones((box_length, box_width, 3))]  # From top

    ground = np.zeros((box_length, box_width))
    elevation = np.zeros(ground.shape, dtype=np.int32)
    gap = 20  # Minimum gap between object at (x, y) plane. (!)Lower gap increases runtime significantly.
    counter = 0

    print(f"BOX {id + 1}: Packing objects...")
    for item, material in zip(args['voxels'], args['materials']):
        offsets = []

        # Find the heights of the top and bottom surface for each pixel
        bottom_surface = np.zeros(item.shape[1:], dtype=np.int32)  # height of the bottom surface
        ceiling = np.zeros_like(bottom_surface).astype(np.bool)
        top_surface = bottom_surface.copy()  # height of the bottom surface
        floor = ceiling.copy()

        # TODO: Merge the loops below (Track the height)
        for h in range(item.shape[0]):
            ceiling = ceiling | item[h]
            bottom_surface[~item[h] & ~ceiling] += 1
        bottom_surface[~ceiling] = 0

        for h in range(item.shape[0] - 1, -1, -1):
            floor = floor | item[h]
            top_surface[~item[h] & ~floor] += 1
        top_surface[~floor] = 0

        # Find the minimum height at each possible position on the ground
        for i in range(0, box.shape[1] - item.shape[1], gap):
            for j in range(0, box.shape[2] - item.shape[2], gap):
                d = bottom_surface - ground[i:i + bottom_surface.shape[0], j:j + bottom_surface.shape[1]]
                offsets.append([i, j, np.min(d)])  # append indices and the offset

        assert len(offsets) > 0
        a = max(offsets, key=lambda x: x[2])
        # Subtract offset from the top surface
        ground[a[0]:a[0] + item.shape[1], a[1]:a[1] + item.shape[2]] = top_surface - a[2]
        # add the objects into the box
        x, y = a[:2]  # Coords at h plane where the offset was lowest
        offset = int(a[2])
        height = np.max(elevation[x:x + item.shape[1], y:y + item.shape[2]])
        offset = offset if offset >= 0 else 0
        if height + offset + item.shape[0] > box.shape[0]:
            # goes beyond the box if the object is placed, try the next one
            continue
        box[height + offset:height + offset + item.shape[0], x:x + item.shape[1], y:y + item.shape[2]] = item
        elevation[x:x + item.shape[1], y:y + item.shape[2]] = top_surface  # height + offset + item.shape[0]
        # Draw the object image on the canvas
        # View from longer side
        xray_image = get_image_array(item, material, axis=2)
        image_height, image_width = xray_image.shape[:2]
        canvases[0][height + offset: height + offset + image_height, x:x + image_width] = canvases[0][
                                                                                          height + offset: height + offset + image_height,
                                                                                          x:x + image_width] * xray_image

        # View from wider side
        xray_image = get_image_array(item, material, axis=1)
        image_height, image_width = xray_image.shape[:2]
        canvases[1][height + offset: height + offset + image_height, y:y + image_width] = canvases[1][
                                                                                          height + offset: height + offset + image_height,
                                                                                          y:y + image_width] * xray_image

        # View from top
        xray_image = get_image_array(item, material, axis=0)
        image_height, image_width = xray_image.shape[:2]
        canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                             y:y + image_width] * xray_image

        # add background
        canvases[0] = get_background(canvases[0])
        canvases[1] = get_background(canvases[1])
        canvases[2] = get_background(canvases[2])
        counter += 1

    print(f"BOX {id + 1}: Packed {counter} objects in the box. Generating images...")

    # Saving the images
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(canvases[0], origin='lower')
    plt.savefig(os.path.join(args['output'], f"sample_{id}_x.png"), dpi=300)
    plt.imshow(canvases[1], origin='lower')
    plt.savefig(os.path.join(args['output'], f"sample_{id}_y.png"), dpi=300)
    plt.imshow(canvases[2], origin='lower')
    plt.savefig(os.path.join(args['output'], f"sample_{id}_z.png"), dpi=300)


def main(args):
    # Load the voxels
    files = glob(
        os.path.join(args['input_dir'], '*' + str(args['scale']) + '_' + str(args['rotated']).lower() + ".npy"))
    files = [f for f in files if os.path.isfile(f)]
    if len(files) == 0:
        raise FileNotFoundError('No numpy (.npy) file found.')

    voxels = [np.load(f) for f in files]
    materials = [get_material(f) for f in files]

    args['voxels'] = voxels
    args['materials'] = materials

    if args['parallel']:
        pool = mp.Pool(min(args['count'], mp.cpu_count()))
        pool.starmap(generate, zip(repeat(args), range(args['count'])))
    else:
        for i in range(args['count']):
            generate(args, i)


def argument_parser():
    parser = ArgumentParser(description='Convert 3D models to false-color xray images')
    parser.add_argument('--input', type=str, required=True, action='store',
                        help="JSON input")
    args = parser.parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError("Input {args.input} not found.")

    with open(args.input) as f:
        args = json.load(f)

    main(args)


if __name__ == '__main__':
    argument_parser()
