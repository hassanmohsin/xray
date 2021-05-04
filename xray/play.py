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
    indx = list(range(len(args['voxels'])))
    random.shuffle(indx)
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
    # TODO: Make sure the object of interest is in the packed box
    ooi = []

    print(f"BOX {id + 1}: Packing objects...")
    for ind in indx:
        voxels, material, item = args['voxels'][ind], args['materials'][ind], args['items'][ind]
        offsets = []

        # Find the heights of the top and bottom surface for each pixel
        bottom_surface = np.zeros(voxels.shape[1:], dtype=np.int32)  # height of the bottom surface
        ceiling = np.zeros_like(bottom_surface).astype(np.bool)
        top_surface = bottom_surface.copy()  # height of the bottom surface
        floor = ceiling.copy()

        # TODO: Merge the loops below (Track the height)
        for h in range(voxels.shape[0]):
            ceiling = ceiling | voxels[h]
            bottom_surface[~voxels[h] & ~ceiling] += 1
        bottom_surface[~ceiling] = 0

        for h in range(voxels.shape[0] - 1, -1, -1):
            floor = floor | voxels[h]
            top_surface[~voxels[h] & ~floor] += 1
        top_surface[~floor] = 0

        # Find the minimum height at each possible position on the ground
        for i in range(0, box.shape[1] - voxels.shape[1], gap):
            for j in range(0, box.shape[2] - voxels.shape[2], gap):
                d = bottom_surface - ground[i:i + bottom_surface.shape[0], j:j + bottom_surface.shape[1]]
                offsets.append([i, j, np.min(d)])  # append indices and the offset

        assert len(offsets) > 0
        a = max(offsets, key=lambda x: x[2])
        # Subtract offset from the top surface
        ground[a[0]:a[0] + voxels.shape[1], a[1]:a[1] + voxels.shape[2]] = top_surface - a[2]
        # add the objects into the box
        x, y = a[:2]  # Coords at h plane where the offset was lowest
        offset = int(a[2])
        height = np.max(elevation[x:x + voxels.shape[1], y:y + voxels.shape[2]])
        offset = offset if offset >= 0 else 0
        if height + offset + voxels.shape[0] > box.shape[0]:
            # goes beyond the box if the object is placed, try the next one
            continue
        box[height + offset:height + offset + voxels.shape[0], x:x + voxels.shape[1], y:y + voxels.shape[2]] = voxels
        elevation[x:x + voxels.shape[1], y:y + voxels.shape[2]] = top_surface  # height + offset + item.shape[0]
        # Draw the object image on the canvas
        # View from longer side
        xray_image = get_image_array(voxels, material)
        image_height, image_width = xray_image[2].shape[:2]
        canvases[0][height + offset: height + offset + image_height, x:x + image_width] = canvases[0][
                                                                                          height + offset: height + offset + image_height,
                                                                                          x:x + image_width] * \
                                                                                          xray_image[2]

        # View from wider side
        image_height, image_width = xray_image[1].shape[:2]
        canvases[1][height + offset: height + offset + image_height, y:y + image_width] = canvases[1][
                                                                                          height + offset: height + offset + image_height,
                                                                                          y:y + image_width] * \
                                                                                          xray_image[1]

        # View from top
        image_height, image_width = xray_image[0].shape[:2]
        canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                             y:y + image_width] * xray_image[0]

        # add background
        canvases[0] = get_background(canvases[0])
        canvases[1] = get_background(canvases[1])
        canvases[2] = get_background(canvases[2])
        if item == args['ooi']:
            ooi = [x, y, height, offset, voxels, material]
        counter += 1

    if len(ooi) > 0:
        x, y, height, offset, voxels, material = ooi
    else:
        # TODO: Force packing the ooi into the box
        print(f"BOX {id + 1}: The object of interest wasn't packed, no image was generated, exiting...")
        return

    print(f"BOX {id + 1}: Packed {counter} objects in the box. Generating images...")

    # Saving the images
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(canvases[0], origin='lower')
    plt.savefig(os.path.join(args['output_dir'], f"sample_{id}_x.png"), dpi=300)
    plt.imshow(canvases[1], origin='lower')
    plt.savefig(os.path.join(args['output_dir'], f"sample_{id}_y.png"), dpi=300)
    plt.imshow(canvases[2], origin='lower')
    plt.savefig(os.path.join(args['output_dir'], f"sample_{id}_z.png"), dpi=300)

    # Save image w and w/o the OOI
    xray_image = get_image_array(voxels, material)
    xray_ooi = get_image_array(voxels, 'ooi')

    image_height, image_width = xray_image[2].shape[:2]
    canvases[0][height + offset: height + offset + image_height, x:x + image_width] = canvases[0][
                                                                                      height + offset: height + offset + image_height,
                                                                                      x:x + image_width] / xray_image[2]
    plt.imshow(canvases[0], origin='lower')
    plt.savefig(os.path.join(args['output_dir'], f"sample_{id}_without_ooi_x.png"), dpi=300)

    image_height, image_width = xray_ooi[2].shape[:2]
    canvases[0][height + offset: height + offset + image_height, x:x + image_width] = canvases[0][
                                                                                      height + offset: height + offset + image_height,
                                                                                      x:x + image_width] * xray_ooi[2]
    plt.imshow(canvases[0], origin='lower')
    plt.savefig(os.path.join(args['output_dir'], f"sample_{id}_with_ooi_x.png"), dpi=300)

    image_height, image_width = xray_image[1].shape[:2]
    canvases[1][height + offset: height + offset + image_height, y:y + image_width] = canvases[1][
                                                                                      height + offset: height + offset + image_height,
                                                                                      y:y + image_width] / xray_image[1]
    plt.imshow(canvases[1], origin='lower')
    plt.savefig(os.path.join(args['output_dir'], f"sample_{id}_without_ooi_y.png"), dpi=300)

    image_height, image_width = xray_ooi[1].shape[:2]
    canvases[1][height + offset: height + offset + image_height, y:y + image_width] = canvases[1][
                                                                                      height + offset: height + offset + image_height,
                                                                                      y:y + image_width] * xray_ooi[1]
    plt.imshow(canvases[1], origin='lower')
    plt.savefig(os.path.join(args['output_dir'], f"sample_{id}_with_ooi_y.png"), dpi=300)

    image_height, image_width = xray_image[0].shape[:2]
    canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                         y:y + image_width] / xray_image[0]
    plt.imshow(canvases[2], origin='lower')
    plt.savefig(os.path.join(args['output_dir'], f"sample_{id}_without_ooi_z.png"), dpi=300)

    image_height, image_width = xray_ooi[0].shape[:2]
    canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                         y:y + image_width] * xray_ooi[0]
    plt.imshow(canvases[2], origin='lower')
    plt.savefig(os.path.join(args['output_dir'], f"sample_{id}_with_ooi_z.png"), dpi=300)


def main(args):
    # Load the voxels
    files = glob(
        os.path.join(args['input_dir'], '*' + str(args['scale']) + '_' + str(args['rotated']).lower() + ".npy"))
    files = [f for f in files if os.path.isfile(f)]
    if len(files) == 0:
        raise FileNotFoundError('No numpy (.npy) file found.')

    if args['ooi'] and not os.path.isfile(os.path.join(args['input_dir'], args['ooi'])):
        raise FileNotFoundError(f"Object of interest {args['ooi']} not found.")

    if not os.path.isdir(args['output_dir']):
        os.makedirs(args['output_dir'])

    voxels = [np.load(f) for f in files]
    materials = [get_material(f) for f in files]

    args['voxels'] = voxels
    args['materials'] = materials
    args['items'] = [os.path.split(x)[1] for x in files]

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
