import json
import multiprocessing as mp
import os
import random
from argparse import ArgumentParser
from glob import glob
from itertools import repeat
from time import time

import matplotlib.pyplot as plt
import numpy as np

from .config import Material
from .generate import get_image_array
from .util import get_background
from .heap import MaxHeap


def get_material(s):
    mat = Material()
    for const in mat.material_constant.keys():
        if const in s:
            return const

    return ''


def find_top_bottom_surfaces(voxels):
    # Find the heights of the top and bottom surface for each pixel
    bottom_surface = np.zeros(voxels.shape[1:], dtype=np.int32)  # height of the bottom surface
    ceiling = np.zeros_like(bottom_surface).astype(np.bool)
    top_surface = bottom_surface.copy()  # height of the bottom surface
    floor = ceiling.copy()

    # TODO: Merge the loops below (Track the height)
    # At each height
    for h in range(voxels.shape[0]):
        # check if it has reached the bottom surface
        ceiling = ceiling | voxels[h]
        # increase the height by 1 if it is not bottom surface and the pixel is not occupied
        bottom_surface[~voxels[h] & ~ceiling] += 1
    # Remove the columns that doesn't contain any part of the object
    bottom_surface[~ceiling] = 0

    for h in range(voxels.shape[0] - 1, -1, -1):
        floor = floor | voxels[h]
        top_surface[~voxels[h] & ~floor] += 1
    top_surface[~floor] = 0

    return [top_surface, bottom_surface]


def generate(args, id):
    # Shuffle the files (object) to change the order they are put in the box
    indx = list(range(len(args['voxels'])))
    random.shuffle(indx)
    box_height, box_length, box_width = args['height'], args['length'], args['width']
    # Xray images along 3 different axes (x, y, z)
    canvases = [np.ones((box_height, box_length, 3)),  # From longer side
                np.ones((box_height, box_width, 3)),  # From wider side
                np.ones((box_length, box_width, 3))]  # From top

    ground = np.zeros((box_length, box_width))
    elevation = np.zeros(ground.shape, dtype=np.int32)
    stride = 20  # higher stride reduces runtime, too high (> object size) causes sparsity
    gap = 150  # minimum gap between objects
    counter = 0
    # TODO: Make sure the object of interest is in the packed box
    ooi = []

    # image rotations
    rotations = np.random.randint(2, size=len(indx))
    ooi_rotation = False

    print(f"BOX {id + 1}: Packing objects...")
    # Find the object positions
    positions = []
    for ind in indx:
        voxels = args['voxels'][ind]
        top_surface, bottom_surface = args['surfaces'][ind]
        offsets = []
        # TODO: Remove redundant search.
        # Find the minimum height at each possible position on the ground
        i = 0
        while i < box_length - voxels.shape[1]:
            j = 0
            while j < box_width - voxels.shape[2]:
                d = bottom_surface - ground[i: i + bottom_surface.shape[0], j:j + bottom_surface.shape[1]]
                offsets.append([i, j, np.min(d)])
                j += stride
            i += stride

        x, y, offset = max(offsets, key=lambda x: x[2])
        ground[x:x + voxels.shape[1], y:y + voxels.shape[2]] = top_surface - offset
        z = np.max(elevation[x:x + voxels.shape[1], y:y + voxels.shape[2]]) + max(0, int(offset))
        if z + voxels.shape[0] > box_height:
            # goes beyond the box if the object is placed, try the next one
            continue
        elevation[x:x + voxels.shape[1], y:y + voxels.shape[2]] = top_surface
        positions.append([ind, x, y, z])

    # Place the objects
    for position in positions:
        ind, x, y, z = position
        voxels, material, item = args['voxels'][ind], args['materials'][ind], args['items'][ind]
        # Draw the object image on the canvas
        # View from longer side
        xray_image = get_image_array(voxels, material)
        if rotations[ind]:
            xray_image = [np.rot90(i, 2, (0, 1)) for i in xray_image]
        image_height, image_width = xray_image[2].shape[:2]
        canvases[0][z: z + image_height, x:x + image_width] = canvases[0][
                                                              z: z + image_height,
                                                              x:x + image_width] * \
                                                              xray_image[2]

        # View from wider side
        image_height, image_width = xray_image[1].shape[:2]
        canvases[1][z: z + image_height, y:y + image_width] = canvases[1][
                                                              z: z + image_height,
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
            ooi = [x, y, z, voxels, material]
            if rotations[ind]:
                ooi_rotation = True
        counter += 1

    if len(ooi) > 0:
        x, y, z, voxels, material = ooi
    else:
        # TODO: Force packing the ooi into the box
        print(f"BOX {id + 1}: The object of interest wasn't packed, no image was generated, exiting...")
        return

    print(f"BOX {id + 1}: Packed {counter} objects in the box. Generating images...")

    # Saving the images
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(canvases[0], origin='lower')
    plt.savefig(os.path.join(args['image_dir'], f"sample_{id}_x.png"), dpi=300)
    plt.imshow(canvases[1], origin='lower')
    plt.savefig(os.path.join(args['image_dir'], f"sample_{id}_y.png"), dpi=300)
    plt.imshow(canvases[2], origin='lower')
    plt.savefig(os.path.join(args['image_dir'], f"sample_{id}_z.png"), dpi=300)

    # Save image w and w/o the OOI
    xray_image = get_image_array(voxels, material)
    xray_ooi = get_image_array(voxels, 'ooi')
    if ooi_rotation:
        xray_image = [np.rot90(i, 2, (0, 1)) for i in xray_image]
        xray_ooi = [np.rot90(i, 2, (0, 1)) for i in xray_ooi]

    image_height, image_width = xray_image[2].shape[:2]
    canvases[0][z: z + image_height, x:x + image_width] = canvases[0][z: z + image_height,
                                                          x:x + image_width] / xray_image[2]
    plt.imshow(canvases[0], origin='lower')
    plt.savefig(os.path.join(args['image_dir'], f"sample_{id}_without_ooi_x.png"), dpi=300)

    image_height, image_width = xray_ooi[2].shape[:2]
    canvases[0][z: z + image_height, x:x + image_width] = canvases[0][z: z + image_height,
                                                          x:x + image_width] * xray_ooi[2]
    plt.imshow(canvases[0], origin='lower')
    plt.savefig(os.path.join(args['image_dir'], f"sample_{id}_with_ooi_x.png"), dpi=300)

    image_height, image_width = xray_image[1].shape[:2]
    canvases[1][z: z + image_height, y:y + image_width] = canvases[1][z: z + image_height,
                                                          y:y + image_width] / xray_image[1]
    plt.imshow(canvases[1], origin='lower')
    plt.savefig(os.path.join(args['image_dir'], f"sample_{id}_without_ooi_y.png"), dpi=300)

    image_height, image_width = xray_ooi[1].shape[:2]
    canvases[1][z: z + image_height, y:y + image_width] = canvases[1][z: z + image_height,
                                                          y:y + image_width] * xray_ooi[1]
    plt.imshow(canvases[1], origin='lower')
    plt.savefig(os.path.join(args['image_dir'], f"sample_{id}_with_ooi_y.png"), dpi=300)

    image_height, image_width = xray_image[0].shape[:2]
    canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                         y:y + image_width] / xray_image[0]
    plt.imshow(canvases[2], origin='lower')
    plt.savefig(os.path.join(args['image_dir'], f"sample_{id}_without_ooi_z.png"), dpi=300)

    image_height, image_width = xray_ooi[0].shape[:2]
    canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                         y:y + image_width] * xray_ooi[0]
    plt.imshow(canvases[2], origin='lower')
    plt.savefig(os.path.join(args['image_dir'], f"sample_{id}_with_ooi_z.png"), dpi=300)


def main(args):
    # Load the voxels
    files = glob(
        os.path.join(args['voxel_dir'], '*' + str(args['scale']) + '_' + str(args['rotated']).lower() + ".npy"))
    files = [f for f in files if os.path.isfile(f)]
    if len(files) == 0:
        raise FileNotFoundError('No numpy (.npy) file found.')

    if args['ooi'] and not os.path.isfile(os.path.join(args['voxel_dir'], args['ooi'])):
        raise FileNotFoundError(f"Object of interest {args['ooi']} not found.")

    if not os.path.isdir(args['image_dir']):
        os.makedirs(args['image_dir'])

    # TODO: Share these variables among the processes instead of passing as an argument
    voxels = [np.load(f) for f in files]
    materials = [get_material(f) for f in files]
    items = [os.path.split(x)[1] for x in files]

    # Find the top and bottom surfaces

    args['voxels'] = voxels
    args['materials'] = materials
    args['items'] = items

    if args['parallel']:
        pool = mp.Pool(mp.cpu_count() if args['nproc'] == -1 else min(mp.cpu_count(), args['nproc']))
        # Find the top and bottom surfaces for each object
        args['surfaces'] = pool.map(find_top_bottom_surfaces, voxels)
        # Generate the images
        pool.starmap(generate, zip(repeat(args), range(args['box_count'])))
        pool.close()
    else:
        for i in range(args['box_count']):
            # Find the top and bottom surfaces for each object
            args['surfaces'] = [find_top_bottom_surfaces(v) for v in voxels]
            # Generate the images
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

    tic = time()
    main(args)
    toc = time() - tic
    print(f"Execution time: {toc} seconds.")


if __name__ == '__main__':
    argument_parser()
