import json
import multiprocessing as mp
import os
import random
from argparse import ArgumentParser
from itertools import repeat
from time import time

import numpy as np
from PIL import Image as Im, ImageDraw
from scipy.ndimage import gaussian_filter

from .config import Material
from .util import get_background, get_image_array


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
    indx = list(range(min(len(args['items']), args['item_count'])))
    random.shuffle(indx)
    box_height = args['height'] + 2 * args['gap']
    box_length = args['length'] + 2 * args['gap']
    box_width = args['width'] + 2 * args['gap']

    # Xray images along 3 different axes (x, y, z)
    canvases = [np.ones((args['height'], args['length'], 3)),  # From longer side
                np.ones((args['height'], args['width'], 3)),  # From wider side
                np.ones((args['length'], args['width'], 3))]  # From top

    ground = np.zeros((box_length, box_width))
    elevation = np.zeros(ground.shape, dtype=np.int32)
    stride = max(args['stride'], args['gap'] // 2)
    counter = 0
    # TODO: Make sure the object of interest is in the packed box
    ooi = []
    ooi_coordinates = {}

    # image rotations
    rotations = np.random.randint(2, size=len(indx))
    ooi_rotation = False

    print(f"BOX {id + 1}: Packing objects...")
    # Find the object positions
    # Place 4 objects in 4 corners
    # TODO: update `ground` and `elevation` for the following four placement
    positions = [
        [indx[0], 0, 0, 0],
        [indx[1], box_length - args['voxels'][indx[1]][1], 0, 0],
        [indx[2], 0, box_width - args['voxels'][indx[2]][2], 0],
        [indx[3], box_length - args['voxels'][indx[3]][1], box_width - args['voxels'][indx[3]][2], 0]
    ]

    for ind in indx[4:]:
        voxels = args['voxels'][ind]
        top_surface, bottom_surface = args['surfaces'][ind]
        top_surface = np.pad(top_surface, args['gap'])
        bottom_surface = np.pad(bottom_surface, args['gap'])
        offsets = []
        # TODO: Remove redundant search.
        # Find the minimum height at each possible position on the ground
        i = 0
        while i < box_length - voxels[1]:
            j = 0
            while j < box_width - voxels[2]:
                d = bottom_surface - ground[i: i + bottom_surface.shape[0], j:j + bottom_surface.shape[1]]
                offsets.append([i, j, np.min(d)])
                j += stride
            i += stride

        x, y, offset = max(offsets, key=lambda x: x[2])
        z = np.max(elevation[x:x + voxels[1], y:y + voxels[2]]) + max(0, int(offset))
        if z + voxels[0] > box_height:
            # goes beyond the box if the object is placed, try the next one
            continue

        ground[x:x + voxels[1], y:y + voxels[2]] = top_surface - offset
        elevation[x:x + voxels[1], y:y + voxels[2]] = top_surface
        positions.append([ind, x, y, z])

    # Place the objects
    for position in positions:
        ind, x, y, z = position
        # Draw the object image on the canvas
        # View from longer side
        xray_image = args['images'][ind]
        if rotations[ind]:
            xray_image = [np.rot90(i, 2, (0, 1)) for i in xray_image]

        image_height, image_width = xray_image[2].shape[:2]
        canvases[0][z: z + image_height, x:x + image_width] = canvases[0][
                                                              z: z + image_height,
                                                              x:x + image_width] * \
                                                              xray_image[2]

        if args['items'][ind] == args['ooi']:
            ooi_coordinates['x'] = [(x, canvases[0].shape[0] - (z + image_height)),
                                    (x + image_width, canvases[0].shape[0] - z)]

        # View from wider side
        image_height, image_width = xray_image[1].shape[:2]
        canvases[1][z: z + image_height, y:y + image_width] = canvases[1][
                                                              z: z + image_height,
                                                              y:y + image_width] * \
                                                              xray_image[1]
        if args['items'][ind] == args['ooi']:
            ooi_coordinates['y'] = [(y, canvases[1].shape[0] - (z + image_height)),
                                    (y + image_width, canvases[1].shape[0] - z)]
        # View from top
        image_height, image_width = xray_image[0].shape[:2]
        canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                             y:y + image_width] * xray_image[0]
        if args['items'][ind] == args['ooi']:
            ooi_coordinates['z'] = [(y, canvases[2].shape[0] - (x + image_height)),
                                    (y + image_width, canvases[2].shape[0] - x)]

        if args['items'][ind] == args['ooi']:
            ooi = [x, y, z]
            if rotations[ind]:
                ooi_rotation = True
        counter += 1

    if len(ooi) > 0:
        x, y, z = ooi
    else:
        # TODO: Force packing the ooi into the box
        print(f"BOX {id + 1}: The object of interest wasn't packed, no image was generated, exiting...")
        return

    print(f"BOX {id + 1}: Packed {counter} objects in the box. Generating images...")

    # TODO: avoid repetitive gaussian filtering
    # Save images with and without bounding boxes
    # Along X-axis
    img = Im.fromarray((gaussian_filter(get_background(canvases[0])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
    img.save(os.path.join(args['dirs']['ooi'], f"image-x_{id}.png"))
    img1 = ImageDraw.Draw(img)
    img1.rectangle(ooi_coordinates['x'], outline="red")
    img.save(os.path.join(args['dirs']['ground_truth'], f"image-x_{id}.png"))

    # Write annotations
    (xmin, ymin), (xmax, ymax) = ooi_coordinates['x']
    info = {
        "id": id,
        "xmin": xmin,
        "ymin": int(ymin),
        "xmax": xmax,
        "ymax": int(ymax),
        "directory": args['dirs']['ooi'],
        "filename": f"image-x_{id}.png"
    }

    with open(os.path.join(args['dirs']['annotations'], f"annot-x_{id}.json"), 'w') as f:
        json.dump(info, f)

    # Along Y-axis
    img = Im.fromarray((gaussian_filter(get_background(canvases[1])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
    img.save(os.path.join(args['dirs']['ooi'], f"image-y_{id}.png"))
    img1 = ImageDraw.Draw(img)
    img1.rectangle(ooi_coordinates['y'], outline="red")
    img.save(os.path.join(args['dirs']['ground_truth'], f"image-y_{id}.png"))

    # Write annotations
    (xmin, ymin), (xmax, ymax) = ooi_coordinates['y']
    info = {
        "id": id,
        "xmin": xmin,
        "ymin": int(ymin),
        "xmax": xmax,
        "ymax": int(ymax),
        "directory": args['dirs']['ooi'],
        "filename": f"image-y_{id}.png"
    }

    with open(os.path.join(args['dirs']['annotations'], f"annot-y_{id}.json"), 'w') as f:
        json.dump(info, f)

    # Along Z-axis
    img = Im.fromarray((gaussian_filter(get_background(canvases[2])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
    img.save(os.path.join(args['dirs']['ooi'], f"image-z_{id}.png"))
    img1 = ImageDraw.Draw(img)
    img1.rectangle(ooi_coordinates['z'], outline="red")
    img.save(os.path.join(args['dirs']['ground_truth'], f"image-z_{id}.png"))

    # Write annotations
    (xmin, ymin), (xmax, ymax) = ooi_coordinates['z']
    info = {
        "id": id,
        "xmin": xmin,
        "ymin": int(ymin),
        "xmax": xmax,
        "ymax": int(ymax),
        "directory": args['dirs']['ooi'],
        "filename": f"image-z_{id}.png"
    }

    with open(os.path.join(args['dirs']['annotations'], f"annot-z_{id}.json"), 'w') as f:
        json.dump(info, f)

    # Save image w and w/o the OOI
    xray_image, xray_ooi = args['ooi_images']
    if ooi_rotation:
        xray_image = [np.rot90(i, 2, (0, 1)) for i in xray_image]
        xray_ooi = [np.rot90(i, 2, (0, 1)) for i in xray_ooi]

    # image_height, image_width = xray_image[2].shape[:2]
    # canvases[0][z: z + image_height, x:x + image_width] = canvases[0][z: z + image_height,
    #                                                       x:x + image_width] / xray_image[2]
    # img = Im.fromarray((gaussian_filter(get_background(canvases[0])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
    # img.save(os.path.join(args['image_dir'], f"image_x_no_ooi_{id}.png"))
    #
    # image_height, image_width = xray_ooi[2].shape[:2]
    # canvases[0][z: z + image_height, x:x + image_width] = canvases[0][z: z + image_height,
    #                                                       x:x + image_width] * xray_ooi[2]
    # img = Im.fromarray((gaussian_filter(get_background(canvases[0])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
    # img.save(os.path.join(args['image_dir'], f"image_x_ooi_{id}.png"))
    #
    # image_height, image_width = xray_image[1].shape[:2]
    # canvases[1][z: z + image_height, y:y + image_width] = canvases[1][z: z + image_height,
    #                                                       y:y + image_width] / xray_image[1]
    # img = Im.fromarray((gaussian_filter(get_background(canvases[1])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
    # img.save(os.path.join(args['image_dir'], f"image_y_no_ooi_{id}.png"))
    #
    # image_height, image_width = xray_ooi[1].shape[:2]
    # canvases[1][z: z + image_height, y:y + image_width] = canvases[1][z: z + image_height,
    #                                                       y:y + image_width] * xray_ooi[1]
    # img = Im.fromarray((gaussian_filter(get_background(canvases[1])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
    # img.save(os.path.join(args['image_dir'], f"image_y_ooi_{id}.png"))

    image_height, image_width = xray_image[0].shape[:2]
    canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                         y:y + image_width] / xray_image[0]
    img = Im.fromarray((gaussian_filter(get_background(canvases[2])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
    img.save(os.path.join(args['dirs']['no_ooi'], f"image-z_{id}.png"))

    # image_height, image_width = xray_ooi[0].shape[:2]
    # canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
    #                                                      y:y + image_width] * xray_ooi[0]
    # img = Im.fromarray((gaussian_filter(get_background(canvases[2])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
    # img.save(os.path.join(args['image_dir'], f"image_z_ooi_{id}.png"))


def main(args):
    # Read the filenames
    files = [f for f in os.listdir(args['voxel_dir']) if
             f.endswith(f"{str(args['scale'])}_{str(args['rotated']).lower()}.npy")]

    if args['ooi'] and not os.path.isfile(os.path.join(args['voxel_dir'], args['ooi'])):
        raise FileNotFoundError(f"Object of interest {args['ooi']} not found.")

    ooi_dir = os.path.join(args['dataset_dir'], 'images', 'ooi')
    no_ooi_dir = os.path.join(args['dataset_dir'], 'images', 'no_ooi')
    ground_truth_dir = os.path.join(args['dataset_dir'], 'images', 'ground_truth')
    annotations_dir = os.path.join(args['dataset_dir'], 'annotations')

    args['dirs'] = {
        'ooi': ooi_dir,
        'no_ooi': no_ooi_dir,
        'ground_truth': ground_truth_dir,
        'annotations': annotations_dir
    }

    for v in args['dirs'].values():
        if not os.path.isdir(v):
            os.makedirs(v)

    # TODO: Share these variables among the processes instead of passing as an argument
    # TODO: assign the variables directly to args
    # Put the ooi at the beginning
    files.remove(args['ooi'])
    files = [args['ooi']] + files
    voxels = [np.load(os.path.join(args['voxel_dir'], f)) for f in files]
    materials = [get_material(f) for f in files]
    images = [get_image_array(v, m) for (v, m) in zip(voxels, materials)]
    # TODO: remove hardcoded 'metal'
    ooi_images = [get_image_array(np.load(os.path.join(args['voxel_dir'], args['ooi'])), 'metal'),
                  get_image_array(np.load(os.path.join(args['voxel_dir'], args['ooi'])), 'ooi')]

    # Find the top and bottom surfaces

    args['voxels'] = [(tuple(j + 2 * args['gap'] for j in i)) for i in [v.shape for v in voxels]]
    args['materials'] = materials
    args['items'] = files
    args['images'] = images
    args['ooi_images'] = ooi_images
    args['surfaces'] = [find_top_bottom_surfaces(v) for v in voxels]
    del voxels

    if args['parallel']:
        pool = mp.Pool(mp.cpu_count() if args['nproc'] == -1 else min(mp.cpu_count(), args['nproc']))
        # Generate the images
        # TODO: remove/find better way for image indexing
        pool.starmap(generate, zip(repeat(args), range(args['sample_count'])))
        pool.close()
    else:
        for i in range(args['sample_count']):
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
    print(f"Execution time: {time() - tic} seconds.")


if __name__ == '__main__':
    argument_parser()
