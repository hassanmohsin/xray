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
    box_height = args['box_height'] + 2 * args['gap']
    box_length = args['box_length'] + 2 * args['gap']
    box_width = args['box_width'] + 2 * args['gap']

    # Xray images along 3 different axes (x, y, z)
    canvases = [np.ones((args['box_height'], args['box_length'], 3)),  # From longer side
                np.ones((args['box_height'], args['box_width'], 3)),  # From wider side
                np.ones((args['box_length'], args['box_width'], 3))]  # From top

    image_size_x = tuple(int(t * args['image']['resize_factor']) for t in canvases[0].shape[::-1][1:])
    image_size_y = tuple(int(t * args['image']['resize_factor']) for t in canvases[1].shape[::-1][1:])
    image_size_z = tuple(int(t * args['image']['resize_factor']) for t in canvases[2].shape[::-1][1:])

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

    def get_info(coords):
        (xmin, ymin), (xmax, ymax) = coords
        info = {
            "id": f"{id:06d}",
            "xmin": int(xmin * args['image']['resize_factor']),
            "ymin": int(ymin * args['image']['resize_factor']),
            "xmax": int(xmax * args['image']['resize_factor']),
            "ymax": int(ymax * args['image']['resize_factor']),
            "filename": f"{id:06d}.png"
        }
        return info

    image_args = args['image']
    if image_args['xview']:
        img = Im.fromarray(
            (gaussian_filter(get_background(canvases[0])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
        if image_args['ooi']:
            img.resize(image_size_x).save(os.path.join(image_args['dir'], f"ooi/xview/{id:06d}.png"))
        if image_args['bounding_box']:
            img1 = ImageDraw.Draw(img)
            img1.rectangle(ooi_coordinates['x'], outline="red")
            img.resize(image_size_x).save(os.path.join(image_args['dir'], f"bounding_box/xview/{id:06d}.png"))

        if image_args['annotations']:
            with open(os.path.join(image_args['dir'], f"annotations/xview/{id:06d}.json"), 'w') as f:
                json.dump(get_info(ooi_coordinates['x']), f)

    if image_args['yview']:
        img = Im.fromarray(
            (gaussian_filter(get_background(canvases[1])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
        if image_args['ooi']:
            img.resize(image_size_y).save(os.path.join(image_args['dir'], f"ooi/yview/{id:06d}.png"))
        if image_args['bounding_box']:
            img1 = ImageDraw.Draw(img)
            img1.rectangle(ooi_coordinates['y'], outline="red")
            img.resize(image_size_y).save(os.path.join(image_args['dir'], f"bounding_box/yview/{id:06d}.png"))

        if image_args['annotations']:
            with open(os.path.join(image_args['dir'], f"annotations/yview/{id:06d}.json"), 'w') as f:
                json.dump(get_info(ooi_coordinates['y']), f)

    if image_args['zview']:
        img = Im.fromarray(
            (gaussian_filter(get_background(canvases[2])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
        if image_args['ooi']:
            img.resize(image_size_z).save(os.path.join(image_args['dir'], f"ooi/zview/{id:06d}.png"))
        if image_args['bounding_box']:
            img1 = ImageDraw.Draw(img)
            img1.rectangle(ooi_coordinates['z'], outline="red")
            img.resize(image_size_z).save(os.path.join(image_args['dir'], f"bounding_box/zview/{id:06d}.png"))

        if image_args['annotations']:
            with open(os.path.join(image_args['dir'], f"annotations/zview/{id:06d}.json"), 'w') as f:
                json.dump(get_info(ooi_coordinates['z']), f)

    if image_args['no_ooi'] and image_args['custom_ooi']:
        xray_image, xray_ooi = args['ooi_images']
        if ooi_rotation:
            xray_image = [np.rot90(i, 2, (0, 1)) for i in xray_image]
            xray_ooi = [np.rot90(i, 2, (0, 1)) for i in xray_ooi]

        if image_args['no_ooi'] and image_args['xview']:
            image_height, image_width = xray_image[2].shape[:2]
            canvases[0][z: z + image_height, x:x + image_width] = canvases[0][z: z + image_height,
                                                                  x:x + image_width] / xray_image[2]
            img = Im.fromarray(
                (gaussian_filter(get_background(canvases[0])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
            img.resize(image_size_x).save(os.path.join(image_args['dir'], f"no_ooi/xview/{id:06d}.png"))

        if image_args['custom_ooi'] and image_args['xview']:
            image_height, image_width = xray_ooi[2].shape[:2]
            canvases[0][z: z + image_height, x:x + image_width] = canvases[0][z: z + image_height,
                                                                  x:x + image_width] * xray_ooi[2]
            img = Im.fromarray(
                (gaussian_filter(get_background(canvases[0])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
            img.resize(image_size_x).save(os.path.join(image_args['dir'], f"custom_ooi/xview/{id:06d}.png"))

        if image_args['no_ooi'] and image_args['yview']:
            image_height, image_width = xray_image[1].shape[:2]
            canvases[1][z: z + image_height, y:y + image_width] = canvases[1][z: z + image_height,
                                                                  y:y + image_width] / xray_image[1]
            img = Im.fromarray(
                (gaussian_filter(get_background(canvases[1])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
            img.resize(image_size_y).save(os.path.join(image_args['dir'], f"no_ooi/yview/{id:06d}.png"))

        if image_args['custom_ooi'] and image_args['yview']:
            image_height, image_width = xray_ooi[1].shape[:2]
            canvases[1][z: z + image_height, y:y + image_width] = canvases[1][z: z + image_height,
                                                                  y:y + image_width] * xray_ooi[1]
            img = Im.fromarray(
                (gaussian_filter(get_background(canvases[1])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
            img.resize(image_size_y).save(os.path.join(image_args['dir'], f"custom_ooi/yview/{id:06d}.png"))

        if image_args['no_ooi'] and image_args['zview']:
            image_height, image_width = xray_image[0].shape[:2]
            canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                                 y:y + image_width] / xray_image[0]
            img = Im.fromarray(
                (gaussian_filter(get_background(canvases[2])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
            img.resize(image_size_z).save(os.path.join(image_args['dir'], f"no_ooi/zview/{id:06d}.png"))

        if image_args['custom_ooi'] and image_args['zview']:
            image_height, image_width = xray_ooi[0].shape[:2]
            canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                                 y:y + image_width] * xray_ooi[0]
            img = Im.fromarray(
                (gaussian_filter(get_background(canvases[2])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
            img.resize(image_size_z).save(os.path.join(image_args['dir'], f"custom_ooi/zview/{id:06d}.png"))


def main(args):
    # Read the filenames
    files = [f for f in os.listdir(args['voxel_dir']) if
             f.endswith(f"{str(args['scale'])}_{str(args['rotated']).lower()}.npy")]

    if args['ooi'] and not os.path.isfile(os.path.join(args['voxel_dir'], args['ooi'])):
        raise FileNotFoundError(f"Object of interest {args['ooi']} not found.")

    # crate directories
    image_args = args['image']
    if not (image_args['ooi'] or image_args['no_ooi'] or image_args['custom_ooi'] or image_args['bounding_box']):
        print("ERROR: Specify at least one of the image output criteria.")
        return

    dataset_dir = os.path.join(image_args['dir'])
    ooi_dir = os.path.join(dataset_dir, 'ooi')
    no_ooi_dir = os.path.join(dataset_dir, 'no_ooi')
    custom_ooi_dir = os.path.join(dataset_dir, 'custom_ooi')
    bounding_box_dir = os.path.join(dataset_dir, 'bounding_box')
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    views_dir = ['xview', 'yview', 'zview']
    if image_args['ooi'] and not os.path.isdir(ooi_dir):
        os.makedirs(ooi_dir)
        for v in views_dir:
            if args['image'][v]:
                os.makedirs(os.path.join(ooi_dir, v))
    if image_args['no_ooi'] and not os.path.isdir(no_ooi_dir):
        os.makedirs(no_ooi_dir)
        for v in views_dir:
            if args['image'][v]:
                os.makedirs(os.path.join(no_ooi_dir, v))
    if image_args['custom_ooi'] and not os.path.isdir(custom_ooi_dir):
        os.makedirs(custom_ooi_dir)
        for v in views_dir:
            if args['image'][v]:
                os.makedirs(os.path.join(custom_ooi_dir, v))
    if image_args['bounding_box'] and not os.path.isdir(bounding_box_dir):
        os.makedirs(bounding_box_dir)
        for v in views_dir:
            if args['image'][v]:
                os.makedirs(os.path.join(bounding_box_dir, v))
    if not os.path.isdir(annotations_dir):
        os.makedirs(annotations_dir)
        for v in views_dir:
            if args['image'][v]:
                os.makedirs(os.path.join(annotations_dir, v))

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
        pool.starmap(generate, zip(repeat(args), range(args['image']['count'])))
        pool.close()
    else:
        for i in range(args['box_count']):
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
    print(f"Execution time: {time() - tic:.3f} seconds.")


if __name__ == '__main__':
    argument_parser()
