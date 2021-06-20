import json
import multiprocessing as mp
import os
import random
from argparse import ArgumentParser
from itertools import repeat
from time import time

import numpy as np
from PIL import Image as Im, ImageDraw

from .model import Model
from .util import get_background, channel_wise_gaussian


def generate(args, id):
    # Randomly choose one of the ooi model
    ooi_model = random.choice([m for m in args['models'] if m.ooi])
    other_models = random.sample([m for m in args['models'] if not m.ooi], args['item_count'] - 1)
    models = [ooi_model] + other_models
    random.shuffle(models)

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
    rotations = np.random.randint(2, size=len(models))
    ooi_rotation = False

    print(f"BOX {id + 1}: Packing objects...")
    # Find the object positions
    # Place 4 objects in 4 corners
    # TODO: update `ground` and `elevation` for the following four placement
    positions = [
        [0, 0, 0, 0],
        [1, box_length - models[1].voxels[1], 0, 0],
        [2, 0, box_width - models[2].voxels[2], 0],
        [3, box_length - models[3].voxels[1], box_width - models[3].voxels[2], 0]
    ]

    for ind, model in enumerate(models[4:], 4):
        voxels = model.voxels
        top_surface, bottom_surface = model.surfaces
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
        xray_image = models[ind].images
        if rotations[ind]:
            xray_image = [np.rot90(i, 2, (0, 1)) for i in xray_image]

        image_height, image_width = xray_image[2].shape[:2]
        canvases[0][z: z + image_height, x:x + image_width] = canvases[0][
                                                              z: z + image_height,
                                                              x:x + image_width] * \
                                                              xray_image[2]

        if models[ind].ooi:
            ooi_coordinates['x'] = [(x, canvases[0].shape[0] - (z + image_height)),
                                    (x + image_width, canvases[0].shape[0] - z)]

        # View from wider side
        image_height, image_width = xray_image[1].shape[:2]
        canvases[1][z: z + image_height, y:y + image_width] = canvases[1][
                                                              z: z + image_height,
                                                              y:y + image_width] * \
                                                              xray_image[1]
        if models[ind].ooi:
            ooi_coordinates['y'] = [(y, canvases[1].shape[0] - (z + image_height)),
                                    (y + image_width, canvases[1].shape[0] - z)]
        # View from top
        image_height, image_width = xray_image[0].shape[:2]
        canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                             y:y + image_width] * xray_image[0]
        if models[ind].ooi:
            ooi_coordinates['z'] = [(y, canvases[2].shape[0] - (x + image_height)),
                                    (y + image_width, canvases[2].shape[0] - x)]

        if models[ind].ooi:
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
            "target": args['ooi'],
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
            (channel_wise_gaussian(get_background(canvases[0])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
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
            (channel_wise_gaussian(get_background(canvases[1])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
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
            (channel_wise_gaussian(get_background(canvases[2])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
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
        if ooi_rotation:
            ooi_model.images = [np.rot90(i, 2, (0, 1)) for i in ooi_model.images]
            ooi_model.custom_color_images = [np.rot90(i, 2, (0, 1)) for i in ooi_model.custom_color_images]

        if image_args['no_ooi'] and image_args['xview']:
            image_height, image_width = ooi_model.images[2].shape[:2]
            canvases[0][z: z + image_height, x:x + image_width] = canvases[0][z: z + image_height,
                                                                  x:x + image_width] / ooi_model.images[2]
            img = Im.fromarray(
                (channel_wise_gaussian(get_background(canvases[0])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
            img.resize(image_size_x).save(os.path.join(image_args['dir'], f"no_ooi/xview/{id:06d}.png"))

        if image_args['custom_ooi'] and image_args['xview']:
            image_height, image_width = ooi_model.custom_color_images[2].shape[:2]
            canvases[0][z: z + image_height, x:x + image_width] = canvases[0][z: z + image_height,
                                                                  x:x + image_width] * ooi_model.custom_color_images[2]
            img = Im.fromarray(
                (channel_wise_gaussian(get_background(canvases[0])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
            img.resize(image_size_x).save(os.path.join(image_args['dir'], f"custom_ooi/xview/{id:06d}.png"))

        if image_args['no_ooi'] and image_args['yview']:
            image_height, image_width = ooi_model.images[1].shape[:2]
            canvases[1][z: z + image_height, y:y + image_width] = canvases[1][z: z + image_height,
                                                                  y:y + image_width] / ooi_model.images[1]
            img = Im.fromarray(
                (channel_wise_gaussian(get_background(canvases[1])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
            img.resize(image_size_y).save(os.path.join(image_args['dir'], f"no_ooi/yview/{id:06d}.png"))

        if image_args['custom_ooi'] and image_args['yview']:
            image_height, image_width = ooi_model.custom_color_images[1].shape[:2]
            canvases[1][z: z + image_height, y:y + image_width] = canvases[1][z: z + image_height,
                                                                  y:y + image_width] * ooi_model.custom_color_images[1]
            img = Im.fromarray(
                (channel_wise_gaussian(get_background(canvases[1])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
            img.resize(image_size_y).save(os.path.join(image_args['dir'], f"custom_ooi/yview/{id:06d}.png"))

        if image_args['no_ooi'] and image_args['zview']:
            image_height, image_width = ooi_model.images[0].shape[:2]
            canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                                 y:y + image_width] / ooi_model.images[0]
            img = Im.fromarray(
                (channel_wise_gaussian(get_background(canvases[2])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
            img.resize(image_size_z).save(os.path.join(image_args['dir'], f"no_ooi/zview/{id:06d}.png"))

        if image_args['custom_ooi'] and image_args['zview']:
            image_height, image_width = ooi_model.custom_color_images[0].shape[:2]
            canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                                 y:y + image_width] * ooi_model.custom_color_images[0]
            img = Im.fromarray(
                (channel_wise_gaussian(get_background(canvases[2])[::-1, :, :], args['sigma']) * 255).astype('uint8'))
            img.resize(image_size_z).save(os.path.join(image_args['dir'], f"custom_ooi/zview/{id:06d}.png"))


def main(args):
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
    if image_args['ooi']:
        for v in views_dir:
            if args['image'][v] and not os.path.isdir(os.path.join(ooi_dir, v)):
                os.makedirs(os.path.join(ooi_dir, v))
    if image_args['no_ooi']:
        for v in views_dir:
            if args['image'][v] and not os.path.isdir(os.path.join(no_ooi_dir, v)):
                os.makedirs(os.path.join(no_ooi_dir, v))
    if image_args['custom_ooi']:
        for v in views_dir:
            if args['image'][v] and not os.path.isdir(os.path.join(custom_ooi_dir, v)):
                os.makedirs(os.path.join(custom_ooi_dir, v))
    if image_args['bounding_box']:
        for v in views_dir:
            if args['image'][v] and not os.path.join(bounding_box_dir, v):
                os.makedirs(os.path.join(bounding_box_dir, v))
    for v in views_dir:
        if args['image'][v] and not os.path.isdir(os.path.join(annotations_dir, v)):
            os.makedirs(os.path.join(annotations_dir, v))

    # TODO: Share these variables among the processes instead of passing as an argument
    # TODO: assign the variables directly to args

    # Read the filenames
    files = [f for f in os.listdir(args['voxel_dir']) if
             f.endswith(f"{str(args['scale'])}_{str(args['rotated']).lower()}.npy")]

    args['models'] = [Model(args, f, args['ooi'] in f) for f in files]
    assert any([m.ooi for m in args['models']]), "No voxel file for the object of interest is found."

    if args['parallel']:
        pool = mp.Pool(mp.cpu_count() if args['nproc'] == -1 else min(mp.cpu_count(), args['nproc']))
        # Generate the images
        # TODO: remove/find better way for image indexing
        pool.starmap(generate, zip(repeat(args), range(image_args['count'])))
        pool.close()
    else:
        for i in range(image_args['count']):
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
