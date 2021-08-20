import json
import os
from pathlib import Path
from time import time

import numpy as np
from PIL import Image as Im

from xray.model import Model
from xray.util import get_background, channel_wise_gaussian


def save_image(canvas, sigma, image_size, output_file):
    Im.fromarray(
        (
                channel_wise_gaussian(
                    image=get_background(canvas)[::-1, :, :],
                    sigmas=sigma
                ) * 255
        ).astype('uint8')
    ).resize(image_size).save(output_file)


def generate(args, id):
    models = args['models']
    box_height = args['box_height'] + 2 * args['gap']
    box_length = args['box_length'] + 2 * args['gap']
    box_width = args['box_width'] + 2 * args['gap']

    # Xray images along 3 different axes (x, y, z)
    canvases = [
        np.ones((args['box_height'], args['box_length'], 3)),  # From longer side
        np.ones((args['box_height'], args['box_width'], 3)),  # From wider side
        np.ones((args['box_length'], args['box_width'], 3))  # From top
    ]

    image_size_x = tuple(int(t * args['image']['resize_factor']) for t in canvases[0].shape[::-1][1:])
    image_size_y = tuple(int(t * args['image']['resize_factor']) for t in canvases[1].shape[::-1][1:])
    image_size_z = tuple(int(t * args['image']['resize_factor']) for t in canvases[2].shape[::-1][1:])

    ground = np.zeros((box_length, box_width))
    elevation = np.zeros(ground.shape, dtype=np.int32)
    stride = max(args['stride'], args['gap'] // 2)
    counter = 0

    # image rotations
    rotations = np.random.randint(2, size=len(models))
    positions = []

    for ind, model in enumerate(models):
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
    backgrounds = [get_background(np.ones_like(canvas)) for canvas in canvases]
    canvases = [b.copy() for b in backgrounds]
    image_args = args['image']
    for obj_id, position in enumerate(positions):
        ind, x, y, z = position
        # Draw the object image on the canvas
        # View from longer side
        xray_image = models[ind].images
        if rotations[ind]:
            xray_image = [np.rot90(i, 2, (0, 1)) for i in xray_image]

        image_height, image_width = xray_image[2].shape[:2]
        canvases[0][z: z + image_height, x:x + image_width] = canvases[0][
                                                              z: z + image_height,
                                                              x:x + image_width
                                                              ] * xray_image[2]

        # save images with single object
        empty_canvas = backgrounds[0].copy()
        empty_canvas[z:z + image_height, x:x + image_width] = empty_canvas[
                                                              z:z + image_height,
                                                              x:x + image_width
                                                              ] * xray_image[2]
        save_image(
            empty_canvas,
            args['sigma'],
            image_size_x,
            os.path.join(image_args['dir'], f"xview/{id:06d}_{obj_id}.png")
        )

        # View from wider side
        image_height, image_width = xray_image[1].shape[:2]
        canvases[1][z: z + image_height, y:y + image_width] = canvases[1][
                                                              z: z + image_height,
                                                              y:y + image_width
                                                              ] * xray_image[1]

        # save images with single object
        empty_canvas = backgrounds[1].copy()
        empty_canvas[z:z + image_height, y:y + image_width] = empty_canvas[
                                                              z:z + image_height,
                                                              y:y + image_width
                                                              ] * xray_image[1]
        save_image(
            empty_canvas,
            args['sigma'],
            image_size_y,
            os.path.join(image_args['dir'], f"yview/{id:06d}_{obj_id}.png")
        )

        # View from top
        image_height, image_width = xray_image[0].shape[:2]
        canvases[2][x:x + image_height, y:y + image_width] = canvases[2][
                                                             x:x + image_height,
                                                             y:y + image_width
                                                             ] * xray_image[0]

        # save images with single object
        empty_canvas = backgrounds[2].copy()
        empty_canvas[x:x + image_height, y:y + image_width] = empty_canvas[
                                                              x:x + image_height,
                                                              y:y + image_width
                                                              ] * xray_image[0]
        save_image(
            empty_canvas,
            args['sigma'],
            image_size_z,
            os.path.join(image_args['dir'], f"zview/{id:06d}_{obj_id}.png")
        )

        counter += 1

    # saved packed objects
    if image_args['xview']:
        save_image(
            canvases[0],
            args['sigma'],
            image_size_x,
            os.path.join(image_args['dir'], f"xview/{id:06d}.png")
        )

    if image_args['yview']:
        save_image(
            canvases[1],
            args['sigma'],
            image_size_y,
            os.path.join(image_args['dir'], f"yview/{id:06d}.png")
        )

    if image_args['zview']:
        save_image(
            canvases[2],
            args['sigma'],
            image_size_z,
            os.path.join(image_args['dir'], f"zview/{id:06d}.png")
        )


def main(args):
    # crate directories
    image_args = args['image']
    views_dir = ['xview', 'yview', 'zview']

    for v in views_dir:
        if args['image'][v]:
            Path(os.path.join(image_args['dir'], v)).mkdir(parents=True, exist_ok=True)

    # Read the filenames
    files = []
    for f in os.listdir(args['voxel_dir']):
        for o in args['objects']:
            if f.startswith(o):
                files.append(f)

    files = [f for f in files if f.endswith(f"{str(args['scale'])}_{str(args['rotated']).lower()}.npy")]

    args['models'] = [Model(args, f) for f in files]
    generate(args, 0)

    # if args['parallel']:
    #     pool = mp.Pool(mp.cpu_count() if args['nproc'] == -1 else min(mp.cpu_count(), args['nproc']))
    #     args['models'] = pool.starmap(
    #         Model,
    #         tqdm(zip(repeat(args), files, [args['ooi'] in f for f in files]), total=len(files),
    #              desc="Loading the models")
    #     )
    #     # Generate the images
    #     # TODO: remove/find better way for image indexing
    #     pool.starmap(
    #         generate,
    #         tqdm(zip(repeat(args), range(image_args['count'])), total=image_args['count'], desc="Generating Images")
    #     )
    #     pool.close()
    # else:
    #     for i in range(image_args['count']):
    #         # Generate the images
    #         generate(args, i)


def argument_parser():
    # parser = ArgumentParser(description='Convert 3D models to false-color xray images')
    # parser.add_argument('--input', type=str, required=True, action='store', help="JSON input")
    # args = parser.parse_args()
    # if not os.path.isfile(args.input):
    #     raise FileNotFoundError("Input {args.input} not found.")
    #
    # with open(args.input) as f:
    #     args = json.load(f)
    args = None
    with open("./mix/config.json") as f:
        args = json.load(f)
    assert len(args) != 0

    tic = time()
    main(args)
    print(f"Execution time: {time() - tic:.3f} seconds.")


if __name__ == '__main__':
    argument_parser()
