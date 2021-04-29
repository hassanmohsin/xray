import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .config import Material
from .generate import get_image_array


def get_material(s):
    mat = Material()
    for const in mat.material_constant.keys():
        if const in s:
            return const

    return ''


def main():
    # print("Reading stl files and converting into voxels...")
    # stl_files = glob("./stl_files/*stl")
    # # Convert into voxels
    # vox_objects = []
    # # TODO: remove limited stl file loading
    # for stl in tqdm(stl_files[:10]):
    #     mesh = read_stl(stl)
    #     # TODO: Add rotation of mesh objects using arguments
    #     mesh.rotate([0.5, 0., 0.0], math.radians(np.random.randint(30, 210)))
    #     mesh.rotate([0., 0.5, 0.0], math.radians(np.random.randint(30, 210)))
    #     # TODO: Remove hardcoded resolution
    #     voxels, _ = get_voxels(mesh, resolution=300)
    #     vox_objects.append(voxels)

    # *True.npy files are rotated and vice versa
    voxel_files = glob("./voxels_scaled/*1.0_true.npy")
    # Shuffle the files (object) to change the order they are put in the box
    random.shuffle(voxel_files)
    box_height, box_length, box_width = 300, 500, 500
    box = np.zeros((box_height, box_length, box_width), dtype=np.bool)  # Box to put the objects in
    # Xray images along 3 different axes (x, y, z)
    canvases = [np.ones((box_height, box_length, 3)),  # From longer side
                np.ones((box_height, box_width, 3)),  # From wider side
                np.ones((box_length, box_width, 3))]  # From top

    ground = np.zeros((box_length, box_width))
    elevation = np.zeros(ground.shape, dtype=np.int32)
    gap = 20  # Minimum gap between object at (x, y) plane. (!)Lower gap increases runtime significantly.
    counter = 0

    print(f"Packing objects into the box of size {box.shape}...")
    for voxel_file in tqdm(voxel_files):
        # Get the material type
        material = get_material(voxel_file)
        # Load the model
        item = np.load(voxel_file)
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
        counter += 1

    print(f"Packed {counter} objects in the box. Now generating images")
    fig, ax = plt.subplots(2, 2, figsize=(20, 15))
    ax[0, 0].imshow(elevation)
    ax[0, 0].set_title("Elevation")
    ax[0, 1].imshow(box.sum(axis=2), origin='lower')
    ax[0, 1].set_title("Box (axis 2)")
    ax[1, 0].imshow(box.sum(axis=1), origin='lower')
    ax[1, 0].set_title("Box (axis 1)")
    ax[1, 1].imshow(box.sum(axis=0), origin='lower')
    ax[1, 1].set_title("Box (axis 0)")
    plt.savefig("depth_view.png", dpi=300)
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(20, 15))
    ax[0].imshow(canvases[2], origin='lower')
    ax[0].set_title("Box (axis 2)")
    ax[1].imshow(canvases[1], origin='lower')
    ax[1].set_title("Box (axis 1)")
    ax[2].imshow(canvases[0], origin='lower')
    ax[2].set_title("Box (axis 0)")
    plt.tight_layout()
    plt.savefig("xray.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
