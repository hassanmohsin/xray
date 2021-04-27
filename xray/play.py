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
    voxel_files = glob("./voxels_scaled/*true.npy")
    box_x, box_y, box_z = 1000, 1000, 2000
    box = np.zeros((box_x, box_y, box_z), dtype=np.bool)  # Box to put the objects in
    # Xray images along 3 different axes (x, y, z)
    canvases = [np.ones((box_y, box_z, 3)),
                np.ones((box_x, box_z, 3)),
                np.ones((box_x, box_y, 3))]

    ground = np.zeros(box[..., 0].shape)
    elevation = np.zeros(box[..., 0].shape, dtype=np.int32)
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
        bottom_surface = np.zeros(item.shape[:2])  # height of the bottom surface
        ceiling = np.zeros_like(bottom_surface).astype(np.bool)
        top_surface = bottom_surface.copy()  # height of the bottom surface
        floor = ceiling.copy()

        # TODO: Merge the loops below (Track the height)
        for h in range(item.shape[-1]):
            ceiling = ceiling | item[..., h]
            bottom_surface[~item[..., h] & ~ceiling] += 1
        bottom_surface[~ceiling] = 0

        for h in range(item.shape[-1] - 1, -1, -1):
            floor = floor | item[..., h]
            top_surface[~item[..., h] & ~floor] += 1
        top_surface[~floor] = 0

        # Find the minimum height at each possible position on the ground
        for i in range(0, box.shape[0] - item.shape[0] + 1, gap):
            for j in range(0, box.shape[1] - item.shape[1] + 1, gap):
                d = bottom_surface - ground[i:i + bottom_surface.shape[0], j:j + bottom_surface.shape[1]]
                offsets.append([i, j, np.min(d)])  # append indices and the offset

        assert len(offsets) > 0
        a = max(offsets, key=lambda x: x[2])
        # Subtract offset from the top surface
        ground[a[0]:a[0] + item.shape[0], a[1]:a[1] + item.shape[1]] = top_surface - a[2]
        # add the objects into the box
        x, y = a[:2]
        offset = int(a[2])
        height = np.max(elevation[x:x + item.shape[0], y:y + item.shape[1]])
        offset = offset if offset >= 0 else 0
        if height + offset + item.shape[2] > box.shape[2]:
            # goes beyond the box if the object is placed, try the next one
            continue
        box[x:x + item.shape[0], y:y + item.shape[1], height + offset:height + offset + item.shape[2]] = item
        elevation[x:x + item.shape[0], y:y + item.shape[1]] = height + offset + item.shape[2]
        # Draw the object image on the canvas
        # Along z-axis
        xray_image = get_image_array(item, material, axis=2)
        image_height, image_width = xray_image.shape[:2]
        canvases[2][x:x + image_height, y:y + image_width] = canvases[2][x:x + image_height,
                                                             y:y + image_width] * xray_image
        # TODO: Projection along x and y axis
        counter += 1

    print(f"Packed {counter} objects in the box. Now generating images")
    fig, ax = plt.subplots(2, 2, figsize=(20, 15))
    ax[0, 0].imshow(elevation)
    ax[0, 0].set_title("Elevation")
    ax[0, 1].imshow(box.sum(axis=0))
    ax[0, 1].set_title("Box (axis 0)")
    ax[1, 0].imshow(box.sum(axis=1))
    ax[1, 0].set_title("Box (axis 1)")
    ax[1, 1].imshow(box.sum(axis=2))
    ax[1, 1].set_title("Box (axis 2)")
    plt.show()

    # fix, ax = plt.subplots(3, figsize=(20, 15))
    # for i in range(3):
    #     ax[i].imshow(canvases[i])
    # plt.show()

    plt.imshow(canvases[2])  # Z-axis
    plt.savefig("packed2.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
