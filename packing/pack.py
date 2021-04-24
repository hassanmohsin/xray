from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def crop_model(voxels):
    s = np.sum(voxels, axis=(1, 2))
    has_voxels = np.where(s > 0)[0]
    voxels = voxels[has_voxels[0]:has_voxels[-1] + 1]
    s = np.sum(voxels, axis=(0, 2))
    has_voxels = np.where(s > 0)[0]
    voxels = voxels[:, has_voxels[0]:has_voxels[-1] + 1]
    s = np.sum(voxels, axis=(0, 1))
    has_voxels = np.where(s > 0)[0]
    voxels = voxels[:, :, has_voxels[0]:has_voxels[-1] + 1]
    return voxels


if __name__ == '__main__':
    voxel_files = glob("../temp/*True.npy")
    box_x, box_y, box_z = 1000, 1000, 1000
    box = np.zeros((box_x, box_y, box_z), dtype=np.bool)

    ground = np.zeros(box[..., 0].shape)  # Ground height
    gap = 20  # pixel
    positions = []

    print("Finding places for objects...")
    for voxel_file in tqdm(voxel_files):
        # Load the model
        item = crop_model(np.load(voxel_file))
        offsets = []

        # Find the heights of the top and bottom surface for each pixel
        bottom_surface = np.zeros(item.shape[:2])  # height of the bottom surface
        ceiling = np.zeros_like(bottom_surface).astype(np.bool)
        top_surface = bottom_surface.copy()  # height of the bottom surface
        floor = ceiling.copy()

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
        if np.max(ground) <= box.shape[2]:
            positions.append([a[:2], a[2]])
        else:
            break

    # Pack the box with objects
    print(f"Found place for {len(positions)} objects, packing into the box...")
    elevation = np.zeros(box[..., 0].shape, dtype=np.int32)
    for voxel_file, position in zip(voxel_files, positions):
        voxels = crop_model(np.load(voxel_file))
        x, y = position[0]
        offset = int(position[1])
        height = np.max(elevation[x:x + voxels.shape[0], y:y + voxels.shape[1]])
        box[x:x + voxels.shape[0], y:y + voxels.shape[1], height + offset:height + offset + voxels.shape[2]] = voxels
        # print(x, x+voxels.shape[0], y, y+voxels.shape[1], height+offset, height+offset+voxels.shape[2], voxels.shape)
        elevation[x:x + voxels.shape[0], y:y + voxels.shape[1]] = height + offset + voxels.shape[2]

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
