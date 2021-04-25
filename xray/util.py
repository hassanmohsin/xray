import os

import numpy as np
from stl import Mesh

from .perimeter import lines_to_voxels
from .slice import to_intersecting_lines, calculate_scale_shift


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def pad_voxel_array(voxels):
    shape = voxels.shape
    new_shape = (shape[0] + 2, shape[1] + 2, shape[2] + 2)
    vol = np.zeros(new_shape, dtype=bool)
    for a in range(shape[0]):
        for b in range(shape[1]):
            for c in range(shape[2]):
                vol[a + 1, b + 1, c + 1] = voxels[a, b, c]
    return vol, (new_shape[1], new_shape[2], new_shape[0])


def get_material(stl_file):
    object_name, ext = os.path.splitext(stl_file)
    return object_name.split('_')[-1]


def read_stl(input_file):
    return Mesh.from_file(input_file)


def get_voxels(triangles, resolution):
    """
    Converts an .stl file into voxels
    :param triangles: Mesh of the object
    :param resolution: Resolution of the voxel cube
    :return: scale, shift, volume and bounding box of the voxel cube
    """
    mesh = triangles.data['vectors'].astype(np.float32)
    (scale, shift, bounding_box) = calculate_scale_shift(mesh, resolution)
    new_points = (mesh.reshape(-1, 3) + shift) * scale
    new_points = new_points.reshape(-1, 9)
    # TODO: Remove duplicate triangles from new_points
    # Note: vol should be addressed with vol[z][x][y]
    vol = np.zeros((bounding_box[2], bounding_box[0], bounding_box[1]), dtype=bool)
    for height in range(bounding_box[2]):
        # find the lines that intersect triangles at height 0 -> z
        lines = to_intersecting_lines(new_points, height)
        lines_to_voxels(lines, vol[height])

    vol, bounding_box = pad_voxel_array(vol)
    return vol, bounding_box


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
