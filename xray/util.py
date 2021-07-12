import math
import os

import numpy as np
from PIL import ImageFilter, Image as Im
from skimage.filters import gaussian
from stl import Mesh

from .config import decay_constant, Material
from .perimeter import lines_to_voxels
from .slice import to_intersecting_lines


def get_material(s):
    mat = Material()
    for const in mat._material_constant.keys():
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


def read_stl(input_file):
    return Mesh.from_file(input_file)


def get_voxels(triangles, xy_scale=2.0):
    """
    Converts an .stl file into voxels
    :param xy_scale: Scale the object in xy-plane
    :param triangles: Mesh of the object
    :return: volume and bounding box of the voxel cube
    """
    mesh = triangles.data['vectors'].astype(np.float32)
    # (scale, shift, bounding_box) = calculate_scale_shift(mesh, resolution)
    all_points = mesh.reshape(-1, 3)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    del all_points
    shift = -1 * mins
    xresolution = 1 + xy_scale * (maxs[0] - mins[0])
    yresolution = 1 + xy_scale * (maxs[1] - mins[1])
    scale = [xy_scale, xy_scale, xy_scale]
    bounding_box = [int(xresolution), int(yresolution), math.ceil((maxs[2] - mins[2]) * xy_scale)]

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


# orange red	#FF4500	(255,69,0)
# dark orange	#FF8C00	(255,140,0)
# orange	#FFA500	(255,165,0)

def get_background(img):
    height, length = img.shape[:2]
    dot_count = 30000
    orange = [255., 165., 0.]
    # orange = np.array([216, 156, 22]) # Sixray
    mask = img == (1., 1., 1.)
    xs = np.random.choice(range(2, height - 2), dot_count)
    ys = np.random.choice(range(2, length - 2), dot_count)

    bg = np.ones(img.shape) * 255.
    bg[xs, ys] = orange
    bg[xs - 1, ys - 1] = orange
    bg[xs + 1, ys + 1] = orange

    img[mask] = np.array(
        Im.fromarray(
            bg.astype(np.uint8)
        ).filter(ImageFilter.GaussianBlur(radius=5))
    )[mask] / 255.

    return img


def get_image_array(voxels, material):
    # TODO: remove hardcoded increment of decay_constant, add it to the config
    dc = decay_constant + 10 if material == 'metal' else decay_constant
    mat = Material()
    mat_const = mat.get_const(material, color_deviation=material == 'metal')
    depth = [np.expand_dims(voxels.sum(axis=i) / 255, axis=2) * mat_const for i in range(3)]
    img = [np.exp(-dc * d) for d in depth]
    return img


def channel_wise_gaussian(image, sigmas):
    """
    Apply channel-wise gaussian filter.
    :param image: Image
    :param sigmas: sigma for R, G and B channel
    :return: Image
    """
    return np.stack(
        [gaussian(image[:, :, i], sigma=sigmas[i], preserve_range=True) for i in range(3)],
        axis=2
    )
