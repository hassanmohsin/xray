import os

import numpy as np

from .util import get_image_array, get_material, find_top_bottom_surfaces


class Model:
    def __init__(self, args, filename, ooi=False):
        self.filename = filename
        self.ooi = ooi
        self.voxels = np.load(os.path.join(args['voxel_dir'], self.filename))
        self.material = get_material(self.filename)
        self.images = get_image_array(self.voxels, self.material)
        self.custom_color_images = get_image_array(self.voxels, 'ooi') if ooi else None
        self.surfaces = find_top_bottom_surfaces(self.voxels)
        # replace with shape information
        self.voxels = tuple(i + 2 * args['gap'] for i in self.voxels.shape)
