import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from packing.split_box_3D import split_box
from xray.generate import get_image_array
from xray.util import read_stl, get_voxels

if __name__ == '__main__':
    # Get the box and split
    x, y, z, dx, dy, dz = 0, 0, 0, 200, 200, 200  # Initial box location and size, box = [x,y,z,dx,dy,dz]
    sphere_size = 100
    box = np.array([x, y, z, dx, dy, dz], dtype=np.float32)
    # TODO: set a minimum threshold in split_box so that there is no size one box
    mini_boxes = np.ceil(split_box(3, box, random_turn=False)).astype('int')
    print(mini_boxes)

    box = np.zeros((dx, dy, dz))

    # Get the sphere
    sphere = read_stl("packing/sphere-50mm.stl")
    sph_voxels, _ = get_voxels(sphere, resolution=sphere_size)
    del sphere

    # Get the sizes of each mini_boxes
    sizes = abs(np.subtract(mini_boxes[:, 3:], mini_boxes[:, :3]))
    shortest_size = np.argmin(sizes, axis=1)
    mini_spheres = []
    for i, sh_size in enumerate(shortest_size):
        # get the zoom factor
        factor = sizes[i][sh_size] / sph_voxels.shape[sh_size]
        # zoom in/out
        mini_sphere = zoom(sph_voxels, (factor, factor, factor))
        p, q, r = mini_boxes[i][:3]
        s, t, u = mini_boxes[i][:3] + mini_sphere.shape
        box[p:s, q:t, r:u] = mini_sphere

    plt.figure()
    plt.imshow(get_image_array(box.sum(axis=2), material='plastic'))
    plt.show()
    # plt.close('all')
    # for vox in mini_spheres:
    #     image = get_image_array(vox.sum(axis=2), material='plastic')
    #
    #     plt.figure()
    #     plt.imshow(image)
    #     plt.show()
