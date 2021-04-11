import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from packing.split_box_3D import split_box
from xray.generate import get_image_array
from xray.util import read_stl, get_voxels

if __name__ == '__main__':
    # Get the box and split
    x, y, z, dx, dy, dz = 0, 0, 0, 100, 100, 100  # Initial box location and size, box = [x,y,z,dx,dy,dz]
    sphere_size = 100
    box = np.array([x, y, z, dx, dy, dz], dtype=np.float32)
    # TODO: set a minimum threshold in split_box so that there is no size one box
    mini_boxes = np.ceil(split_box(3, box, random_turn=False)).astype('int')
    box = np.zeros((dx, dy, dz))

    # Get the sphere
    sphere = read_stl("packing/sphere-50mm.stl")
    sph_voxels, _ = get_voxels(sphere, resolution=sphere_size)
    del sphere

    # Get the sizes of each mini_boxes
    sizes = mini_boxes[:, 3:]
    shortest_size = np.argmin(sizes, axis=1)
    for i, sh_size in enumerate(shortest_size):
        # get the zoom factor
        factor = (sizes[i][sh_size] - 1) / sph_voxels.shape[sh_size]  # subtract 1 to get rid of off-by-one error
        # zoom in/out
        mini_sphere = zoom(sph_voxels, (factor, factor, factor))
        p, q, r = mini_boxes[i][:3]
        s, t, u = mini_boxes[i][:3] + mini_sphere.shape
        box[p:s, q:t, r:u] = mini_sphere

    plt.figure()
    plt.imshow(get_image_array(box.sum(axis=0), material='plastic'))
    plt.show()

    ax = plt.figure(figsize=(20, 20)).add_subplot(projection='3d')
    ax.voxels(box, edgecolor='k')
    plt.show()
