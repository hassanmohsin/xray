import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from packing.split_box_3D import split_box
from xray.generate import get_image_array
from xray.util import read_stl, get_voxels

if __name__ == '__main__':
    # Get the box and split
    x, y, z, dx, dy, dz = 0, 0, 0, 100, 100, 100  # Initial box location and size, box = [x,y,z,dx,dy,dz]
    box = np.array([x, y, z, dx, dy, dz], dtype=np.float32)
    boxes = split_box(3, box, random_turn=False)
    print(boxes)

    # Get the sphere
    mesh = read_stl("packing/sphere-50mm.stl")
    voxels, _ = get_voxels(mesh, resolution=100)
    voxels = zoom(voxels, (2, 2, 2))
    image = get_image_array(voxels.sum(axis=2), material='plastic')

    plt.figure()
    plt.imshow(image)
    plt.show()
