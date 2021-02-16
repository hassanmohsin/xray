import unittest

from xray.util import *


class UtilTest(unittest.TestCase):
    stl_file = "./stl_files/handgun.stl"

    def test_read_stl(self):
        mesh = read_stl(self.stl_file)
        assert isinstance(mesh, Mesh)

    def test_get_voxels(self):
        mesh = read_stl(self.stl_file)
        voxels, _ = get_voxels(mesh, 10)
        assert np.any(voxels)
        assert voxels.shape == (9, 12, 12)


if __name__ == '__main__':
    unittest.main()
