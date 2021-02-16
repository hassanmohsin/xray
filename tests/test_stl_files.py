import os
import unittest

from xray.config import material_constant


class TestSTL(unittest.TestCase):
    stl_dir = "./stl_files"

    def test_material(self):
        stl_files = os.listdir(self.stl_dir)
        for stl_file in stl_files:
            object_name, ext = os.path.splitext(stl_file)
            if ext != ".stl":
                continue
            material = object_name.split('_')[-1]
            assert material in material_constant.keys(), f"The material for {stl_file} is not found in the config"
