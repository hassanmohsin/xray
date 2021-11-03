import json
import os
import unittest


class TestFiles(unittest.TestCase):
    with open("./jsons/test.json") as f:
        args = json.load(f)

    def test_stl_files(self):
        stl_files = os.listdir(self.args["stl_dir"])
        assert self.args['item_count'] <= len(stl_files)
