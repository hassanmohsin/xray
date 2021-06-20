import os
import unittest


# Method names should start with `test`
class SampleTest(unittest.TestCase):
    def test_package_directory(self):
        assert os.path.isdir('./xray')
