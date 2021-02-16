import argparse

from .util import dir_path


def get_voxel(stl_file):
    pass


def get_voxels(stl_dir):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert STL files to false colored 2D image')
    parser.add_argument('--input', type=dir_path, action='store', help="Input directory containing .stl files.")
    parser.add_argument('--vres', type=int, default=100, action='store', help="Voxel resolution")
    parser.add_argument('--width', type=int, default=1024, action='store', help="Image width.")
    parser.add_argument('--height', type=int, default=768, action='store', help="Image height.")
    parser.add_argument('--count', type=int, default=1, action='store', help='Number of images.')
    args = parser.parse_args()
