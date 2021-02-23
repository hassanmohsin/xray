# xray
Convert 3D models to false color x-ray images. Supported file format: `.stl`


## Setup
- `bash setup.sh`
- `source .venv/bin/activate`


## Test
- `python -m unittest`


## Usage
- `python -m xray --help`

    ```bash
    usage: __main__.py [-h] --input INPUT [--vres VRES] [--width WIDTH]
                       [--height HEIGHT] [--count COUNT] [--output OUTPUT]
                       [--nproc NPROC]
    
    Convert STL files to false-color xray images
    
    optional arguments:
      -h, --help       show this help message and exit
      --input INPUT    Input directory containing .stl files.
      --vres VRES      Voxel resolution (default: 100)
      --width WIDTH    Image width (default: 512)
      --height HEIGHT  Image height (default: 512)
      --count COUNT    Number of samples to generate (default: 100)
      --output OUTPUT  Output directory (default: output)
      --nproc NPROC    Number of CPUs to use. (default: 12)
    ```