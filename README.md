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
      --caching CACHING  Enable (1) or disable (0) caching. (default: 1)
    ```
  
## STL files
Following are the objects in `stl_files` directory:

| Object              | Material Type |
|---------------------|---------------|
| 15mm_cube           | plastic       |
| bearing_guide       | metal         |
| bearing             | metal         |
| black_bear          | plastic       |
| bolt_long           | metal         |
| bookmark            | plastic       |
| bottle_opener       | metal         |
| bottle_opener_round | metal         |
| box                 | plastic       |
| bronze_dragon       | metal         |
| comb                | plastic       |
| cube                | plastic       |
| disk                | metal         |
| door_handle         | metal         |
| dragon              | plastic       |
| fan_blade           | metal         |
| gear                | metal         |
| handle              | metal         |
| knot                | plastic       |
| metal_plate         | metal         |
| mouse               | plastic       |
| ring                | metal         |
| shuriken            | metal         |
| three               | plastic       |
| toy_statue          | plastic       |
| two                 | metal         |
| water_bottle        | plastic       |

Naming convention: `<object_name>_<material_type>.stl`

## Parameters
- Voxel resolution (`vres`) and scaling: Voxel resolution is the number of slices along X and Y axis. Slices required along Z axis is calculated using the resolution and the scale in X-Y plane ($$\Delta Z \times (\frac{resolution}{\max{\Delta X, \Delta Y}})$$). Standard resolution of the voxels is 100 (unit) that provides faster image generation. However, it can be varied between 50 to 500 or until the RAM memory gets exhausted. It is recommended to enable `caching` when a higher resolution is used. The scale of the objects is automatically calculated depending on the voxel resolution. Increase the `width` and `height` to contain all the objects.
- Rotation: The rotation determines at what angle the individual object images are rotated. The value is random between 0 and 360.

### Known issues:
- When `caching` is enabled, random rotation of mesh is disabled. However, 2D images are still rotated.
