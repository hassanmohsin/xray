# Xray
This package generates synthetic x-ray images of 3D objects. Currently, the only supported format is `.stl`


## Setup
- `bash setup.sh` - Creates a virtual environment, installs the required python packages and downloads the 3D models (.stl files).
- `source .venv/bin/activate` - Activates the virtual environment.

## Test
- `python -m unittest` - Checks that the code is functioning properly.


## Usage
- `python -m xray --help`
  
  Output:

  ```bash
  usage: __main__.py [-h] --input INPUT
    
  Convert 3D models to false-color xray images

  optional arguments:
  -h, --help     show this help message and exit
  --input INPUT  JSON input
  ```

## Parameters
Here is an example json file where the required parameters are specified.
```json
{
  "stl_dir": "./stls",
  "voxel_dir": "./voxels",
  "ooi": "pick-x",
  "scale": 2.0,
  "rotated": true,
  "box_length": 1000,
  "box_width": 1000,
  "box_height": 500,
  "image": {
    "dir": "./dataset",
    "xview": false,
    "yview": false,
    "zview": true,
    "count": 4,
    "resize_factor": 0.5,
    "ooi": true,
    "no_ooi": false,
    "custom_ooi": false,
    "bounding_box": true,
    "annotations": true
  },
  "item_count": 45,
  "parallel": true,
  "nproc": -1,
  "gap": 0,
  "stride": 50,
  "sigma": [8, 5, 1]
}
```

### Description
- `stl_dir`: Directory to store the 3D objects.
- `voxel_dir`: Directory to store the voxels that are converted from the 3D objects.
- `ooi`: Filename of the 3D object to consider as the object of interest.
- `scale`: Scale factor of the 3D objects. For example, a value of 2 will double the size of the objects.
- `rotated`: Objects are rotated if set to `true`.
- `box_length`: Length of the packing box.
- `box_width`: Width of the packing box.
- `box_height`: Height of the packing box.
- `image`: Parameters for the generated images.
  - `dir`: Directory to store the images.
  - `xview`: If set to `true`, a view of the packing box along the x-axis will be generated. 
  - `yview`: If set to `true`, a view of the packing box along the y-axis will be generated.
  - `zview`: If set to `true`, a view of the packing box along the z-axis will be generated.
  - `count`: Number of images to generate.
  - `resize_factor`: To resize the generated images. Values < 1.0 will reduce the image size by the factor specified and vice versa. 
  - `ooi`: Whether to generate images with the object of interest.
  - `no_ooi`: Whether to generate images without the object of interest.
  - `custom_ooi`: Whether to generate images with the object of interest colored in black
  - `bounding_box`: Whether to generate images with a bounding box around the object of interest
  - `annotations`: Whether to generate the annotation files that has the coordinates of the bounding boxes.
- `item_count`: Number of 3D objects to pack in the box.
- `parallel`: To generate images in parallel.
- `nproc`: Number of thread to use when generating images in parallel.
- `gap`: Space between the objects in the packing box in all directions.
- `stride`: Space between two consecutive search locations while packing the objects.
- `sigma`: Amount of channel blur. These are `sigma` values for each RGB channels when applying channel-wise Gaussian blur.


## Generating images
There are two main steps to generate the images.
1. First, the 3D objects (in `.stl` format) need to be converted into voxelized format. This process will create numpy arrays and store those for the next step.
  
    ```bash
    python -m xray.stl_to_voxel --input <json_file>
    ```
2. Second, the voxelized objects need be packed into an imaginary 3D box to create x-ray images from different views.
    
    ```bash
    python -m xray --input <json_file>
    ``` 

## Adding new objects
Steps to follow while adding new 3D objects:
- Copy the `.stl` file into `stls` directory or the directory specified in the input json file on `stl_dir` field.
- Follow the filename pattern: `<object_name>_<material_type>.stl`. Material type is the material that the object is possible made of. For example, `ball_plastic.stl`. Currently, only one material type could be specified.
- Run `python -m xray.stl_to_voxel --input <json_file>` to convert the new object into voxelized format.

