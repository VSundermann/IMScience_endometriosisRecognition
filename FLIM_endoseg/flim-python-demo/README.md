# FLIM-Python

This is the official repository of the FLIM-Python library. FLIM-Python provides an easy to use and fast library for learning FLIM-based CNNs.

## How to run:
#### Docker
We recommend running the docker image:

```
docker run <all_wanted_arguments> jleomelo/flim-python:v1 bash
```
The image comes with all necessary libraries installed. You can skip the "installing python packages" step.

### Installing python packages
We recommend using conda to install our library (especially for Mac users). We also provide a pip option.
#### Conda
In the project root directory, create a new environment by running:
```
conda env create -f flim-python-env.yml
```

To use the created environment, run:
```
conda activate flim-python
```

#### Pip
To install the packages using pip (not recommended for Mac users), within your wanted environment, go to the project's root and execute:
```
pip install -r requirements.txt
```


### Installing pyflim
If you want to install it in development mode (changes to the lib will update automatically), in the root folder, run:

```
pip install -e .
```

Otherwise:
```
pip install .
```

### Setting up a dataset
For training a FLIM network you need to provide the training images and markers, no pixel-wise ground-truth is necessary. The markers can either be marker images (with zeros on non-marker pixels and a semantic label on the marker-pixels) or .txt files (example in tests/example_marker_file.txt).

You can use any annotation tool to create the marker images. To create the marker files, you can check the other tools from LIDS lab: [FLIMBuilder](https://github.com/LIDS-UNICAMP/FLIMBuilder) and [MISe](https://github.com/LIDS-UNICAMP/MISe).

If the code can't find a marker file with the same base name as the original image, the image will not be used during training.

If you don't want to use all marked images for training, you can provide a .txt with a list of image names (with or without extension) when creating the dataset. This is particularly useful for running multiple splits of cross-validation without having to change your file structure.

### Setting up an architecture
The easiest way to get started is to edit the "arch.json" file to detail your desired architecture. You can also easily create an architecture within python (example in the flim.ipynb).

### Running
We recommend the python notebook "flim.ipynb" for training and executing a FLIM Network. We also provided the python script flim.py.

To run the script, an example execution would be:

```
python3 flim.py --architecture arch.json --input "data/example_dataset/orig/" --markers "data/example_dataset/markers/" \
--output "results/" --save_model_at "saved_models/"
```

### Useful options
#### GPU execution with FAISS
For speeding up the code, you can use the "device=cuda:X" parameter when creating the model to allow for the clustering to be done using FAISS instead of sklearn:
```
model = flim.FLIMModel(architecture, device="cuda:0")
```
(when using the python script, pass "--use_gpu" as a flag)

#### Original image sizes or Rescale

For using the original images sizes and proportions, run:
```
dataset = data.FLIMData(orig_folder, file_list, marker_folder=marker_folder, \
              orig_ext=orig_ext, marker_ext=marker_ext, transform=data.transforms.Compose([data.ToTensor()]))

model.fit(dataset)
```

For using the all images with a fixed squared size, run:
```
dataset = data.FLIMData(orig_folder, file_list, marker_folder=marker_folder, orig_ext=orig_ext, \
                marker_ext=marker_ext, transform=data.transforms.Compose([data.Rescale(256), data.ToTensor()]))
dataloader = DataLoader(dataset, batch_size=5)
model.fit(dataloader)
```
