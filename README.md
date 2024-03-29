# eReg - A Simple Registration Tool

## Need

Because of security concerns, users in clinical environments do not have access to virtualization and containerization technologies such as Docker and Singularity. This becomes a problem, because most research code (especially for image registration) is built around the need to have access to these technologies. Alternatively, some tools only work on a Linux environment, or they need specific hardware resources (such as a DL accelerator card), which are not always available in clinical settings.

**eReg** is a simple registration tool that can be used in clinical environments without the need for virtualization or containerization technologies. It supports most platforms across various hardware configurations.

## Installation

With a Python 3.8+ environment, you can install **eReg** from [pypi.org](https://pypi.org/project/eReg/).

1. Create a virtual environment

```sh
python3 -m venv venv_ereg ## using native python venv
# conda create -n venv_ereg python=3.8 ## using conda
```

2. Activate the virtual environment

```sh
source venv_ereg/bin/activate ## using native python venv
# conda activate venv_ereg ## using conda
```

3. Install eReg

```sh
pip install ereg
```

## Extending eReg

To extend eReg, you first need to install **eReg** from source. Clone the repository and install the package:

```sh
git clone https://github.com/BrainLesion/eReg.git
cd eReg
pip install -e .
```

## Usage

**eReg** can be used via the command line or as a Python package. 

### Command Line Interface

The command line interface is available via the `ereg` command:

```sh
(venv_ereg) ~> ereg -h
usage: eReg version0.0.4.post76.dev0+0d89ce7 [-h] -m  -t  -o  -c  [-tff] [-lf] [-gt]

Simple registration.

options:
  -h, --help           show this help message and exit
  -m , --movingImg     The moving image to register. Can be comma-separated list of images or directory of images.
  -t , --targetImg     The target image to register to.
  -o , --output        The output. Can be single file or a directory.
  -c , --config        The configuration file to use.
  -tff , --transfile   Registration transform file; if provided, will use this transform instead of computing a new one or will save. Defaults to None.
  -lf , --log_file     The log file to write to. Defaults to None.
  -gt , --gt           The ground truth image.
```

### Pythonic Interface
The `ereg` package provides two Python interfaces, an object-oriented interface, as well as convenience functions.

#### Object-Oriented Interface

The `register` method represents the core-of the object-oriented interface:

```python
from ereg.registration import RegistrationClass

registration_obj = RegistrationClass(configuration_file) # the configuration file to use to customize the registration, and is optional
registration_obj.register(
    target_image=target_image_file, # the target image, which can be either a file or SimpleITK.Image object
    moving_image=moving_image_file, # the moving image, which can be either a file or SimpleITK.Image object
    output_image=output_file, # the output image to save the registered image to
    transform_file=transform_file, # the transform file to save the transform to; if already present, will use this transform instead of computing a new one
    log_file=log_file, # the log file to write to
)
```

Further, a resample method is available for explicitly calling transformations:
```python
registration_obj.resample_image(
    target_image=target_image_file,
    moving_image=moving_image_file,
    output_image=output_file,
    transform_file=transform_file,
    log_file=log_file,
)
```


#### Functional Interface

Additionally, **eReg** provides functional wrappers for convenience.

```python
from ereg import registration_function

ssim = registration_function(
    target_image=target_image_file, # the target image, which can be either a file or SimpleITK.Image object
    moving_image=moving_image_file, # the moving image, which can be either a file or SimpleITK.Image object
    output_image=output_file, # the output image to save the registered image to
    transform_file=transform_file, # the transform file to save the transform to; if already present, will use this transform instead of computing a new one
    log_file=log_file, # the log file to write to
    configuration=configuration_file, # the configuration file to use to customize the registration, and is optional
)
```

## Customization

eReg's registration and transformation parameters can be customized using a configuration file. The configuration file is a YAML file that contains the parameters for the registration. The default configuration file is present [here](https://github.com/BrainLesion/eReg/blob/main/ereg/configurations/sample_config.yaml). More details on the parameters and their options can be found in the configuration file itself.

<!-- ## Citation TODO -->
