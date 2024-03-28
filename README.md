# eReg

## Installation

With a Python 3.8+ environment, you can install eReg from [pypi.org](https://pypi.org/project/eReg/).

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

To extend eReg, you first need to install eReg from source. Clone the repository and install the package:

```sh
git clone https://github.com/BrainLesion/eReg.git
cd eReg
pip install -e .
```

## Usage

eReg can be used via the command line or as a Python package. 

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

The Pythonic interface is available via the `ereg` package, and can be used in two ways: as a functional interface or as an object-oriented interface.

#### Functional Interface

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

#### Object-Oriented Interface

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

## TODO
