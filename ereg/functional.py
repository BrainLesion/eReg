import os
from typing import Union

import SimpleITK as sitk

from ereg.registration import RegistrationClass


def registration_function(
    target_image: Union[str, sitk.Image],
    moving_image: Union[str, sitk.Image],
    output_image: str,
    configuration: str,
    transform_file: str = None,
    log_file: str = None,
    **kwargs,
) -> float:
    """
    This is a functional wrapper for the registration class.

    Args:
        target_image (Union[str, sitk.Image]): The target image.
        moving_image (Union[str, sitk.Image]): The moving image.
        output_image (str): The output image.
        config_file (str): The config file for the registration.
        transform_file (str, optional): The transform file. Defaults to None.

    Returns:
        float: The structural similarity index.
    """
    if isinstance(configuration, str):
        assert os.path.isfile(configuration), "Config file does not exist."
    elif isinstance(configuration, dict):
        pass
    else:
        raise ValueError("Config file must be a string or dictionary.")

    registration_obj = RegistrationClass(configuration)
    registration_obj.register(
        target_image=target_image,
        moving_image=moving_image,
        output_image=output_image,
        transform_file=transform_file,
        log_file=log_file,
        **kwargs,
    )
    return registration_obj.ssim_score


# TODO we also need a transformation/resample function
