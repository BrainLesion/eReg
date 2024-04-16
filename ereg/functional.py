import os
from typing import Union

import SimpleITK as sitk

from ereg.registration import RegistrationClass
from ereg.utils import initialize_configuration


def registration_function(
    target_image: Union[str, sitk.Image],
    moving_image: Union[str, sitk.Image],
    output_image: str,
    configuration: Union[str, dict] = None,
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
        configuration (Union[str, dict], optional): The configuration file or dictionary. Defaults to None.
        transform_file (str, optional): The transform file. Defaults to None.

    Returns:
        float: The structural similarity index.
    """
    configuration = initialize_configuration(configuration)

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


def resample_function(
    target_image: Union[str, sitk.Image],
    moving_image: Union[str, sitk.Image],
    output_image: str,
    transform_file: str,
    configuration: Union[str, dict] = None,
    log_file: str = None,
    **kwargs,
) -> float:
    """
    Resample the moving image onto the space of the target image using a given transformation.

    Args:
        target_image (Union[str, sitk.Image]): The target image onto which the moving image will be resampled.
        moving_image (Union[str, sitk.Image]): The image to be resampled.
        output_image (str): The filename or path where the resampled image will be saved.
        transform_file (str): The file containing the transformation to be applied.
        configuration (Union[str, dict], optional): The configuration file or dictionary. Defaults to None.
        log_file (str, optional): The file to log progress and details of the resampling process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the resampling function.

    Returns:
        float: The structural similarity index (SSIM) between the resampled image and the target image.
    """
    configuration = initialize_configuration(configuration)

    registration_obj = RegistrationClass(configuration)

    registration_obj.resample_image(
        target_image=target_image,
        moving_image=moving_image,
        output_image=output_image,
        transform_file=transform_file,
        log_file=log_file,
        **kwargs,
    )

    return registration_obj.ssim_score
