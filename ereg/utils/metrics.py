# import os, tempfile, requestsw
from typing import Union

import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim

from ereg.utils.io import read_image_and_cast_to_32bit_float


def get_ssim(
    ground_truth: Union[str, sitk.Image],
    prediction: Union[str, sitk.Image],
) -> float:
    """
    Compare the ground truth image to the prediction image.

    Args:
        ground_truth (Union[str, sitk.Image]): The ground truth image.
        prediction (Union[str, sitk.Image]): The prediction image.

    Returns:
        float: SSIM
    """
    gt_image = sitk.GetArrayFromImage(read_image_and_cast_to_32bit_float(ground_truth))
    pred_image = sitk.GetArrayFromImage(read_image_and_cast_to_32bit_float(prediction))

    dynamic_range = gt_image.max() - gt_image.min()
    return ssim(gt_image, pred_image, data_range=dynamic_range)
