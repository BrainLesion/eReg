# import os, tempfile, requestsw
from typing import Union
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim

# def download_to_temp_image_path(url_to_download: str) -> str:
#     """
#     This function downloads the image from the url and returns the path to the downloaded image.

#     Args:
#         url_to_download (str): The url to download the image from.

#     Returns:
#         str: The path to the downloaded image.
#     """
#     downloaded_file = os.path.join(tempfile.gettempdir(), "image.nii.gz")
#     try:
#         print("Downloading and extracting sample data")
#         r = requests.get(url_to_download)
#         with open(downloaded_file, "wb") as fd:
#             fd.write(r.content)
#     except Exception as error:
#         print(f"Error downloading atlas file: {error}")

#     return downloaded_file


def read_image_and_cast_to_32bit_float(input: Union[str, sitk.Image]) -> sitk.Image:
    """
    Cast an image to 32-bit float.

    Args:
        input (sitk.Image): The input image.

    Returns:
        sitk.Image: The casted image.
    """
    if isinstance(input, str):
        input = sitk.ReadImage(input, sitk.sitkFloat32)
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkFloat32)
    return caster.Execute(input)


def get_ssim(ground_truth: Union[str, sitk.Image], prediction: Union[str, sitk.Image]) -> float:
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