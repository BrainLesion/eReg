# import os, tempfile, requestsw
from typing import Union

import SimpleITK as sitk

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


def read_image_and_cast_to_32bit_float(
    input_image: Union[str, sitk.Image],
) -> sitk.Image:
    """
    Cast an image to 32-bit float.

    Args:
        input (sitk.Image): The input image.

    Returns:
        sitk.Image: The casted image.
    """
    if isinstance(input_image, str):
        input_image = sitk.ReadImage(input_image, sitk.sitkFloat32)
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkFloat32)
    return caster.Execute(input_image)
