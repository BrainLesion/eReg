import os
import tempfile
from pathlib import Path

import yaml

from ereg.cli.run import main
from ereg.functional import registration_function
from ereg.registration import RegistrationClass
from ereg.utils.io import read_image_and_cast_to_32bit_float


def _image_sanity_check(image1, image2):
    image_1 = read_image_and_cast_to_32bit_float(image1)
    image_2 = read_image_and_cast_to_32bit_float(image2)
    assert image_1.GetSize() == image_2.GetSize(), "Image sizes do not match."
    assert image_1.GetSpacing() == image_2.GetSpacing(), "Image spacings do not match."
    assert image_1.GetOrigin() == image_2.GetOrigin(), "Image origins do not match."
    assert (
        image_1.GetDirection() == image_2.GetDirection()
    ), "Image directions do not match."


def test_main():
    cwd = Path.cwd()
    test_data_dir = (cwd / "data").absolute().as_posix()
    atlas_data_dir = (cwd / "atlases").absolute().as_posix()
    base_config_file = os.path.join(test_data_dir, "test_config.yaml")
    moving_image = os.path.join(test_data_dir, "tcia_aaac_t1ce.nii.gz")
    output_image = os.path.join(
        tempfile.gettempdir(), "tcia_aaac_t1ce_registered.nii.gz"
    )
    output_transform = os.path.join(
        tempfile.gettempdir(), "tcia_aaac_t1ce_transform.mat"
    )
    atlas_sri = os.path.join(atlas_data_dir, "sri24", "image.nii.gz")
    test_config = {"initialization": "moments"}
    with open(base_config_file, "w") as f:
        yaml.dump(test_config, f)

    main(
        [
            "--movingImg",
            moving_image,
            "--targetImg",
            atlas_sri,
            "--output",
            output_image,
            "--transfile",
            output_transform,
            "--config",
            base_config_file,
        ]
    )
    _image_sanity_check(atlas_sri, output_image)
    os.remove(output_image)
    os.remove(output_transform)


## todo: this is not working for some reason -- will fix later
# def test_main_dir():
#     cwd = Path.cwd()
#     test_data_dir = (cwd / "data").absolute().as_posix()
#     atlas_data_dir = (cwd / "atlases").absolute().as_posix()
#     base_config_file = os.path.join(test_data_dir, "test_config.yaml")
#     atlas_sri = os.path.join(atlas_data_dir, "sri24", "image.nii.gz")
#     # check dir processing
#     output_dir = os.path.join(tempfile.gettempdir(), "dir_output")
#     test_config = {"initialization": "moments"}
#     with open(base_config_file, "w") as f:
#         yaml.dump(test_config, f)
#     main(
#         [
#             "--movingImg",
#             test_data_dir,
#             "--targetImg",
#             atlas_sri,
#             "--output",
#             output_dir,
#             "--config",
#             base_config_file,
#         ]
#     )
#     shutil.rmtree(output_dir)


def test_registration_function():
    cwd = Path.cwd()
    test_data_dir = (cwd / "data").absolute().as_posix()
    atlas_data_dir = (cwd / "atlases").absolute().as_posix()
    moving_image = os.path.join(test_data_dir, "tcia_aaac_t1ce.nii.gz")
    temp_output_dir = tempfile.gettempdir()
    output_image = os.path.join(temp_output_dir, "tcia_aaac_t1ce_registered.nii.gz")
    atlas_sri = os.path.join(atlas_data_dir, "sri24", "image.nii.gz")
    transform_file = os.path.join(temp_output_dir, "tcia_aaac_t1ce_transform.mat")
    test_config = {"initialization": "moments", "bias": True}
    registration_function(
        target_image=atlas_sri,
        moving_image=moving_image,
        output_image=output_image,
        config_file=test_config,
        transform_file=transform_file,
    )
    _image_sanity_check(atlas_sri, output_image)
    assert os.path.exists(transform_file), "Transform file not created."
    os.rmdir(temp_output_dir)


def test_bias():
    cwd = Path.cwd()
    test_data_dir = (cwd / "data").absolute().as_posix()
    moving_image = os.path.join(test_data_dir, "tcia_aaac_t1ce.nii.gz")
    register_obj = RegistrationClass()
    moving_bias = register_obj._bias_correct_image(moving_image)
    _image_sanity_check(moving_image, moving_bias)
