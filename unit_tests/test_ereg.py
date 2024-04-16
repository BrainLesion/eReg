import logging
import os
import tempfile
import unittest

import yaml
from ereg.cli.run import main
from ereg.functional import registration_function, resample_function
from ereg.registration import RegistrationClass
from ereg.utils.io import read_image_and_cast_to_32bit_float


class TestEReg(unittest.TestCase):

    def setUp(self):
        # While this is already set within eReg it is necessary to specify it as well in the test environment
        # else pytest will overwrite it with its own logging configuration (level WARNING)
        logging.getLogger().setLevel(logging.DEBUG)

        test_data_dir = "data"
        atlas_data_dir = "atlases"
        self.moving_image = os.path.join(test_data_dir, "tcia_aaac_t1ce.nii.gz")
        self.atlas_sri = os.path.join(atlas_data_dir, "sri24", "image.nii.gz")
        self.test_config_file = os.path.join(test_data_dir, "test_config.yaml")

        test_config = {"initialization": "moments"}
        with open(self.test_config_file, "w") as f:
            yaml.dump(test_config, f)

    # Helper function
    def _image_sanity_check(self, image1, image2):
        image_1 = read_image_and_cast_to_32bit_float(image1)
        image_2 = read_image_and_cast_to_32bit_float(image2)
        assert image_1.GetSize() == image_2.GetSize(), "Image sizes do not match."
        assert (
            image_1.GetSpacing() == image_2.GetSpacing()
        ), "Image spacings do not match."
        assert image_1.GetOrigin() == image_2.GetOrigin(), "Image origins do not match."
        assert (
            image_1.GetDirection() == image_2.GetDirection()
        ), "Image directions do not match."

    ###### TESTS ######

    def test_cli_run_main(self):

        with tempfile.TemporaryDirectory() as temp_dir:
            output_image = os.path.join(temp_dir, "reg.nii.gz")
            transform_file = os.path.join(temp_dir, "trans.mat")
            main(
                [
                    "--movingImg",
                    self.moving_image,
                    "--targetImg",
                    self.atlas_sri,
                    "--output",
                    output_image,
                    "--transfile",
                    transform_file,
                    "--config",
                    self.test_config_file,
                ]
            )
            self._image_sanity_check(self.atlas_sri, output_image)

    def test_registration_function(self):
        test_config = {"initialization": "moments", "bias": True}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_image = os.path.join(temp_dir, "reg.nii.gz")
            transform_file = os.path.join(temp_dir, "trans.mat")
            log_file = os.path.join(temp_dir, "reg.log")
            registration_function(
                target_image=self.atlas_sri,
                moving_image=self.moving_image,
                output_image=output_image,
                transform_file=transform_file,
                configuration=test_config,
                log_file=log_file,
            )

            self._image_sanity_check(self.atlas_sri, output_image)

            assert os.path.exists(transform_file), "Transform file not created."
            assert os.path.exists(log_file), "Log file not created."
            # check if log_file is empty
            assert os.path.getsize(log_file) > 0, "Log file is empty."

    def test_registration_and_resampling_function(self):
        test_config = {"initialization": "moments", "bias": True}
        with tempfile.TemporaryDirectory() as temp_dir:
            reg_output_image = os.path.join(temp_dir, "reg.nii.gz")
            transform_file = os.path.join(temp_dir, "trans.mat")
            reg_log_file = os.path.join(temp_dir, "reg.log")

            registration_function(
                target_image=self.atlas_sri,
                moving_image=self.moving_image,
                output_image=reg_output_image,
                transform_file=transform_file,
                configuration=test_config,
                log_file=reg_log_file,
            )

            self._image_sanity_check(self.atlas_sri, reg_output_image)

            assert os.path.exists(transform_file), "Transform file not created."
            assert os.path.exists(reg_log_file), "Registration log file not created."
            assert os.path.getsize(reg_log_file) > 0, "Registration log file is empty."

            ## Resample
            resample_log_file = os.path.join(temp_dir, "resample.log")
            resample_output_image = os.path.join(temp_dir, "resample.nii.gz")
            resample_function(
                target_image=self.atlas_sri,
                moving_image=self.moving_image,
                output_image=resample_output_image,
                transform_file=transform_file,
                configuration=test_config,
                log_file=resample_log_file,
            )

            self._image_sanity_check(self.atlas_sri, resample_output_image)
            assert os.path.exists(transform_file), "Transform file not created"
            assert os.path.exists(resample_log_file), "Resample log file not created"
            assert os.path.getsize(resample_log_file) > 0, "Resample log file is empty"

    def test_bias_correct_image(self):
        register_obj = RegistrationClass()
        moving_bias = register_obj._bias_correct_image(self.moving_image)
        self._image_sanity_check(self.moving_image, moving_bias)
