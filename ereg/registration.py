import logging
import os
from pathlib import Path
from typing import Union

import numpy as np

# from pprint import pprint
import SimpleITK as sitk
import yaml

from ereg.utils.io import read_image_and_cast_to_32bit_float
from ereg.utils.metrics import get_ssim


class RegistrationClass:
    def __init__(
        self,
        configuration: Union[str, dict] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the class. If a config file is provided, it will be used to update the parameters.

        Args:
            config_file (Union[str, dict]): The config file or dictionary.
        """
        self.available_metrics = [
            "mattes_mutual_information",
            "ants_neighborhood_correlation",
            "correlation",
            "demons",
            "joint_histogram_mutual_information",
            "mean_squares",
        ]
        self.available_transforms = [
            "translation",
            "versor",
            "versorrigid",
            "euler",
            "similarity",
            "scale",
            "scaleversor",
            "scaleskewversor",
            "affine",
            "bspline",
            "displacement",
        ]
        self.initialization_type = {
            "geometry": sitk.CenteredTransformInitializerFilter.GEOMETRY,
            "moments": sitk.CenteredTransformInitializerFilter.MOMENTS,
        }
        self.interpolator_type = {
            "linear": sitk.sitkLinear,
            "bspline": sitk.sitkBSpline,
            "nearestneighbor": sitk.sitkNearestNeighbor,
            "gaussian": sitk.sitkGaussian,
            "labelgaussian": sitk.sitkLabelGaussian,
        }
        self.available_initializations = list(self.initialization_type.keys())
        self.available_interpolators = list(self.interpolator_type.keys())
        self.available_sampling_strategies = [
            "random",
            "regular",
            "none",
        ]
        self.total_attempts = 5
        self.transform = None

        if configuration is not None:
            self.update_parameters(configuration)
        else:
            self.parameters = self._generate_default_parameters()

    def _generate_default_parameters(self) -> dict:
        python_file_path = Path(os.path.normpath(os.path.abspath(__file__)))

        defaults_file = python_file_path.parent / "configurations/default_rigid.yaml"
        default_parameters = self.parameters = yaml.safe_load(open(defaults_file, "r"))
        return default_parameters

    @property
    def configuration(self) -> dict:
        return self.parameters

    @configuration.setter
    def configuration(
        self,
        new_config_file: Union[str, dict],
    ) -> None:
        self.parameters = self._generate_default_parameters()
        self.update_parameters(configuration=new_config_file)

    def update_parameters(
        self,
        configuration: Union[str, dict],
    ):
        """
        Update the parameters for the registration.

        Args:
            config_file (Union[str, dict]): The tring path pointing to a .yml configuration file or configuration dictionary.
        """
        if isinstance(configuration, str):
            config_data = yaml.safe_load(open(configuration, "r"))
        elif isinstance(configuration, dict):
            config_data = configuration
        else:
            raise ValueError(
                "Configuration must be a string path pointing to a .yml file or dictionary."
            )

        # Update only the keys present in the YAML file
        for key, value in config_data.items():
            if key in self.parameters:
                self.parameters[key] = value

    def register(
        self,
        target_image: Union[str, sitk.Image],
        moving_image: Union[str, sitk.Image],
        output_image: str,
        transform_file: str = None,
        log_file: str = None,
        **kwargs,
    ) -> None:
        """
        Register the moving image to the target image.

        Args:
            logger (logging.Logger): The logger to use.
            target_image (Union[str, sitk.Image]): The target image.
            moving_image (Union[str, sitk.Image]): The moving image.
            output_image (str): The output image.
            transform_file (str, optional): The transform file. Defaults to None.
        """

        if log_file is None:
            # TODO this will create trouble for non ".nii.gz" files
            log_file = output_image.replace(".nii.gz", ".log")
        logging.basicConfig(
            filename=log_file,
            format="%(asctime)s,%(name)s,%(levelname)s,%(message)s",
            datefmt="%H:%M:%S",
            level=logging.DEBUG,
        )
        self.logger = logging.getLogger("registration")

        self.logger.info(f"Target image: {target_image}, Moving image: {moving_image}")
        target_image = read_image_and_cast_to_32bit_float(target_image)
        moving_image = read_image_and_cast_to_32bit_float(moving_image)

        if self.parameters["bias_correct"]:
            self.logger.info("Bias correcting images.")
            target_image = self._bias_correct_image(target_image)
            moving_image = self._bias_correct_image(moving_image)

        compute_transform = True
        # check if transform file exists
        if transform_file is not None:
            if os.path.isfile(transform_file):
                try:
                    self.transform = sitk.ReadTransform(transform_file)
                    compute_transform = False
                except:
                    self.logger.info(
                        "Could not read transform file. Computing transform."
                    )
                    pass
        if compute_transform:
            self.logger.info(
                f"Starting registration with parameters:: {self.parameters}"
            )
            self.transform = self._register_image_and_get_transform(
                target_image=target_image,
                moving_image=moving_image,
            )
        if transform_file is not None:
            sitk.WriteTransform(self.transform, transform_file)

        # apply composite transform if provided
        if self.parameters["composite_transform"] is not None:
            self.logger.info("Applying composite transform.")
            transform_composite = sitk.ReadTransform(
                self.parameters["composite_transform"]
            )
            self.transform = sitk.CompositeTransform(
                transform_composite, self.transform
            )

        if self.parameters["composite_transform"]:
            self.logger.info("Applying previous transforms.")
            current_transform = None
            for previous_transform in self.parameters["previous_transforms"]:
                previous_transform = sitk.ReadTransform(previous_transform)
                current_transform = (
                    sitk.CompositeTransform(previous_transform, self.transform)
                    if current_transform is None
                    else sitk.CompositeTransform(previous_transform, current_transform)
                )

            self.transform = current_transform

        # no need for logging since resample_image will log by itself
        logging.shutdown()

        # resample the moving image to the target image
        self.resample_image(
            target_image=target_image,
            moving_image=moving_image,
            output_image=output_image,
            transform_file=transform_file,
        )

    def resample_image(
        self,
        target_image: Union[str, sitk.Image],
        moving_image: Union[str, sitk.Image],
        output_image: str,
        transform_file: str = None,
        log_file: str = None,
        **kwargs,
    ) -> None:
        """
        Resample the moving image to the target image.

        Args:
            logger (logging.Logger): The logger to use.
            target_image (Union[str, sitk.Image]): The target image.
            moving_image (Union[str, sitk.Image]): The moving image.
            output_image (str): The output image.
            transform_file (str, optional): The transform file. Defaults to None.
        """

        # check if output image exists
        if not os.path.exists(output_image):
            if self.transform is not None:
                if log_file is None:
                    # TODO this will create trouble for non ".nii.gz" file
                    log_file = output_image.replace(".nii.gz", ".log")
                logging.basicConfig(
                    filename=log_file,
                    format="%(asctime)s,%(name)s,%(levelname)s,%(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.DEBUG,
                )
                self.logger = logging.getLogger("registration")

                self.logger.info(
                    f"Target image: {target_image}, Moving image: {moving_image}, Transform file: {transform_file}"
                )
                target_image = read_image_and_cast_to_32bit_float(target_image)
                moving_image = read_image_and_cast_to_32bit_float(moving_image)

                self.logger.info("Resampling image.")
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(target_image)
                interpolator_type = self.interpolator_type.get(
                    self.parameters["interpolator"]
                )
                resampler.SetInterpolator(interpolator_type)
                resampler.SetDefaultPixelValue(0)
                resampler.SetTransform(self.transform)
                output_image_struct = resampler.Execute(moving_image)
                sitk.WriteImage(output_image_struct, output_image)
                self.ssim_score = get_ssim(target_image, output_image_struct)
                self.logger.info(
                    f"SSIM score of moving against target image: {self.ssim_score}"
                )
                logging.shutdown()

    def _get_transform_wrapper(self, transform: str, dim: int) -> sitk.Transform:
        """
        Get the transform class.

        Args:
            transform (str): The transform type.
            dim (int): The dimension of the transform.

        Raises:
            ValueError: If the transform is not available.

        Returns:
            sitk.Transform: The transform class.
        """
        transform_wrap = (
            transform.lower().replace(" ", "").replace("_", "").replace("-", "")
        )
        if transform_wrap == "versor":
            return sitk.VersorTransform()
        elif transform_wrap == "versorrigid":
            if dim == 3:
                return sitk.VersorRigid3DTransform()
            else:
                return eval("sitk.Euler%dDTransform()" % (dim))
        elif transform_wrap == "scaleversor":
            return sitk.ScaleVersor3DTransform()
        elif transform_wrap == "scaleskewversor":
            return sitk.ScaleSkewVersor3DTransform()
        # transforms that have specifically defined dimensions
        elif transform_wrap == "euler":
            return eval("sitk.Euler%dDTransform()" % (dim))
        elif transform_wrap == "similarity":
            return eval("sitk.Similarity%dDTransform()" % (dim))
        # transforms that use the dimension as an argument
        elif transform_wrap == "translation":
            return sitk.TranslationTransform(dim)
        elif transform_wrap == "scale":
            return sitk.ScaleTransform(dim)
        elif transform_wrap == "affine":
            return sitk.AffineTransform(dim)
        elif transform_wrap == "bspline":
            return sitk.BSplineTransform(dim)
        elif transform_wrap == "displacement":
            return sitk.DisplacementFieldTransform(dim)
        else:
            raise ValueError(f"Transform {transform} not recognized.")

    def _bias_correct_image(self, input: Union[str, sitk.Image]) -> sitk.Image:
        """
        Bias correct an image using N4BiasFieldCorrectionImageFilter.
        Taken from https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html

        Args:
            input (Union[str, sitk.Image]): The input image to bias correct.

        Returns:
            sitk.Image: The bias corrected image.
        """
        if isinstance(input, str):
            input = read_image_and_cast_to_32bit_float(input)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetSplineOrder(3)
        corrector.SetWienerFilterNoise(0.01)
        corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
        corrector.SetConvergenceThreshold(0.000001)
        # corrector.SetMaximumNumberOfIterations(50)
        expected_max_size = 25
        shrink_factor = int(max(np.asarray(input.GetSize()) / expected_max_size))
        input_scaled = sitk.Shrink(input, [shrink_factor] * input.GetDimension())

        _ = corrector.Execute(input_scaled)
        log_bias_field = corrector.GetLogBiasFieldAsImage(input)

        return input / sitk.Exp(log_bias_field)

    def _register_image_and_get_transform(
        self,
        target_image: Union[str, sitk.Image],
        moving_image: Union[str, sitk.Image],
        # metric: str = "mean_squares",
        # transform: str = "versor",
        # initialization: str = "geometry",
        # max_step: float = 5.0,
        # min_step: float = 0.01,
        # iterations: int = 200,
        # relaxation: float = 0.5,
        # grad_tolerance: float = 1e-4,
        # shrink_factors: list = [8, 4, 2],
        # smoothing_sigmas: list = [3, 2, 1],
        # sampling_strategy: str = "random",
        # sampling_percentage: float = 0.01,
        # attempts: int = 1,
        **kwargs,
    ) -> sitk.Transform:
        """
        Register two images using SimpleITK's ImageRegistrationMethod.

        Args:
            target_image (Union[str, sitk.Image]): The fixed image.
            moving_image (Union[str, sitk.Image]): The moving image.
            metric (str, optional): Registration metric. Defaults to "mean_squares".

        Returns:
            sitk.Transform: The transform from the registration.
        """

        moving_image = read_image_and_cast_to_32bit_float(moving_image)
        target_image = read_image_and_cast_to_32bit_float(target_image)

        physical_units = 1
        dimension = target_image.GetDimension()
        for dim in range(dimension):
            physical_units *= target_image.GetSpacing()[dim]

        self.logger.info("Initializing registration.")
        R = sitk.ImageRegistrationMethod()
        metric = self.parameters["metric"].lower()
        if (
            (metric == "mattesmutualinformation")
            or (metric == "mattes")
            or (metric == "mmi")
            or ("mattes" in metric)
        ):
            R.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=self.parameters["metric_parameters"][
                    "histogram_bins"
                ]
            )
        elif (
            (metric == "antsneighborhoodcorrelation")
            or (metric == "ants")
            or ("ants" in metric)
        ):
            R.SetMetricAsANTSNeighborhoodCorrelation(
                radius=self.parameters["metric_parameters"]["radius"]
            )
        elif metric == "correlation":
            R.SetMetricAsCorrelation()
        elif metric == "demons":
            R.SetMetricAsDemons(
                intensityDifferenceThreshold=self.parameters["metric_parameters"][
                    "intensityDifferenceThreshold"
                ]
            )
        elif (
            (metric == "joint_histogram_mutual_information")
            or (metric == "joint")
            or ("joint" in metric)
        ):
            R.SetMetricAsJointHistogramMutualInformation(
                numberOfHistogramBins=self.parameters["metric_parameters"][
                    "histogram_bins"
                ],
                varianceForJointPDFSmoothing=self.parameters["metric_parameters"][
                    "varianceForJointPDFSmoothing"
                ],
            )
        else:
            R.SetMetricAsMeanSquares()

        sampling_strategy_parsed = {
            "random": R.RANDOM,
            "regular": R.REGULAR,
            "none": R.NONE,
        }
        R.SetMetricSamplingStrategy(
            sampling_strategy_parsed[self.parameters["sampling_strategy"]]
        )
        R.SetMetricSamplingPercentagePerLevel(self.parameters["sampling_percentage"])

        if self.parameters["optimizer"] == "regular_step_gradient_descent":
            R.SetOptimizerAsRegularStepGradientDescent(
                minStep=self.parameters["optimizer_parameters"]["min_step"],
                numberOfIterations=self.parameters["optimizer_parameters"][
                    "iterations"
                ],
                learningRate=self.parameters["optimizer_parameters"]["learningrate"],
                # gradientMagnitudeTolerance=grad_tolerance,
                relaxationFactor=self.parameters["optimizer_parameters"]["relaxation"],
                gradientMagnitudeTolerance=self.parameters["optimizer_parameters"][
                    "tolerance"
                ],
                estimateLearningRate=R.EachIteration,
                maximumStepSizeInPhysicalUnits=self.parameters["optimizer_parameters"][
                    "max_step"
                ]
                * physical_units,
            )
        elif self.parameters["optimizer"] == "gradient_descent":
            R.SetOptimizerAsGradientDescent(
                learningRate=self.parameters["optimizer_parameters"]["learningrate"],
                numberOfIterations=self.parameters["optimizer_parameters"][
                    "iterations"
                ],
                convergenceMinimumValue=self.parameters["optimizer_parameters"][
                    "convergence_minimum"
                ],
                convergenceWindowSize=self.parameters["optimizer_parameters"][
                    "convergence_window_size"
                ],
                estimateLearningRate=R.EachIteration,
                maximumStepSizeInPhysicalUnits=self.parameters["optimizer_parameters"][
                    "max_step"
                ]
                * physical_units,
            )
        elif self.parameters["optimizer"] == "gradient_descent_line_search":
            R.SetOptimizerAsGradientDescentLineSearch(
                learningRate=self.parameters["optimizer_parameters"]["learningrate"],
                numberOfIterations=self.parameters["optimizer_parameters"][
                    "iterations"
                ],
                convergenceMinimumValue=self.parameters["optimizer_parameters"][
                    "convergence_minimum"
                ],
                convergenceWindowSize=self.parameters["optimizer_parameters"][
                    "convergence_window_size"
                ],
                lineSearchLowerLimit=self.parameters["optimizer_parameters"][
                    "line_search_lower_limit"
                ],
                lineSearchUpperLimit=self.parameters["optimizer_parameters"][
                    "line_search_upper_limit"
                ],
                lineSearchEpsilon=self.parameters["optimizer_parameters"][
                    "line_search_epsilon"
                ],
                lineSearchMaximumIterations=self.parameters["optimizer_parameters"][
                    "line_search_maximum_iterations"
                ],
                estimateLearningRate=R.EachIteration,
                maximumStepSizeInPhysicalUnits=self.parameters["optimizer_parameters"][
                    "max_step"
                ]
                * physical_units,
            )
        elif (
            self.parameters["optimizer"]
            == "Conjugate_step_gradient_descent_line_search"
        ):
            R.SetOptimizerAsConjugateGradientLineSearch(
                learningRate=self.parameters["optimizer_parameters"]["learningrate"],
                numberOfIterations=self.parameters["optimizer_parameters"][
                    "iterations"
                ],
                convergenceMinimumValue=self.parameters["optimizer_parameters"][
                    "convergence_minimum"
                ],
                convergenceWindowSize=self.parameters["optimizer_parameters"][
                    "convergence_window_size"
                ],
                lineSearchLowerLimit=self.parameters["optimizer_parameters"][
                    "line_search_lower_limit"
                ],
                lineSearchUpperLimit=self.parameters["optimizer_parameters"][
                    "line_search_upper_limit"
                ],
                lineSearchEpsilon=self.parameters["optimizer_parameters"][
                    "line_search_epsilon"
                ],
                lineSearchMaximumIterations=self.parameters["optimizer_parameters"][
                    "line_search_maximum_iterations"
                ],
                estimateLearningRate=R.EachIteration,
                maximumStepSizeInPhysicalUnits=self.parameters["optimizer_parameters"][
                    "max_step"
                ]
                * physical_units,
            )
        elif self.parameters["optimizer"] == "exhaustive":
            R.SetOptimizerAsExhaustive(
                numberOfSteps=self.parameters["optimizer_parameters"]["iterations"],
                stepLength=self.parameters["optimizer_parameters"]["step_length"],
            )
        elif self.parameters["optimizer"] == "amoeba":
            R.SetOptimizerAsAmoeba(
                numberOfIterations=self.parameters["optimizer_parameters"][
                    "iterations"
                ],
                simplexDelta=self.parameters["optimizer_parameters"]["simplex_delta"],
            )
        elif self.parameters["optimizer"] == "lbfgsb":
            R.SetOptimizerAsLBFGSB(
                numberOfIterations=self.parameters["optimizer_parameters"][
                    "iterations"
                ],
                maximumNumberOfCorrections=self.parameters["optimizer_parameters"][
                    "maximum_number_of_corrections"
                ],
                maximumNumberOfFunctionEvaluations=self.parameters[
                    "optimizer_parameters"
                ]["maximum_number_of_function_evaluations"],
                costFunctionConvergenceFactor=self.parameters["optimizer_parameters"][
                    "cost_function_convergence_factor"
                ],
            )
        elif self.parameters["optimizer"] == "lbfgs2":
            R.SetOptimizerAsLBFGS2(
                numberOfIterations=self.parameters["optimizer_parameters"][
                    "iterations"
                ],
                solutionAccuracy=self.parameters["optimizer_parameters"][
                    "solution_accuracy"
                ],
                hessianApproximateAccuracy=self.parameters["optimizer_parameters"][
                    "hessian_approximate_accuracy"
                ],
                deltaConvergenceDistance=self.parameters["optimizer_parameters"][
                    "delta_convergence_distance"
                ],
                deltaConvergenceTolerance=self.parameters["optimizer_parameters"][
                    "delta_convergence_tolerance"
                ],
                lineSearchMaximumEvaluations=self.parameters["optimizer_parameters"][
                    "line_search_maximum_evaluations"
                ],
                lineSearchMinimumStep=self.parameters["optimizer_parameters"][
                    "line_search_minimum_step"
                ],
                lineSearchMaximumStep=self.parameters["optimizer_parameters"][
                    "line_search_maximum_step"
                ],
                lineSearchAccuracy=self.parameters["optimizer_parameters"][
                    "line_search_accuracy"
                ],
            )
        elif self.parameters["optimizer"] == "one_plus_one_evolutionary":
            R.SetOptimizerAsOnePlusOneEvolutionary(
                numberOfIterations=self.parameters["optimizer_parameters"][
                    "iterations"
                ],
                epsilon=self.parameters["optimizer_parameters"]["epsilon"],
                initialRadius=self.parameters["optimizer_parameters"]["initial_radius"],
                growthFactor=self.parameters["optimizer_parameters"]["growth_factor"],
                shrinkFactor=self.parameters["optimizer_parameters"]["shrink_factor"],
            )
        elif self.parameters["optimizer"] == "powell":
            R.SetOptimizerAsPowell(
                numberOfIterations=self.parameters["optimizer_parameters"][
                    "iterations"
                ],
                maximumLineIterations=self.parameters["optimizer_parameters"][
                    "maximum_line_iterations"
                ],
                stepLength=self.parameters["optimizer_parameters"]["step_length"],
                stepTolerance=self.parameters["optimizer_parameters"]["step_tolerance"],
                valueTolerance=self.parameters["optimizer_parameters"][
                    "value_tolerance"
                ],
            )

        # R.SetOptimizerScalesFromJacobian()
        R.SetOptimizerScalesFromPhysicalShift()

        R.SetShrinkFactorsPerLevel(self.parameters["shrink_factors"])
        R.SetSmoothingSigmasPerLevel(self.parameters["smoothing_sigmas"])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        transform_function = self._get_transform_wrapper(
            self.parameters["transform"], dimension
        )
        ## todo: evaluate the viability of having default options for "rigid", "affine", and "deformable" registrations
        # rigid_registration = False
        # # euler transforms need special processing
        # if isinstance(transform_function, sitk.Euler3DTransform) or isinstance(
        #     transform_function, sitk.Euler2DTransform
        # ):
        #     rigid_registration = True
        #     initial_transform = sitk.CenteredTransformInitializer(
        #         target_image,
        #         moving_image,
        #         transform_function,
        #         self.initialization_type.get(initialization.lower(), "geometry"),
        #     )
        #     final_transform = eval(
        #         "sitk.Euler%dDTransform(initial_transform)"
        #         % (target_image.GetDimension())
        #     )
        # else:
        #     final_transform = sitk.CenteredTransformInitializer(
        #         target_image,
        #         moving_image,
        #         transform_function,
        #         self.initialization_type.get(initialization.lower(), "geometry"),
        #     )

        if self.parameters["initialization"] is not None:
            temp_moving = moving_image
            temp_initialization = self.parameters["initialization"].upper()
            # check for self initialization
            if "SELF" in temp_initialization:
                temp_moving = target_image
                temp_initialization.replace("SELF", "")
            if temp_initialization in ["MOMENTS", "GEOMETRY"]:
                initializer_type = temp_initialization
            else:
                raise ValueError(
                    "Initializer type '%s' unknown"
                    % (self.parameters["initialization"])
                )
            final_transform = sitk.CenteredTransformInitializer(
                target_image,
                temp_moving,
                transform_function,
                eval("sitk.CenteredTransformInitializerFilter.%s" % (initializer_type)),
            )
        R.SetInitialTransform(final_transform, inPlace=False)
        ## set the interpolator - all options: https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5
        # this should be linear to optimize results and computational efficacy
        R.SetInterpolator(sitk.sitkLinear)

        # R.AddCommand(sitk.sitkIterationEvent, lambda: R)
        self.logger.info("Starting registration.")
        output_transform = None
        for _ in range(self.parameters["attempts"]):
            try:
                output_transform = R.Execute(target_image, moving_image)
                break
            except RuntimeError as e:
                self.logger.warning(
                    "Registration failed with error: %s. Retrying." % (e)
                )
                continue

        if output_transform is None:
            raise RuntimeError("Registration failed.")

        registration_transform_sitk = output_transform
        # if user is requesting a rigid registration, convert the transform to a rigid transform
        if self.parameters["transform"] in ["euler", "versorrigid"]:
            try:
                # Euler Transform used:
                registration_transform_sitk = eval(
                    "sitk.Euler%dDTransform(registration_transform_sitk)" % (dimension)
                )
            except:
                # VersorRigid used: Transform from VersorRigid to Euler
                registration_transform_sitk = eval(
                    "sitk.VersorRigid%dDTransform(registration_transform_sitk)"
                    % (dimension)
                )
                tmp = eval("sitk.Euler%dDTransform()" % (dimension))
                tmp.SetMatrix(registration_transform_sitk.GetMatrix())
                tmp.SetTranslation(registration_transform_sitk.GetTranslation())
                tmp.SetCenter(registration_transform_sitk.GetCenter())
                registration_transform_sitk = tmp
        ## additional information
        # print("Metric: ", R.MetricEvaluate(target_image, moving_image), flush=True)
        # print(
        #     "Optimizer stop condition: ",
        #     R.GetOptimizerStopConditionDescription(),
        #     flush=True,
        # )
        # print("Number of iterations: ", R.GetOptimizerIteration(), flush=True)
        # print("Final metric value: ", R.GetMetricValue(), flush=True)

        # if rigid_registration:
        #     if target_image.GetDimension() == 2:
        #         output_transform = eval(
        #             "sitk.Euler%dDTransform(output_transform)"
        #             % (target_image.GetDimension())
        #         )
        #     elif target_image.GetDimension() == 3:
        #         output_transform = eval(
        #             "sitk.Euler%dDTransform(output_transform)"
        #             % (target_image.GetDimension())
        #         )
        # # VersorRigid used: Transform from VersorRigid to Euler
        # output_transform = eval(
        #     "sitk.VersorRigid%dDTransform(output_transform)"
        #     % (target_image.GetDimension())
        # )
        # tmp = eval("sitk.Euler%dDTransform()" % (target_image.GetDimension()))
        # tmp.SetMatrix(output_transform.GetMatrix())
        # tmp.SetTranslation(output_transform.GetTranslation())
        # tmp.SetCenter(output_transform.GetCenter())
        # output_transform = tmp
        return registration_transform_sitk
