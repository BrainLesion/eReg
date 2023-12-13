import argparse
import importlib.metadata
import os
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

# TODO we need to fix this version thing in a different way
# from ereg import RegistrationClass, __version__
from ereg import RegistrationClass
from ereg.utils.metrics import get_ssim

version = importlib.metadata.version("ereg")


def main(args=None):
    parser = argparse.ArgumentParser(
        # TODO needs fix
        prog=f"eReg version{version}",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Simple registration.\n\n",
    )
    registration_obj = RegistrationClass()
    # required parameters
    parser.add_argument(
        "-m",
        "--movingImg",
        metavar="",
        type=str,
        required=True,
        help="The moving image to register. Can be comma-separated list of images or directory of images.",
    )
    parser.add_argument(
        "-t",
        "--targetImg",
        metavar="",
        type=str,
        required=True,
        help=f"The target image to register to.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="",
        type=str,
        required=True,
        help="The output. Can be single file or a directory.",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="",
        type=str,
        required=True,
        help="The configuration file to use.",
    )
    # optional parameters
    parser.add_argument(
        "-tff",
        "--transfile",
        metavar="",
        type=str,
        required=False,
        default=None,
        help="Registration transform file; if provided, will use this transform instead of computing a new one or will save. Defaults to None.",
    )
    parser.add_argument(
        "-lf",
        "--log_file",
        metavar="",
        type=str,
        required=False,
        default=None,
        help="The log file to write to. Defaults to None.",
    )
    ## this should be removed
    parser.add_argument(
        "-gt",
        "--gt",
        metavar="",
        type=str,
        required=False,
        default=None,
        help=f"The ground truth image.",
    )

    args = parser.parse_args(args)

    moving_images = args.movingImg.split(",")
    moving_images_to_process = []
    dir_input = False
    for moving_image in moving_images:
        if os.path.isdir(moving_image):
            dir_input = True
            moving_images_to_process.extend(
                [str(x) for x in Path(moving_image).rglob("*.nii.gz") if x.is_file()]
            )
        else:
            moving_images_to_process.append(moving_image)

    target_image = args.targetImg
    # if target_image in available_atlases_keys:
    #     target_image = download_to_temp_image_path(available_atlases[target_image])

    ssim_scores = {}
    pbar = tqdm(
        moving_images,
        desc="Registering images",
        total=len(moving_images),
    )

    if dir_input:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    for moving_image in moving_images_to_process:
        pbar.set_description(f"Registering {os.path.basename(moving_image)}")

        output_file = args.output
        if dir_input:
            output_file = os.path.join(
                args.output,
                f"{os.path.splitext(os.path.basename(moving_image))[0]}_registered.nii.gz",
            )

        if os.path.exists(args.config):
            registration_obj.update_parameters(args.config)
        registration_obj.register(
            target_image=target_image,
            moving_image=moving_image,
            output_image=output_file,
            transform_file=args.transfile,
        )
        ssim_scores[moving_image] = registration_obj.ssim_score

        if args.gt is not None:
            print(f"Calculating GT SSIM score...{get_ssim(args.gt, output_file)}")

    pprint(f"SSIM scores: {ssim_scores}")

    # do something with args.input
    print("Finished.")


if __name__ == "__main__":
    main()
