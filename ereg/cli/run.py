import argparse, ast, os
from pathlib import Path
from tqdm import tqdm
from pprint import pprint

from SIMPLEREG import __version__, RegistrationClass
from SIMPLEREG.utilities import get_ssim


def main(args=None):
    parser = argparse.ArgumentParser(
        prog=f"MAIN_Entry v{__version__}",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Description goes here.\n\n",
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
                [str(x) for x in Path(moving_image).rglob("*") if x.is_file()]
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

        registration_obj.register(
            target_image=target_image,
            moving_image=moving_image,
            output_image=output_file,
            transform_file=args.transfile,
            bias=args.bias,
            metric=args.metric,
            transform=args.transform,
            transform_composite=args.transcomp,
            initialization=args.initialize,
            iterations=args.iterations,
            relaxation=args.relaxation,
            tolerance=args.tolerance,
            maxstep=args.maxstep,
            minstep=args.minstep,
            interpolator=args.interpltr,
            shrink_factors=args.shrink,
            smoothing_sigmas=args.smooth,
            sampling_strategy=args.sampling,
            sampling_percentage=args.samplePerc,
            attempts=args.attempts,
            log_file=args.log_file,
        )
        ssim_scores[moving_image] = registration_obj.ssim_score

        if args.gt is not None:
            print(f"Calculating GT SSIM score...{get_ssim(args.gt, output_file)}")

    pprint(f"SSIM scores: {ssim_scores}")

    # do something with args.input
    print("Finished.")


if __name__ == "__main__":
    main()
