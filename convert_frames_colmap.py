#!/usr/bin/env python3
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path


def run_command(command: list[str], env: dict[str, str] | None = None) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, env=env)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser("COLMAP frame converter with GPU/CPU controls")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--last_frame_id", default=299, type=int)

    parser.add_argument("--skip_matching", action="store_true")
    parser.add_argument("--skip_undistortion", action="store_true")
    parser.add_argument("--resize", action="store_true")

    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help=(
            "Physical GPU index to use exclusively. "
            "Set to -1 or pass --no_gpu to avoid CUDA_VISIBLE_DEVICES pinning."
        ),
    )
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=4,
        help="Maximum CPU threads for COLMAP and math backends.",
    )

    parser.add_argument(
        "--enable-dense",
        action="store_true",
        help="Enable dense stage in automatic_reconstructor (disabled by default).",
    )

    parser.add_argument(
        "--env-name",
        default="colmap-cuda",
        type=str,
        help="Conda environment name used when --colmap_executable is not set.",
    )
    parser.add_argument(
        "--colmap_executable",
        default="",
        type=str,
        help="Optional absolute path to COLMAP executable; overrides conda run.",
    )
    parser.add_argument("--magick_executable", default="", type=str)

    parser.add_argument(
        "--cpu-affinity",
        default="",
        type=str,
        help="Optional CPU affinity mask, e.g. 0-3, equivalent to taskset -c 0-3.",
    )

    return parser


def build_colmap_prefix(args) -> list[str]:
    if args.colmap_executable:
        return [args.colmap_executable]
    return ["conda", "run", "-n", args.env_name, "colmap"]


def maybe_with_taskset(command: list[str], cpu_affinity: str) -> list[str]:
    if not cpu_affinity:
        return command
    return ["taskset", "-c", cpu_affinity, *command]


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    if args.max_cpu_threads < 1:
        raise ValueError("--max-cpu-threads must be >= 1")

    source_path = Path(args.source_path).resolve()
    distorted_sparse = source_path / "distorted" / "sparse" / "0"
    matching_image_path = source_path / "input"

    if not source_path.is_dir():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    if not distorted_sparse.is_dir():
        raise FileNotFoundError(f"Sparse model does not exist: {distorted_sparse}")

    run_matching = not args.skip_matching
    if run_matching and not matching_image_path.is_dir():
        print(
            "Warning: matching image directory does not exist, auto-skipping matching stage: "
            f"{matching_image_path}"
        )
        run_matching = False

    thread_limit = str(args.max_cpu_threads)
    run_dense = args.enable_dense
    use_gpu = 0 if args.no_gpu else 1

    colmap_env = dict(os.environ)
    colmap_env["OMP_NUM_THREADS"] = thread_limit
    colmap_env["OPENBLAS_NUM_THREADS"] = thread_limit
    colmap_env["MKL_NUM_THREADS"] = thread_limit
    colmap_env["NUMEXPR_NUM_THREADS"] = thread_limit
    colmap_env["VECLIB_MAXIMUM_THREADS"] = thread_limit

    if use_gpu == 1 and args.gpu_index >= 0:
        colmap_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    print(f"CPU threads limit: {thread_limit}")
    if use_gpu == 0:
        print("GPU disabled via --no_gpu")
    elif args.gpu_index >= 0:
        print(f"Using physical GPU: {args.gpu_index} (mapped to local GPU 0)")
    else:
        print("Using COLMAP default GPU selection")
    print(f"Dense reconstruction enabled: {run_dense}")
    if args.cpu_affinity:
        print(f"CPU affinity: {args.cpu_affinity}")

    colmap_prefix = build_colmap_prefix(args)
    magick_exec = args.magick_executable if args.magick_executable else "magick"

    if run_matching:
        auto_recon_cmd = [
            *colmap_prefix,
            "automatic_reconstructor",
            "--workspace_path",
            str(source_path),
            "--image_path",
            str(matching_image_path),
            "--camera_model",
            args.camera,
            "--use_gpu",
            str(use_gpu),
        ]

        if use_gpu == 1 and args.gpu_index >= 0:
            auto_recon_cmd.extend(["--gpu_index", "0"])
        if not run_dense:
            auto_recon_cmd.extend(["--dense", "0", "--mesher", "poisson"])

        run_command(maybe_with_taskset(auto_recon_cmd, args.cpu_affinity), env=colmap_env)
    else:
        print("Skipping matching/reconstruction stage.")

    if args.skip_undistortion:
        print("Skipping undistortion because --skip_undistortion was set.")
        print("Done.")
        return

    for frame_num in range(1, args.last_frame_id + 1):
        frame_id = f"{frame_num:0>6}"
        frame_name = f"frame{frame_id}"
        frame_path = source_path / frame_name
        temp_undist_path = frame_path / "_colmap_undist_tmp"
        images_path = frame_path / "images"

        print(f"Processing /{frame_name}")

        if not frame_path.is_dir():
            raise FileNotFoundError(f"Frame directory does not exist: {frame_path}")

        if temp_undist_path.exists():
            shutil.rmtree(temp_undist_path)

        undist_cmd = [
            *colmap_prefix,
            "image_undistorter",
            "--image_path",
            str(frame_path),
            "--input_path",
            str(distorted_sparse),
            "--output_path",
            str(temp_undist_path),
            "--output_type",
            "COLMAP",
        ]
        run_command(maybe_with_taskset(undist_cmd, args.cpu_affinity), env=colmap_env)

        undist_images_path = temp_undist_path / "images"
        if not undist_images_path.is_dir():
            raise FileNotFoundError(f"Undistorted images were not generated: {undist_images_path}")

        images_path.mkdir(parents=True, exist_ok=True)
        moved_images = 0
        for item in undist_images_path.iterdir():
            if item.is_file():
                shutil.move(str(item), str(images_path / item.name))
                moved_images += 1

        if moved_images == 0:
            raise RuntimeError(
                "No undistorted images were produced. "
                "Please ensure frame image filenames exactly match the names in "
                "distorted/sparse/0/images.bin (or images.txt)."
            )

        sparse_path = temp_undist_path / "sparse"
        sparse_zero_path = frame_path / "sparse" / "0"
        sparse_zero_path.mkdir(parents=True, exist_ok=True)

        for item in sparse_path.iterdir():
            if item.name == "0":
                continue
            shutil.move(str(item), str(sparse_zero_path / item.name))

        shutil.rmtree(temp_undist_path, ignore_errors=True)

        if args.resize:
            print("Copying and resizing...")
            images2_path = frame_path / "images_2"
            images2_path.mkdir(parents=True, exist_ok=True)

            for file_name in os.listdir(images_path):
                source_file = images_path / file_name
                destination_file = images2_path / file_name
                shutil.copy2(source_file, destination_file)
                print(f"Resizing {source_file} to {destination_file}")
                resize_cmd = [magick_exec, "mogrify", "-resize", "50%", str(destination_file)]
                run_command(resize_cmd)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as err:
        logging.error("Command failed with code %s: %s", err.returncode, " ".join(err.cmd))
        raise
