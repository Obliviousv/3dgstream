#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import json

import numpy as np
import torch
import torch.nn.functional as F

try:
    import commentjson as ctjs
except ImportError:
    ctjs = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ntc import NeuralTransformationCache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm up NTC from init_3dgs point cloud and auto-computed AABB"
    )
    parser.add_argument(
        "--postfixes",
        nargs="+",
        default=["F_4"],
        help="Cache config postfixes used as configs/cache/cache_<postfix>.json",
    )
    parser.add_argument(
        "--ntc-conf-paths",
        nargs="+",
        default=None,
        help="Explicit list of NTC config paths; overrides --postfixes when provided",
    )
    parser.add_argument(
        "--save-paths",
        nargs="+",
        default=None,
        help="Output .pth paths; defaults to ntc/<scene>_ntc_params_<postfix>.pth",
    )
    parser.add_argument(
        "--scene-name",
        default="scene",
        help="Scene name used for default output filenames",
    )
    parser.add_argument(
        "--init-3dgs-dir",
        type=Path,
        default=PROJECT_ROOT / "test" / "flame_steak_suite" / "flame_steak_init",
        help="Path to init_3dgs directory containing point_cloud/iteration_x/point_cloud.ply",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=15000,
        help="Iteration id under point_cloud/iteration_<id> (ignored if --pcd-path is set)",
    )
    parser.add_argument(
        "--pcd-path",
        type=Path,
        default=None,
        help="Direct path to point_cloud.ply; overrides --init-3dgs-dir/--iteration",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=86.6,
        help="Percentile used to compute AABB from point cloud quantiles",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3000,
        help="Warm-up optimization iterations",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for NTC warm-up",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.01,
        help="Noise scale added to normalized xyz during warm-up",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print loss every N iterations",
    )
    parser.add_argument(
        "--only-mlp",
        action="store_true",
        help="Use tcnn.Network without input encoding",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for warm-up, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--quat-loss-dtype",
        choices=["float32", "float64"],
        default="float64",
        help=(
            "Dtype used in quaternion loss computation. "
            "Issue #16 recommends float64 to avoid NaN in some environments."
        ),
    )
    parser.add_argument(
        "--required-conda-env",
        default="3dgstream",
        help="Expected conda environment name for running this script.",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip conda environment name check.",
    )
    return parser.parse_args()


def ensure_expected_env(args: argparse.Namespace) -> None:
    if args.skip_env_check:
        return

    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env != args.required_conda_env:
        raise EnvironmentError(
            "cache_warmup.py must run in conda env "
            f"'{args.required_conda_env}', but current env is '{current_env or 'unknown'}'.\n"
            "Use: conda run -n 3dgstream python scripts/cache_warmup.py ..."
        )


def resolve_pcd_path(args: argparse.Namespace) -> Path:
    if args.pcd_path is not None:
        return args.pcd_path.resolve()
    return (
        args.init_3dgs_dir.resolve()
        / "point_cloud"
        / f"iteration_{args.iteration}"
        / "point_cloud.ply"
    )


def resolve_conf_paths(args: argparse.Namespace) -> list[Path]:
    if args.ntc_conf_paths:
        return [Path(p).resolve() for p in args.ntc_conf_paths]
    return [
        (PROJECT_ROOT / "configs" / "cache" / f"cache_{postfix}.json").resolve()
        for postfix in args.postfixes
    ]


def resolve_save_paths(args: argparse.Namespace, conf_paths: list[Path]) -> list[Path]:
    if args.save_paths:
        paths = [Path(p).resolve() for p in args.save_paths]
        if len(paths) != len(conf_paths):
            raise ValueError("--save-paths count must match config count")
        return paths

    out_dir = (PROJECT_ROOT / "ntc").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for conf in conf_paths:
        stem = conf.stem
        suffix = stem.split("cache_")[-1] if "cache_" in stem else stem
        names.append(out_dir / f"{args.scene_name}_ntc_params_{suffix}.pth")
    return names


def fetch_xyz(path: Path, device: torch.device) -> torch.Tensor:
    from plyfile import PlyData

    plydata = PlyData.read(str(path))
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    return torch.tensor(xyz, dtype=torch.float32, device=device)


def get_xyz_bound(xyz: torch.Tensor, percentile: float = 86.6) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 < percentile < 100.0):
        raise ValueError("--percentile must be in (0, 100)")
    half_percentile = (100.0 - percentile) / 200.0
    xyz_bound_min = torch.quantile(xyz, half_percentile, dim=0)
    xyz_bound_max = torch.quantile(xyz, 1.0 - half_percentile, dim=0)
    return xyz_bound_min, xyz_bound_max


def get_contracted_xyz(
    xyz: torch.Tensor,
    xyz_bound_min: torch.Tensor,
    xyz_bound_max: torch.Tensor,
) -> torch.Tensor:
    denom = torch.clamp(xyz_bound_max - xyz_bound_min, min=1e-8)
    normalized_xyz = (xyz - xyz_bound_min) / denom
    return normalized_xyz


def quaternion_loss(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.dim() == 1:
        q1 = q1.unsqueeze(0)
    if q2.dim() == 1:
        q2 = q2.unsqueeze(0)
    if q2.shape[0] == 1 and q1.shape[0] > 1:
        q2 = q2.expand(q1.shape[0], -1)

    q1 = F.normalize(q1, dim=1)
    q2 = F.normalize(q2, dim=1)
    cos_theta = F.cosine_similarity(q1, q2, dim=1)
    cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)
    return 1 - torch.pow(cos_theta, 2).mean()


def l1loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs(network_output - gt).mean()


def cache_loss(resi: torch.Tensor, d_xyz_gt: torch.Tensor, d_rot_gt: torch.Tensor, dummy_gt: torch.Tensor) -> torch.Tensor:
    masked_d_xyz = resi[:, :3]
    masked_d_rot = resi[:, 3:7]
    masked_dummy = resi[:, 7:8]
    loss_xyz = l1loss(masked_d_xyz, d_xyz_gt)
    loss_rot = quaternion_loss(masked_d_rot, d_rot_gt)
    loss_dummy = l1loss(masked_dummy, dummy_gt)
    return loss_xyz + loss_rot + loss_dummy


def select_quat_loss_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    return torch.float64


def build_ntc(ntc_conf: dict, only_mlp: bool, device: torch.device) -> torch.nn.Module:
    try:
        import tinycudann as tcnn
    except ImportError as err:
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        hint = (
            "Failed to import tinycudann. This is often caused by loading the system "
            "libstdc++ instead of the conda one.\n"
            "Try running with:\n"
            "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\n"
            "python scripts/cache_warmup.py ..."
        )
        if conda_prefix:
            hint += f"\nCurrent CONDA_PREFIX: {conda_prefix}"
        raise ImportError(f"{hint}\nOriginal error: {err}") from err

    if only_mlp:
        return tcnn.Network(
            n_input_dims=3,
            n_output_dims=8,
            network_config=ntc_conf["network"],
        ).to(device)
    return tcnn.NetworkWithInputEncoding(
        n_input_dims=3,
        n_output_dims=8,
        encoding_config=ntc_conf["encoding"],
        network_config=ntc_conf["network"],
    ).to(device)


def main() -> None:
    args = parse_args()
    ensure_expected_env(args)

    device = torch.device(args.device)
    pcd_path = resolve_pcd_path(args)
    conf_paths = resolve_conf_paths(args)
    save_paths = resolve_save_paths(args, conf_paths)

    if not pcd_path.is_file():
        raise FileNotFoundError(f"Point cloud not found: {pcd_path}")
    for conf_path in conf_paths:
        if not conf_path.is_file():
            raise FileNotFoundError(f"NTC config not found: {conf_path}")

    xyz = fetch_xyz(pcd_path, device)
    xyz_bound_min, xyz_bound_max = get_xyz_bound(xyz, args.percentile)
    normalized_xyz = get_contracted_xyz(xyz, xyz_bound_min, xyz_bound_max)

    mask = (normalized_xyz >= 0) & (normalized_xyz <= 1)
    mask = mask.all(dim=1)
    ntc_inputs = normalized_xyz[mask]
    if ntc_inputs.numel() == 0:
        raise RuntimeError("No points remain inside computed AABB; adjust --percentile")

    noisy_inputs = ntc_inputs + args.noise_scale * torch.rand_like(ntc_inputs)
    quat_loss_dtype = select_quat_loss_dtype(args.quat_loss_dtype)

    d_xyz_gt = torch.tensor([0.0, 0.0, 0.0], device=device)
    d_rot_gt = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=quat_loss_dtype)
    dummy_gt = torch.tensor([1.0], device=device)

    print(f"Using point cloud: {pcd_path}")
    print(f"AABB min: {xyz_bound_min.detach().cpu().numpy().tolist()}")
    print(f"AABB max: {xyz_bound_max.detach().cpu().numpy().tolist()}")
    print(f"Input points: {xyz.shape[0]}, in-AABB points: {ntc_inputs.shape[0]}")
    print(f"Quaternion loss dtype: {args.quat_loss_dtype}")

    for idx, conf_path in enumerate(conf_paths):
        with open(conf_path, "r", encoding="utf-8") as conf_file:
            if ctjs is not None:
                ntc_conf = ctjs.load(conf_file)
            else:
                ntc_conf = json.load(conf_file)

        ntc = build_ntc(ntc_conf, args.only_mlp, device)
        ntc_optimizer = torch.optim.Adam(ntc.parameters(), lr=args.lr)

        print(f"Warm-up NTC [{idx + 1}/{len(conf_paths)}]: {conf_path}")
        for iteration in range(args.iterations):
            ntc_inputs_w_noisy = torch.cat(
                [noisy_inputs, ntc_inputs, torch.rand_like(ntc_inputs)], dim=0
            )
            ntc_output = ntc(ntc_inputs_w_noisy)
            ntc_output_for_loss = ntc_output.to(quat_loss_dtype)
            loss = cache_loss(ntc_output_for_loss, d_xyz_gt, d_rot_gt, dummy_gt)

            if iteration % args.log_interval == 0:
                print(f"iter={iteration:04d}, loss={float(loss):.8f}")

            loss.backward()
            ntc_optimizer.step()
            ntc_optimizer.zero_grad(set_to_none=True)

        wrapped_ntc = NeuralTransformationCache(ntc, xyz_bound_min, xyz_bound_max)

        save_path = save_paths[idx]
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(wrapped_ntc.state_dict(), save_path)
        print(f"Saved warmed NTC: {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
