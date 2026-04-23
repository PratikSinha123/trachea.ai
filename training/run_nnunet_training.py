#!/usr/bin/env python3
"""Convenient wrapper to launch nnU‑Net training for the Trachea task.

Assumes you have already run ``data_preparation/nnunet_dataset.py`` which
creates ``nnUNet_raw_data/Task001_Trachea``.

The script will:
  1. Detect the best device (CUDA > MPS > CPU).
  2. Set the required environment variables for nnU‑Net.
  3. Execute the ``nnUNet`` CLI to plan, preprocess and train.

Usage examples:
  # Train on the college server (CUDA GPU) – runs full‑resolution 3D U‑Net
  python3 training/run_nnunet_training.py --fold 0 --epochs 500

  # Quick test on the local M2 Mac (MPS) – 2‑epoch run for sanity check
  python3 training/run_nnunet_training.py --fold 0 --epochs 2 --device mps

The trained model will be saved under
``nnUNet_results/Task001_Trachea/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0``.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_best_device(preferred: str = None) -> str:
    """Return 'cuda', 'mps' or 'cpu' based on availability.
    If *preferred* is given (e.g. "mps"), it is returned when available.
    """
    if preferred:
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        if (
            preferred == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"
        if preferred == "cpu":
            return "cpu"
    # auto‑detect
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_cmd(cmd: list[str]):
    """Run *cmd* via subprocess, streaming output to stdout/stderr."""
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Train nnU‑Net on the Trachea dataset")
    parser.add_argument(
        "--fold", type=int, default=0, help="Fold index (0‑4). Use 0 for a quick test."
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of training epochs."
    )
    parser.add_argument(
        "--device", choices=["cuda", "mps", "cpu"], help="Force a specific device."
    )
    parser.add_argument(
        "--plan", action="store_true", help="Only run the planning step and exit."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Base directory for nnU-Net raw data (default: $HOME/.nnUNet/nnUNet_raw_data)",
    )
    args = parser.parse_args()

    # Detect device and set nnU‑Net env vars accordingly
    device = get_best_device(args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if device == "cuda" else ""
    os.environ["nnUNet_training_env"] = device  # nnU‑Net reads this for MPS support
    print(f"Detected device: {device}")

    # Set nnU-Net environment variables for data paths
    if args.data_path:
        os.environ["nnUNet_raw_data_base"] = args.data_path
        print(f"Using custom nnU-Net raw data base: {args.data_path}")
    else:
        # Use default nnU-Net path if not specified
        default_nnunet_path = str(Path.home() / ".nnUNet" / "nnUNet_raw_data")
        os.environ["nnUNet_raw_data_base"] = default_nnunet_path
        print(f"Using default nnU-Net raw data base: {default_nnunet_path}")

    task_name = "Task001_Trachea"
    # 1️⃣ Planning & preprocessing
    run_cmd(["nnUNet_plan_and_preprocess", "-t", task_name, "-pl3", "d", "-tl", "2"])

    if args.plan:
        print("Planning completed. Exiting as requested.")
        return

    # 2️⃣ Training – we use the default trainer (nnUNetTrainerV2)
    run_cmd(
        [
            "nnUNet_train",
            "3d_fullres",
            "-t",
            task_name,
            "-f",
            str(args.fold),
            "--npz",
            "--disable_postprocessing_on_folds",
            f"--max_num_epochs={args.epochs}",
        ]
    )

    print("Training finished. Model checkpoints are in nnUNet_results/...")


if __name__ == "__main__":
    main()
