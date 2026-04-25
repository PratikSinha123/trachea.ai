#!/usr/bin/env python3
"""nnU-Net v2 training launcher for the Trachea segmentation task.

Usage examples
--------------
# Full training on college GPU server (fold 0, 500 epochs):
    python3 training/run_nnunet_training.py --fold 0 --epochs 500

# Quick sanity check (2 epochs, local Mac):
    python3 training/run_nnunet_training.py --fold 0 --epochs 2

# Plan + preprocess only (no training):
    python3 training/run_nnunet_training.py --plan-only

# Train ALL folds for final deployment model:
    python3 training/run_nnunet_training.py --fold all --epochs 500

After training, run inference with:
    python3 training/predict.py --input scan.nii.gz --scan-id MyPatient
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def detect_device(preferred: str = None) -> str:
    """Return 'cuda', 'mps' or 'cpu' based on availability."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if preferred == "cpu":
        return "cpu"
    # Auto-detect
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_cmd(cmd: list[str], env: dict = None):
    """Stream *cmd* output to stdout; exit on failure."""
    merged_env = {**os.environ, **(env or {})}
    print("\n▶", " ".join(cmd))
    result = subprocess.run(cmd, env=merged_env)
    if result.returncode != 0:
        sys.exit(f"\n❌ Command failed (exit code {result.returncode})")


def ensure_env_vars(base: str) -> dict[str, str]:
    """Return the 3 required nnU-Net env vars, derived from *base*."""
    base_path = Path(base).expanduser().resolve()
    return {
        "nnUNet_raw":          str(base_path / "nnUNet_raw"),
        "nnUNet_preprocessed": str(base_path / "nnUNet_preprocessed"),
        "nnUNet_results":      str(base_path / "nnUNet_results"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="nnU-Net v2 training wrapper for Trachea segmentation"
    )
    parser.add_argument(
        "--fold", default="0",
        help="Fold to train: 0-4, or 'all' to train all 5 folds (default: 0)"
    )
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Max training epochs (default: 500). Use 5 for a quick sanity check."
    )
    parser.add_argument(
        "--device", choices=["cuda", "mps", "cpu"], default=None,
        help="Force a device (default: auto-detect cuda > mps > cpu)"
    )
    parser.add_argument(
        "--dataset-id", type=int, default=1,
        help="nnU-Net dataset ID (default: 1 → Dataset001_Trachea)"
    )
    parser.add_argument(
        "--base-dir", type=str, default=None,
        help="Base directory for all nnU-Net folders. "
             "Defaults to $nnUNet_raw parent or ./nnunet_workspace"
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Only run planning + preprocessing, skip training."
    )
    parser.add_argument(
        "--trainer", type=str, default="nnUNetTrainer",
        help="Trainer class (default: nnUNetTrainer). "
             "Use nnUNetTrainer_5epochs for a quick test."
    )
    args = parser.parse_args()

    # Resolve workspace base ──────────────────────────────────────────────────
    if args.base_dir:
        base = args.base_dir
    elif os.environ.get("nnUNet_raw"):
        # Already configured in env — derive base from nnUNet_raw parent
        base = str(Path(os.environ["nnUNet_raw"]).parent)
    else:
        base = str(Path.cwd() / "nnunet_workspace")

    env_vars = ensure_env_vars(base)
    # Create directories
    for p in env_vars.values():
        Path(p).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  nnU-Net v2 — Trachea Segmentation Training")
    print("=" * 60)
    for k, v in env_vars.items():
        print(f"  {k} = {v}")

    device = detect_device(args.device)
    print(f"  Device       = {device}")
    print(f"  Dataset ID   = {args.dataset_id:03d}")
    print(f"  Fold         = {args.fold}")
    print(f"  Epochs       = {args.epochs}")
    print("=" * 60)

    dataset_id_str = str(args.dataset_id)

    # ── Step 1: plan + preprocess ─────────────────────────────────────────────
    print("\n[1/2] Planning & preprocessing …")
    run_cmd(
        ["nnUNetv2_plan_and_preprocess", "-d", dataset_id_str,
         "--verify_dataset_integrity", "-c", "3d_fullres"],
        env=env_vars
    )

    if args.plan_only:
        print("\n✅ Planning complete. Run without --plan-only to start training.")
        return

    # ── Step 2: train ─────────────────────────────────────────────────────────
    if args.fold == "all":
        folds_to_train = ["0", "1", "2", "3", "4"]
    else:
        folds_to_train = [args.fold]

    for fold in folds_to_train:
        print(f"\n[2/2] Training fold {fold} …")
        train_cmd = [
            "nnUNetv2_train",
            dataset_id_str,
            "3d_fullres",
            fold,
            "-tr", args.trainer,
            "--npz",                    # save softmax outputs (needed for ensemble)
        ]
        if args.epochs != 1000:        # nnUNet default is 1000; override if different
            train_cmd += ["-num_epochs_per_val", "1"]  # validate every epoch (quick runs)

        # For MPS/CPU — disable AMP (Automatic Mixed Precision, CUDA-only)
        if device != "cuda":
            train_cmd.append("--disable_checkpointing")
            os.environ["nnUNet_compile"] = "F"

        run_cmd(train_cmd, env={**env_vars, "CUDA_VISIBLE_DEVICES": "0" if device == "cuda" else ""})

    print("\n✅ Training complete!")
    print("   Model saved to:", env_vars["nnUNet_results"])
    print()
    print("To predict on a new CT scan, run:")
    print(f"  python3 training/predict.py --input <ct.nii.gz> --scan-id <PatientID>")
    print(f"  python3 training/predict.py --input <ct.nii.gz> --scan-id <PatientID> --fold {args.fold}")


if __name__ == "__main__":
    main()
