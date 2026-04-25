#!/usr/bin/env python3
"""Run nnU-Net v2 inference on a new CT scan and feed it into the Trachea AI pipeline.

This script:
  1. Takes a raw CT .nii.gz file as input
  2. Runs nnUNetv2_predict to generate a trachea segmentation mask
  3. Saves the mask into processed_data/<scan-id>/trachea_mask.nii.gz
  4. Triggers auto_pipeline.py to compute measurements and 3D meshes
  5. The web viewer can then load the result by scan-id

Usage
-----
# Basic prediction + full pipeline:
    python3 training/predict.py --input /path/to/ct.nii.gz --scan-id Patient_001

# Predict only (skip auto_pipeline):
    python3 training/predict.py --input /path/to/ct.nii.gz --scan-id Patient_001 --no-pipeline

# Use a specific fold's model:
    python3 training/predict.py --input ct.nii.gz --scan-id P001 --fold 0

Requirements
------------
    nnunetv2 must be installed: pip3 install nnunetv2
    nnUNet_raw / nnUNet_preprocessed / nnUNet_results must be set (or --base-dir given)
    The model must already be trained (see training/run_nnunet_training.py)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def resolve_env(base_dir: str | None) -> dict[str, str]:
    """Build nnU-Net path environment from *base_dir* or existing env vars."""
    if base_dir:
        base = Path(base_dir).expanduser().resolve()
        return {
            "nnUNet_raw":          str(base / "nnUNet_raw"),
            "nnUNet_preprocessed": str(base / "nnUNet_preprocessed"),
            "nnUNet_results":      str(base / "nnUNet_results"),
        }
    # Fall back to env
    env = {}
    for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        v = os.environ.get(k)
        if not v:
            # Try to derive from nnunet_workspace if it exists
            ws = Path.cwd() / "nnunet_workspace"
            mapping = {
                "nnUNet_raw":          str(ws / "nnUNet_raw"),
                "nnUNet_preprocessed": str(ws / "nnUNet_preprocessed"),
                "nnUNet_results":      str(ws / "nnUNet_results"),
            }
            v = mapping[k]
        env[k] = v
    return env


def run_cmd(cmd: list[str], env: dict = None, check: bool = True) -> int:
    merged = {**os.environ, **(env or {})}
    print("\n▶", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, env=merged)
    if check and result.returncode != 0:
        sys.exit(f"\n❌ Command failed (exit code {result.returncode})")
    return result.returncode


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="nnU-Net v2 inference → Trachea AI pipeline"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the input CT scan (.nii.gz or .nii)"
    )
    parser.add_argument(
        "--scan-id", required=True,
        help="Unique name for this scan (used as directory name in processed_data/)"
    )
    parser.add_argument(
        "--dataset-id", type=int, default=1,
        help="nnU-Net dataset ID (default: 1)"
    )
    parser.add_argument(
        "--fold", type=str, default="0",
        help="Which fold's model to use for prediction (default: 0). "
             "Use 'all' to ensemble all trained folds."
    )
    parser.add_argument(
        "--base-dir", type=str, default=None,
        help="Base directory for nnU-Net workspaces. Defaults to ./nnunet_workspace"
    )
    parser.add_argument(
        "--processed-data", type=str, default="processed_data",
        help="Path to processed_data directory (default: processed_data)"
    )
    parser.add_argument(
        "--no-pipeline", action="store_true",
        help="Skip auto_pipeline.py after prediction (only save the mask)"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default=None,
        help="Device for nnU-Net inference (default: auto)"
    )
    args = parser.parse_args()

    # ── Validate input ────────────────────────────────────────────────────────
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        sys.exit(f"❌ Input file not found: {input_path}")

    scan_id = args.scan_id.replace(" ", "_")
    env_vars = resolve_env(args.base_dir)

    print("=" * 60)
    print("  nnU-Net v2 — Trachea Prediction")
    print("=" * 60)
    print(f"  Input scan : {input_path}")
    print(f"  Scan ID    : {scan_id}")
    print(f"  Dataset ID : {args.dataset_id:03d}")
    print(f"  Fold       : {args.fold}")
    print("=" * 60)

    # ── Step 1: stage input in a temp nnUNet imagesTs dir ────────────────────
    with tempfile.TemporaryDirectory(prefix="nnunet_pred_") as tmp:
        tmp_path = Path(tmp)
        images_ts = tmp_path / "imagesTs"
        images_ts.mkdir()
        output_dir = tmp_path / "predictions"
        output_dir.mkdir()

        # nnU-Net expects filename pattern: <case>_0000.nii.gz
        case_name = f"{scan_id}_0000"
        if input_path.name.endswith(".nii.gz"):
            staged = images_ts / f"{case_name}.nii.gz"
            shutil.copy2(input_path, staged)
        else:
            # Convert .nii → .nii.gz
            import gzip
            staged = images_ts / f"{case_name}.nii.gz"
            with open(input_path, "rb") as fi, gzip.open(staged, "wb") as fo:
                shutil.copyfileobj(fi, fo)

        print(f"\n[1/3] Staged input → {staged}")

        # ── Step 2: nnUNetv2_predict ──────────────────────────────────────────
        print("\n[2/3] Running nnU-Net prediction …")
        predict_cmd = [
            "nnUNetv2_predict",
            "-d", str(args.dataset_id),
            "-c", "3d_fullres",
            "-i", str(images_ts),
            "-o", str(output_dir),
            "-f", args.fold,
            "--save_probabilities",       # save softmax (optional, for review)
        ]
        if args.device:
            predict_cmd += ["--device", args.device]

        run_cmd(predict_cmd, env=env_vars)

        # ── Step 3: copy predicted mask to processed_data ─────────────────────
        predicted_mask = output_dir / f"{scan_id}.nii.gz"
        if not predicted_mask.exists():
            # Try alternative naming
            candidates = list(output_dir.glob("*.nii.gz"))
            if candidates:
                predicted_mask = candidates[0]
                print(f"  Using mask: {predicted_mask.name}")
            else:
                sys.exit("❌ nnU-Net did not produce a mask. Check model + environment.")

        out_scan_dir = Path(args.processed_data) / scan_id
        out_scan_dir.mkdir(parents=True, exist_ok=True)
        mask_dst = out_scan_dir / "trachea_mask.nii.gz"
        shutil.copy2(predicted_mask, mask_dst)
        # Also copy the CT image so the pipeline can access it
        ct_dst = out_scan_dir / f"{scan_id}_ct.nii.gz"
        shutil.copy2(staged, ct_dst)

        print(f"\n[3/3] Saved mask → {mask_dst}")

    # ── Step 4 (optional): run auto_pipeline ──────────────────────────────────
    if not args.no_pipeline:
        print("\n[4/3] Running Trachea AI pipeline …")
        run_cmd(
            [sys.executable, "auto_pipeline.py", "--scan-dir", str(out_scan_dir),
             "--scan-id", scan_id],
            check=False  # non-fatal if pipeline fails
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Prediction complete!")
    print(f"  Scan ID    : {scan_id}")
    print(f"  Mask saved : {mask_dst}")
    print()
    print("  Open in web viewer:")
    print(f"    http://localhost:8000  → select '{scan_id}' from the dropdown")
    print("=" * 60)


if __name__ == "__main__":
    main()
