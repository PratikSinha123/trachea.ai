#!/usr/bin/env python3
"""Prepare a nnU-Net v2 dataset from the auto_pipelined processed_data directory.

nnU-Net v2 expects:
  Dataset001_Trachea/
    imagesTr/<scan_id>_0000.nii.gz   ← CT image (modality 0)
    labelsTr/<scan_id>.nii.gz         ← binary trachea mask (0=bg, 1=trachea)
    dataset.json

Run:
    python3 data_preparation/nnunet_dataset.py --processed-data processed_data/
"""

import argparse
import json
import os
import shutil
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_ct_nifti(scan_dir: Path) -> Path:
    """Return the CT NIfTI file for a processed scan directory.

    For fully-processed scans (e.g. LIDC-0035-main), we look for the
    original CT stored alongside the masks.  For raw LIDC scan directories
    (files dropped directly in processed_data/), we pick the .nii.gz that
    does NOT contain 'mask', 'healthy', or 'trachea' in its name.
    """
    candidates = []
    for f in sorted(scan_dir.iterdir()):
        name = f.name.lower()
        if f.is_file() and (name.endswith(".nii.gz") or name.endswith(".nii")):
            if all(skip not in name for skip in ["mask", "healthy", "trachea"]):
                candidates.append(f)
    if not candidates:
        raise FileNotFoundError(f"No CT NIfTI found in {scan_dir}")
    # Prefer .nii.gz over .nii
    gz = [c for c in candidates if c.name.endswith(".gz")]
    return gz[0] if gz else candidates[0]


def short_id(scan_dir: Path) -> str:
    """Return a short, nnU-Net-safe case identifier (max 30 chars)."""
    name = scan_dir.name
    # Already short names (e.g. LIDC-0035-main) → use as-is
    if len(name) <= 30:
        return name.replace(" ", "_")
    # Long DICOM UIDs → take last 8 digits
    parts = name.split("__")
    tail = parts[-1].replace(".", "")[-8:]
    return f"case_{tail}"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare nnU-Net v2 dataset from processed trachea scans"
    )
    parser.add_argument(
        "--processed-data", type=str, default="processed_data",
        help="Path to the processed_data directory (default: processed_data)"
    )
    parser.add_argument(
        "--output-base", type=str, default=None,
        help="nnUNet_raw base dir. Defaults to $nnUNet_raw env var or './nnUNet_raw'"
    )
    parser.add_argument(
        "--dataset-id", type=int, default=1,
        help="nnU-Net dataset ID (default: 1 → Dataset001_Trachea)"
    )
    args = parser.parse_args()

    # Resolve output root ──────────────────────────────────────────────────────
    raw_base = (
        args.output_base
        or os.environ.get("nnUNet_raw")
        or os.environ.get("nnUNet_raw_data_base")  # v1 compat
        or "nnUNet_raw"
    )
    dataset_name = f"Dataset{args.dataset_id:03d}_Trachea"
    output_root = Path(raw_base) / dataset_name

    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "imagesTr").mkdir(parents=True)
    (output_root / "labelsTr").mkdir(parents=True)

    print(f"nnU-Net raw base : {raw_base}")
    print(f"Dataset directory: {output_root}")

    # Collect cases ───────────────────────────────────────────────────────────
    processed_root = Path(args.processed_data)
    images, labels, case_ids = [], [], []

    for scan_dir in sorted(processed_root.iterdir()):
        # Only subdirectories that contain a trachea_mask.nii.gz
        if not scan_dir.is_dir():
            continue
        mask_path = scan_dir / "trachea_mask.nii.gz"
        if not mask_path.is_file():
            print(f"  [skip] {scan_dir.name} — no trachea_mask.nii.gz")
            continue

        # Find CT image
        try:
            ct_path = find_ct_nifti(scan_dir)
        except FileNotFoundError:
            # For dirs like LIDC-0035-main the original CT may be in the parent
            # directory as a loose file with the scan's prefix, or we use the
            # trachea_healthy.nii.gz as a proxy CT (contains the anatomy)
            candidates = []
            # Check for any nii.gz in the scan dir that's not mask/healthy
            for f in sorted(scan_dir.iterdir()):
                name = f.name.lower()
                if f.is_file() and name.endswith(".nii.gz") and "mask" not in name:
                    candidates.append(f)
            # Look in parent dir for a matching CT
            parent = scan_dir.parent
            scan_prefix = scan_dir.name.split("-")[0] if "-" in scan_dir.name else scan_dir.name
            for f in sorted(parent.iterdir()):
                name = f.name.lower()
                if f.is_file() and name.endswith(".nii.gz") and "mask" not in name and "healthy" not in name:
                    candidates.append(f)

            if candidates:
                ct_path = candidates[0]
                print(f"  [info] Using CT proxy: {ct_path.name}")
            else:
                print(f"  [skip] {scan_dir.name} — no CT NIfTI found anywhere")
                continue

        cid = short_id(scan_dir)
        img_dst = output_root / "imagesTr" / f"{cid}_0000.nii.gz"
        lbl_dst = output_root / "labelsTr" / f"{cid}.nii.gz"

        # Copy (convert .nii → .nii.gz on-the-fly if needed)
        if ct_path.name.endswith(".nii.gz"):
            shutil.copy2(ct_path, img_dst)
        else:
            import gzip
            with open(ct_path, "rb") as fi, gzip.open(img_dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)

        shutil.copy2(mask_path, lbl_dst)

        images.append(f"imagesTr/{cid}_0000.nii.gz")
        labels.append(f"labelsTr/{cid}.nii.gz")
        case_ids.append(cid)
        print(f"  [add]  {scan_dir.name}  →  {cid}")

    if not images:
        print("\n❌ No valid cases found. Check that processed_data/ contains "
              "subdirectories with trachea_mask.nii.gz files.")
        return

    # Write dataset.json ──────────────────────────────────────────────────────
    # nnU-Net v2 schema
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "trachea": 1},
        "numTraining": len(images),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": (
            "Trachea segmentation from chest CT. "
            "Labels: 0=background, 1=trachea lumen."
        ),
        "reference": "TracheaAI auto_pipelined",
        "licence": "private",
        "release": "1.0.0",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "training": [
            {"image": img, "label": lbl}
            for img, lbl in zip(images, labels)
        ],
    }
    with open(output_root / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n✅ Dataset ready: {output_root}")
    print(f"   Cases       : {len(images)}")
    print(f"   Case IDs    : {case_ids}")
    print()
    print("Next steps:")
    print(f"  export nnUNet_raw={raw_base}")
    print(f"  export nnUNet_preprocessed={Path(raw_base).parent / 'nnUNet_preprocessed'}")
    print(f"  export nnUNet_results={Path(raw_base).parent / 'nnUNet_results'}")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id:03d} --verify_dataset_integrity")
    print(f"  nnUNetv2_train {args.dataset_id:03d} 3d_fullres 0")


if __name__ == "__main__":
    main()
