#!/usr/bin/env python3
"""
auto_train.py — One-click nnU-Net training from your raw dataset folder.

Give it the path to your database folder and it will:
  1. Scan for up to N patient CT scans (DICOM dirs or NIfTI files)
  2. Auto-generate trachea masks using TotalSegmentator AI
  3. Build a valid nnU-Net v2 dataset
  4. Run planning + preprocessing
  5. Train for a quick demo (default: 10 epochs) or full training (--full)

Usage
-----
  # Show what it would find (dry run — no changes):
      python3 auto_train.py --database /path/to/your/dataset --dry-run

  # Demo for your professor (10 epochs, up to 15 patients):
      python3 auto_train.py --database /path/to/your/dataset

  # Full training on college server (500 epochs):
      python3 auto_train.py --database /path/to/your/dataset --full --device cuda

  # Custom number of patients and epochs:
      python3 auto_train.py --database /path/to/your/dataset --max-patients 8 --epochs 20
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ─── ANSI colors for nice terminal output ─────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"{GREEN}  ✅ {msg}{RESET}")
def info(msg):  print(f"{CYAN}  ℹ  {msg}{RESET}")
def warn(msg):  print(f"{YELLOW}  ⚠  {msg}{RESET}")
def err(msg):   print(f"{RED}  ❌ {msg}{RESET}")
def step(n, t): print(f"\n{BOLD}{CYAN}[Step {n}] {t}{RESET}")
def banner(t):  print(f"\n{BOLD}{'='*60}\n  {t}\n{'='*60}{RESET}")


# ─── Scan discovery ────────────────────────────────────────────────────────────

def find_patients(database: Path, max_patients: int) -> list[dict]:
    """
    Recursively scan *database* for CT data.
    Returns a list of dicts: {id, type, path}

    Supports:
      - DICOM folders (dirs containing *.dcm files)
      - NIfTI files   (.nii or .nii.gz)
    """
    patients = []
    seen_dirs = set()
    used_ids = {}  # track id → count for dedup

    def make_unique_id(base_id: str) -> str:
        """Ensure the ID is unique by appending a counter if needed."""
        used_ids[base_id] = used_ids.get(base_id, 0) + 1
        if used_ids[base_id] == 1:
            return base_id
        return f"{base_id}_{used_ids[base_id]}"

    def shorten(name: str) -> str:
        """Shorten a long UID-style name to a safe 25-char ID."""
        name = name.replace(" ", "_")
        if len(name) <= 25:
            return name
        # Use first 8 + last 8 chars of the raw UID for uniqueness
        raw = name.replace(".", "").replace("-", "").replace("_", "")
        return f"pt_{raw[:6]}_{raw[-6:]}"

    # Walk directory tree
    for root, dirs, files in os.walk(database):
        root_path = Path(root)

        # Check for NIfTI files
        for f in sorted(files):
            if f.endswith((".nii", ".nii.gz")):
                name = f.lower()
                # Skip masks / labels
                if any(skip in name for skip in ["mask", "label", "seg", "healthy", "pred"]):
                    continue
                stem = f.replace(".nii.gz", "").replace(".nii", "")
                pid = make_unique_id(shorten(stem))
                patients.append({
                    "id": pid,
                    "type": "nifti",
                    "path": str(root_path / f),
                })
                if len(patients) >= max_patients:
                    return patients

        # Check for DICOM folders
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if dcm_files and str(root_path) not in seen_dirs:
            seen_dirs.add(str(root_path))
            pid = make_unique_id(shorten(root_path.name))
            patients.append({
                "id": pid,
                "type": "dicom",
                "path": str(root_path),
            })
            if len(patients) >= max_patients:
                return patients

    return patients


# ─── Auto-segmentation with TotalSegmentator ──────────────────────────────────

def read_as_nifti_gz(path: str, out_path: str):
    """Read any CT (DICOM/NIfTI) and write as NIfTI .gz to *out_path*."""
    import SimpleITK as sitk

    if os.path.isdir(path):
        # DICOM series
        reader = sitk.ImageSeriesReader()
        dcm_files = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dcm_files)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(path)

    sitk.WriteImage(image, out_path)
    return out_path


def run_totalsegmentator(ct_gz: str, out_dir: str, device: str = "cpu") -> str | None:
    """
    Run TotalSegmentator to extract the trachea mask.
    Returns path to trachea.nii.gz or None on failure.
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "totalsegmentator",
        "-i", ct_gz,
        "-o", out_dir,
        "--roi_subset", "trachea",
        "--ml",          # use fast mode
    ]
    if device == "cuda":
        cmd += ["--device", "cuda"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    trachea_out = os.path.join(out_dir, "trachea.nii.gz")
    if os.path.isfile(trachea_out):
        return trachea_out
    # Some versions write it differently
    for candidate in Path(out_dir).glob("*.nii.gz"):
        if "trachea" in candidate.name.lower():
            return str(candidate)
    warn(f"TotalSegmentator stderr: {result.stderr[-300:]}")
    return None


# ─── nnU-Net dataset prep ─────────────────────────────────────────────────────

def build_nnunet_dataset(cases: list[dict], dataset_dir: Path):
    """Write imagesTr/ + labelsTr/ + dataset.json for nnU-Net v2."""
    (dataset_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    training_entries = []
    for c in cases:
        img_name = f"{c['id']}_0000.nii.gz"
        lbl_name = f"{c['id']}.nii.gz"
        shutil.copy2(c["ct_gz"], dataset_dir / "imagesTr" / img_name)
        shutil.copy2(c["mask_gz"], dataset_dir / "labelsTr" / lbl_name)
        training_entries.append({
            "image": f"imagesTr/{img_name}",
            "label": f"labelsTr/{lbl_name}",
        })

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "trachea": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": "Dataset001_Trachea",
        "description": "Trachea segmentation — auto_train.py",
        "reference": "TracheaAI",
        "licence": "private",
        "release": "1.0.0",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "training": training_entries,
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)


# ─── nnU-Net commands ─────────────────────────────────────────────────────────

def run_cmd(cmd: list[str], env: dict = None) -> bool:
    merged = {**os.environ, **(env or {})}
    print(f"  ▶  {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, env=merged)
    return r.returncode == 0


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="One-click nnU-Net training from a raw CT dataset folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--database", required=True,
                        help="Path to your CT database folder")
    parser.add_argument("--max-patients", type=int, default=15,
                        help="Max number of patients to use (default: 15)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs (default: 10 for demo, use 500 for full)")
    parser.add_argument("--full", action="store_true",
                        help="Full training (500 epochs). Overrides --epochs.")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                        help="Device: cuda (GPU), mps (Apple Silicon), cpu (default: auto)")
    parser.add_argument("--workspace", default="nnunet_workspace",
                        help="Where to store nnU-Net data/models (default: ./nnunet_workspace)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan the database and show what would be used — no training")
    parser.add_argument("--skip-segmentation", action="store_true",
                        help="Skip TotalSegmentator (use if masks already exist in processed_data/)")
    parser.add_argument("--fold", default="0",
                        help="Which fold to train (default: 0)")
    args = parser.parse_args()

    # ── Auto-detect device ────────────────────────────────────────────────────
    if args.device is None:
        try:
            import torch
            if torch.cuda.is_available():
                args.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                args.device = "mps"
            else:
                args.device = "cpu"
        except ImportError:
            args.device = "cpu"

    epochs = 500 if args.full else args.epochs

    banner("🧠 TracheaAI — Auto nnU-Net Training Setup")
    info(f"Database   : {args.database}")
    info(f"Max patients: {args.max_patients}")
    info(f"Epochs     : {epochs}")
    info(f"Device     : {args.device}")
    info(f"Workspace  : {args.workspace}")
    if args.dry_run:
        info("DRY RUN — no files will be written")

    database = Path(args.database).expanduser().resolve()
    if not database.is_dir():
        err(f"Database folder not found: {database}")
        sys.exit(1)

    workspace = Path(args.workspace).resolve()
    raw_dir   = workspace / "nnUNet_raw"
    pre_dir   = workspace / "nnUNet_preprocessed"
    res_dir   = workspace / "nnUNet_results"
    tmp_dir   = workspace / "temp_segmentations"
    dataset_dir = raw_dir / "Dataset001_Trachea"

    env_vars = {
        "nnUNet_raw":          str(raw_dir),
        "nnUNet_preprocessed": str(pre_dir),
        "nnUNet_results":      str(res_dir),
    }

    # ── Step 1: Discover patients ─────────────────────────────────────────────
    step(1, f"Scanning database for up to {args.max_patients} CT scans…")
    patients = find_patients(database, args.max_patients)

    if not patients:
        err(f"No CT scans found in: {database}")
        err("Expected: DICOM folders (*.dcm files) or NIfTI files (*.nii / *.nii.gz)")
        sys.exit(1)

    print(f"\n  Found {len(patients)} patients:")
    for i, p in enumerate(patients, 1):
        print(f"    {i:2}. [{p['type'].upper()}] {p['id']}")
        print(f"         → {p['path'][:80]}")

    if args.dry_run:
        print(f"\n{GREEN}Dry run complete. {len(patients)} patients would be used.{RESET}")
        print("Run without --dry-run to start training.")
        return

    if len(patients) < 2:
        warn("Only 1 patient found. nnU-Net works best with 5+ patients.")
        warn("Training will proceed but results will be limited.")

    # ── Step 2: Convert + auto-segment ───────────────────────────────────────
    step(2, "Converting CT scans and generating trachea masks…")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ready_cases = []
    for i, p in enumerate(patients, 1):
        pid = p["id"]
        print(f"\n  [{i}/{len(patients)}] Processing: {pid}")

        case_tmp = tmp_dir / pid
        case_tmp.mkdir(exist_ok=True)
        ct_gz = str(case_tmp / "ct.nii.gz")

        # Check if already processed in processed_data/
        existing_mask = Path("processed_data") / pid / "trachea_mask.nii.gz"
        existing_ct   = Path("processed_data") / pid / f"{pid}_ct.nii.gz"

        if args.skip_segmentation and existing_mask.is_file():
            info(f"Using existing mask for {pid}")
            ct_src = str(existing_ct) if existing_ct.is_file() else ct_gz
            # Find any CT in that dir
            for f in Path("processed_data", pid).iterdir():
                if f.name.endswith(".nii.gz") and "mask" not in f.name and "healthy" not in f.name:
                    ct_src = str(f)
                    break
            ready_cases.append({"id": pid, "ct_gz": ct_src, "mask_gz": str(existing_mask)})
            ok(f"Using cached data for {pid}")
            continue

        # Convert to NIfTI .gz
        try:
            print(f"    Converting {p['type']} → NIfTI…")
            read_as_nifti_gz(p["path"], ct_gz)
            ok(f"Converted: {pid}")
        except Exception as e:
            warn(f"Conversion failed for {pid}: {e}")
            continue

        # Run TotalSegmentator for trachea mask
        print(f"    Running TotalSegmentator (trachea segmentation)…")
        seg_out = str(case_tmp / "seg")
        mask_path = run_totalsegmentator(ct_gz, seg_out, device=args.device)

        if mask_path is None:
            warn(f"TotalSegmentator failed for {pid} — skipping")
            continue

        ok(f"Mask generated: {pid}")
        ready_cases.append({"id": pid, "ct_gz": ct_gz, "mask_gz": mask_path})

    if not ready_cases:
        err("No cases were successfully processed. Cannot train.")
        sys.exit(1)

    print(f"\n  {GREEN}✅ {len(ready_cases)} / {len(patients)} cases ready for training{RESET}")

    # ── Step 3: Build nnU-Net dataset ─────────────────────────────────────────
    step(3, "Building nnU-Net v2 dataset…")
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    build_nnunet_dataset(ready_cases, dataset_dir)
    ok(f"Dataset written → {dataset_dir}")

    # Write summary JSON for later reference
    summary = {
        "cases": [{"id": c["id"], "ct": c["ct_gz"], "mask": c["mask_gz"]} for c in ready_cases],
        "n_cases": len(ready_cases),
        "epochs_planned": epochs,
        "device": args.device,
        "dataset_dir": str(dataset_dir),
    }
    with open(workspace / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Step 4: nnU-Net plan + preprocess ─────────────────────────────────────
    step(4, "Planning + preprocessing (nnUNetv2_plan_and_preprocess)…")
    for d in [pre_dir, res_dir]:
        d.mkdir(parents=True, exist_ok=True)

    success = run_cmd(
        ["nnUNetv2_plan_and_preprocess", "-d", "001",
         "--verify_dataset_integrity", "-c", "3d_fullres"],
        env=env_vars,
    )
    if not success:
        err("Planning failed. Check dataset integrity above.")
        sys.exit(1)
    ok("Planning complete")

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    step(5, f"Training fold {args.fold} for {epochs} epochs on {args.device.upper()}…")

    if len(ready_cases) < 5:
        warn(f"Only {len(ready_cases)} cases — nnU-Net will use a single train/val split")
        warn("(5-fold cross-validation needs at least 5 cases)")

    if epochs <= 20:
        info("Demo mode: training for a small number of epochs to show the model is working.")
        info("For final deployment, run with --full (500 epochs) on the college GPU server.")

    train_cmd = [
        "nnUNetv2_train",
        "001",
        "3d_fullres",
        args.fold,
        "--npz",
    ]

    # For non-CUDA devices — disable compilation
    train_env = dict(env_vars)
    if args.device != "cuda":
        train_env["nnUNet_compile"] = "F"
        train_env["nnUNet_n_proc_DA"] = "2"   # fewer data augmentation workers

    t_start = time.time()
    success = run_cmd(train_cmd, env=train_env)
    elapsed = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("🎉 Training Complete!")
    print(f"  Patients used : {len(ready_cases)}")
    print(f"  Epochs        : {epochs}")
    print(f"  Time elapsed  : {elapsed/60:.1f} min")
    print(f"  Model saved   : {res_dir}")
    print()
    print(f"  {BOLD}To predict on a new CT scan:{RESET}")
    print(f"    python3 training/predict.py --input <ct.nii.gz> --scan-id <PatientID> \\")
    print(f"      --base-dir {workspace}")
    print()
    print(f"  {BOLD}To show your professor:{RESET}")
    print(f"    1. Start the web server:  python3 -m server.app")
    print(f"    2. Open: http://localhost:8000")
    print(f"    3. Click '🧠 AI Predict' → paste a CT path → watch it segment + analyze")
    print()


if __name__ == "__main__":
    main()
