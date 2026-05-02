import argparse
import os
from pathlib import Path

import SimpleITK as sitk


DEFAULT_INPUT_ROOT = Path("/Volumes/Untitled/TCIA_LIDC-IDRI_20200921/lidc_idri")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "processed_data"


def dicom_to_nifti(dicom_folder, output_base_path):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder) or []

    if not series_ids:
        print(f"No readable DICOM series found in {dicom_folder}")
        return []

    output_files = []

    for i, series_id in enumerate(series_ids):
        series_files = reader.GetGDCMSeriesFileNames(dicom_folder, series_id)
        if not series_files:
            print(f"No files found for series {series_id} in {dicom_folder}")
            continue

        reader.SetFileNames(series_files)

        try:
            image = reader.Execute()
            output_path = f"{output_base_path}_series{i}.nii.gz"
            sitk.WriteImage(image, output_path)
            output_files.append(output_path)
        except Exception as exc:
            print(f"Error in series {i} of {dicom_folder}: {exc}")

    return output_files


def threshold_airway(image):
    return sitk.BinaryThreshold(
        image,
        lowerThreshold=-900,
        upperThreshold=-300,
        insideValue=1,
        outsideValue=0,
    )


def clean_mask(mask):
    closed = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3])
    cc = sitk.ConnectedComponent(closed)
    relabeled = sitk.RelabelComponent(cc, minimumObjectSize=1000)
    largest = sitk.BinaryThreshold(relabeled, 1, 1)
    return largest


def sanitize_scan_id(value):
    cleaned = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "scan"


def discover_series_directories(root_path):
    series_directories = []

    for dirpath, dirnames, filenames in os.walk(root_path, topdown=True):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not dirname.startswith(".") and dirname != "ctkDICOM-Database"
        ]

        has_dicom_files = any(
            filename.lower().endswith((".dcm", ".dicom")) for filename in filenames
        )
        if not has_dicom_files:
            continue

        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dirpath) or []
        if series_ids:
            series_directories.append(dirpath)

    return sorted(series_directories)


def build_scan_id(series_dir, input_root, scan_id):
    relative_path = os.path.relpath(series_dir, input_root)
    if relative_path == ".":
        return sanitize_scan_id(scan_id)

    relative_id = sanitize_scan_id(relative_path.replace(os.sep, "__"))
    prefix = sanitize_scan_id(scan_id)

    if scan_id == "single_scan":
        return relative_id

    return f"{prefix}__{relative_id}"


def process_scan(scan_path, scan_id, output_root):
    scan_path = os.path.abspath(scan_path)
    output_root = os.path.abspath(output_root)

    if not os.path.isdir(scan_path):
        print(f"Input scan folder does not exist: {scan_path}")
        return False

    os.makedirs(output_root, exist_ok=True)

    try:
        base_path = os.path.join(output_root, scan_id)
        nifti_files = dicom_to_nifti(scan_path, base_path)
        if not nifti_files:
            return False

        for nifti_file in nifti_files:
            image = sitk.ReadImage(nifti_file)
            mask = clean_mask(threshold_airway(image))

            mask_path = nifti_file.replace(".nii.gz", "_mask.nii.gz")
            sitk.WriteImage(mask, mask_path)

        print(f"Done (all series): {scan_id}")
        return True
    except Exception as exc:
        print(f"Error in {scan_id}: {exc}")
        return False


def process_input(input_path, scan_id, output_root):
    input_path = os.path.abspath(input_path)

    direct_series = sitk.ImageSeriesReader.GetGDCMSeriesIDs(input_path) or []
    if direct_series:
        return process_scan(input_path, scan_id, output_root)

    series_directories = discover_series_directories(input_path)
    if not series_directories:
        print(f"No readable DICOM series found in {input_path}")
        return False

    print(f"Found {len(series_directories)} DICOM series folders under {input_path}")

    success_count = 0
    for series_dir in series_directories:
        series_scan_id = build_scan_id(series_dir, input_path, scan_id)
        if process_scan(series_dir, series_scan_id, output_root):
            success_count += 1

    print(f"Processed {success_count}/{len(series_directories)} series folders")
    return success_count == len(series_directories)


def build_parser():
    parser = argparse.ArgumentParser(
        description="TracheaAI — AI-powered trachea segmentation, reconstruction & 3D viewer."
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        default=os.environ.get(
            "AUTO_PIPELINE_INPUT",
            str(DEFAULT_INPUT_ROOT) if DEFAULT_INPUT_ROOT.exists() else None,
        ),
        help="Path to a DICOM series folder or a dataset root containing nested series.",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("AUTO_PIPELINE_OUTPUT", str(DEFAULT_OUTPUT_ROOT)),
        help="Directory where NIfTI and mask outputs will be written.",
    )
    parser.add_argument(
        "--scan-id",
        default="single_scan",
        help="Base name used for generated output files.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the web server after processing.",
    )
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Skip processing and just start the web server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web server (default: 8000).",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Use the AI trachea segmentation pipeline instead of basic threshold.",
    )
    parser.add_argument(
        "--process-nifti",
        type=str,
        default=None,
        help="Process a specific NIfTI file through the AI pipeline.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Server-only mode
    if args.server_only:
        from server.app import start_server
        start_server(port=args.port)
        return

    # AI pipeline mode
    if args.ai or args.process_nifti:
        from server.pipeline import Pipeline
        from segmentation.unet3d import get_device

        device = get_device()
        print(f"🧠 TracheaAI using device: {device}")

        pipeline = Pipeline(output_root=args.output_root, device=device)

        if args.process_nifti:
            pipeline.process_nifti(args.process_nifti, args.scan_id)
        elif args.input_root:
            # Check if the root itself is a DICOM series
            direct_series = sitk.ImageSeriesReader.GetGDCMSeriesIDs(args.input_root) or []
            if direct_series:
                # Root folder is itself a DICOM series
                scan_id = args.scan_id or os.path.basename(args.input_root)
                pipeline.process_dicom(args.input_root, scan_id)
            else:
                # Recursively discover nested DICOM series directories
                series_dirs = discover_series_directories(args.input_root)
                if not series_dirs:
                    print(f"❌ No DICOM series found in {args.input_root}")
                    print("   Make sure your SSD is mounted and the path is correct.")
                    raise SystemExit(1)

                print(f"\n🔍 Found {len(series_dirs)} DICOM series under {args.input_root}")
                success_count = 0
                for i, series_dir in enumerate(series_dirs, 1):
                    scan_id = build_scan_id(series_dir, args.input_root, args.scan_id or "single_scan")
                    print(f"\n[{i}/{len(series_dirs)}] Processing: {scan_id}")
                    try:
                        pipeline.process_dicom(series_dir, scan_id)
                        success_count += 1
                    except Exception as exc:
                        print(f"  ❌ Failed: {exc}")

                print(f"\n✅ AI pipeline complete: {success_count}/{len(series_dirs)} scans processed")
        else:
            parser.error("Provide an input path or --process-nifti for AI mode.")

        if args.serve:
            from server.app import start_server
            start_server(port=args.port)
        return

    # Legacy mode (original threshold pipeline)
    if not args.input_root:
        parser.error(
            "an input folder is required. Pass it as an argument or set AUTO_PIPELINE_INPUT."
        )

    success = process_input(args.input_root, args.scan_id, args.output_root)

    if args.serve:
        from server.app import start_server
        start_server(port=args.port)
    else:
        raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()