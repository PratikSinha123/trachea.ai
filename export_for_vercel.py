"""Export processed scan data for Vercel static hosting.

Takes the processed_data/ directory produced by the local AI pipeline
and exports a lightweight, web-ready version into public/data/ that can
be deployed directly to Vercel.

Exports per scan:
    - metadata.json (already exists, just copy)
    - meshes/*.glb  (copy all mesh files)
    - slices/       (pre-rendered PNG slices for axial/coronal/sagittal)

Usage:
    python export_for_vercel.py                          # Export all scans
    python export_for_vercel.py --scan-id LIDC-0035-main # Export one scan
    python export_for_vercel.py --skip-slices            # Skip heavy slice rendering
"""

import argparse
import json
import os
import shutil
import sys

# Slice rendering needs these — but we import lazily so the script
# can at least show --help without them installed.
SLICES_PER_AXIS = 64  # Pre-render this many evenly-spaced slices per axis


def get_scan_dirs(processed_root, scan_id=None):
    """Discover all processed scan directories (those with metadata.json)."""
    dirs = []
    if not os.path.isdir(processed_root):
        return dirs

    if scan_id:
        candidate = os.path.join(processed_root, scan_id)
        if os.path.isfile(os.path.join(candidate, "metadata.json")):
            dirs.append((scan_id, candidate))
        return dirs

    for name in sorted(os.listdir(processed_root)):
        meta = os.path.join(processed_root, name, "metadata.json")
        if os.path.isfile(meta):
            dirs.append((name, os.path.join(processed_root, name)))

    return dirs


def copy_meshes(scan_dir, output_dir):
    """Copy all .glb mesh files preserving subdirectory structure."""
    meshes_src = os.path.join(scan_dir, "meshes")
    meshes_dst = os.path.join(output_dir, "meshes")

    if not os.path.isdir(meshes_src):
        print("    ⚠ No meshes/ directory found, skipping")
        return 0

    count = 0
    for root, dirs, files in os.walk(meshes_src):
        for f in files:
            if f.endswith(".glb"):
                src = os.path.join(root, f)
                rel = os.path.relpath(src, meshes_src)
                dst = os.path.join(meshes_dst, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                count += 1

    return count


def render_slices(scan_dir, output_dir, num_slices=SLICES_PER_AXIS):
    """Pre-render PNG slices for all three axes."""
    import numpy as np
    import SimpleITK as sitk

    # Find the CT NIfTI file
    ct_path = None
    skip_names = {"mask", "healthy", "trachea"}
    for f in sorted(os.listdir(scan_dir)):
        if f.endswith((".nii.gz", ".nii")) and not any(s in f for s in skip_names):
            ct_path = os.path.join(scan_dir, f)
            break

    if ct_path is None:
        print("    ⚠ No CT NIfTI found, skipping slice export")
        return {}

    image = sitk.ReadImage(ct_path)
    arr = sitk.GetArrayFromImage(image)  # shape: (Z, Y, X)

    # Load mask for overlay if available
    mask_path = os.path.join(scan_dir, "trachea_mask.nii.gz")
    mask_arr = None
    if os.path.isfile(mask_path):
        try:
            mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            if mask_arr.shape != arr.shape:
                mask_arr = None
        except Exception:
            mask_arr = None

    dimensions = {
        "axial": arr.shape[0],
        "coronal": arr.shape[1],
        "sagittal": arr.shape[2],
    }

    slices_dir = os.path.join(output_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)

    # Map of axis → list of {index, filename}
    slice_manifest = {}

    for axis in ["axial", "coronal", "sagittal"]:
        axis_dir = os.path.join(slices_dir, axis)
        os.makedirs(axis_dir, exist_ok=True)

        max_idx = dimensions[axis]
        indices = _even_indices(max_idx, num_slices)
        entries = []

        for idx in indices:
            # Extract slice
            if axis == "axial":
                sl = arr[idx, :, :]
                m_sl = mask_arr[idx, :, :] if mask_arr is not None else None
            elif axis == "coronal":
                sl = arr[:, idx, :]
                m_sl = mask_arr[:, idx, :] if mask_arr is not None else None
            else:
                sl = arr[:, :, idx]
                m_sl = mask_arr[:, :, idx] if mask_arr is not None else None

            # Window/level for soft tissue
            sl = np.clip(sl, -1024, 600)
            sl = ((sl + 1024) / 1624 * 255).astype(np.uint8)

            # Convert to RGB
            rgb = np.stack([sl, sl, sl], axis=-1)

            # Overlay mask
            if m_sl is not None:
                overlay = rgb.copy()
                overlay[m_sl > 0] = [52, 211, 153]
                alpha = 0.35
                rgb = (overlay * alpha + rgb * (1 - alpha)).astype(np.uint8)

            # Save as PNG
            fname = f"{idx:04d}.png"
            fpath = os.path.join(axis_dir, fname)
            _save_png(rgb, fpath)
            entries.append({"index": idx, "file": f"slices/{axis}/{fname}"})

        slice_manifest[axis] = entries

    return {"dimensions": dimensions, "slices": slice_manifest}


def _even_indices(total, count):
    """Generate evenly-spaced indices across a range."""
    if total <= count:
        return list(range(total))
    step = total / count
    return [int(i * step) for i in range(count)]


def _save_png(rgb_array, path):
    """Save an RGB numpy array as PNG."""
    from PIL import Image
    img = Image.fromarray(rgb_array, mode="RGB")
    img.save(path, format="PNG", optimize=True)


def export_scan(scan_id, scan_dir, output_root, skip_slices=False):
    """Export a single scan to the output directory."""
    print(f"\n  📦 Exporting: {scan_id}")

    out_dir = os.path.join(output_root, scan_id)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Copy metadata.json
    meta_src = os.path.join(scan_dir, "metadata.json")
    meta_dst = os.path.join(out_dir, "metadata.json")
    shutil.copy2(meta_src, meta_dst)
    print("    ✅ metadata.json")

    # 2. Copy meshes
    mesh_count = copy_meshes(scan_dir, out_dir)
    print(f"    ✅ {mesh_count} mesh files (.glb)")

    # 3. Pre-render slices
    slice_info = {}
    if not skip_slices:
        print("    🖼  Rendering slices (this may take a moment)...")
        slice_info = render_slices(scan_dir, out_dir)
        if slice_info:
            total = sum(len(v) for v in slice_info.get("slices", {}).values())
            print(f"    ✅ {total} slice PNGs across 3 axes")
        else:
            print("    ⚠ No slices rendered (no CT scan found)")
    else:
        print("    ⏭  Skipped slice rendering")

    # 4. Write an export manifest
    manifest = {
        "scan_id": scan_id,
        "has_meshes": mesh_count > 0,
        "has_slices": bool(slice_info),
    }
    if slice_info:
        manifest["dimensions"] = slice_info.get("dimensions", {})
        manifest["slice_manifest"] = slice_info.get("slices", {})

    manifest_path = os.path.join(out_dir, "export_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def build_scan_index(output_root):
    """Build a top-level scans.json index of all exported scans."""
    scans = []
    for name in sorted(os.listdir(output_root)):
        meta_path = os.path.join(output_root, name, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                scans.append(json.load(f))

    index_path = os.path.join(output_root, "scans.json")
    with open(index_path, "w") as f:
        json.dump(scans, f, indent=2)

    print(f"\n✅ Scan index: {index_path} ({len(scans)} scans)")
    return scans


def main():
    parser = argparse.ArgumentParser(
        description="Export processed TracheaAI data for Vercel static deployment."
    )
    parser.add_argument(
        "--processed-root",
        default=os.path.join(os.path.dirname(__file__), "processed_data"),
        help="Path to the processed_data/ directory (default: ./processed_data)",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(os.path.dirname(__file__), "public", "data"),
        help="Output directory for Vercel-ready static files (default: ./public/data)",
    )
    parser.add_argument(
        "--scan-id",
        default=None,
        help="Export only this specific scan ID",
    )
    parser.add_argument(
        "--skip-slices",
        action="store_true",
        help="Skip pre-rendering CT slice PNGs (much faster export)",
    )
    args = parser.parse_args()

    processed_root = os.path.abspath(args.processed_root)
    output_root = os.path.abspath(args.output_root)

    print(f"🫁 TracheaAI — Vercel Export")
    print(f"   Source:  {processed_root}")
    print(f"   Output:  {output_root}")

    scans = get_scan_dirs(processed_root, args.scan_id)
    if not scans:
        print("\n❌ No processed scans found. Run the pipeline first:")
        print("   python auto_pipeline.py --ai /path/to/dicom")
        sys.exit(1)

    print(f"\n📋 Found {len(scans)} scan(s) to export")

    os.makedirs(output_root, exist_ok=True)

    for scan_id, scan_dir in scans:
        export_scan(scan_id, scan_dir, output_root, skip_slices=args.skip_slices)

    # Build top-level index
    build_scan_index(output_root)

    print(f"\n🚀 Export complete! Deploy with:")
    print(f"   cd {os.path.dirname(__file__)}")
    print(f"   vercel --prod")


if __name__ == "__main__":
    main()
