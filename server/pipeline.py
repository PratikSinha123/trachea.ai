"""Pipeline orchestrator — runs the full segmentation + reconstruction pipeline."""

import json
import os
import time

import SimpleITK as sitk

from segmentation.trachea_segmentor import TracheaSegmentor
from reconstruction.healthy_predictor import HealthyTracheaPredictor
from visualization.mesh_generator import MeshGenerator


class Pipeline:
    """Full trachea analysis pipeline: DICOM → Segment → Reconstruct → Mesh."""

    def __init__(self, output_root="processed_data", device="cpu"):
        self.output_root = output_root
        self.segmentor = TracheaSegmentor(device=device)
        self.predictor = HealthyTracheaPredictor()
        self.mesh_gen = MeshGenerator(smooth_iterations=15, decimate_ratio=0.3)

    def process_nifti(self, nifti_path, scan_id=None):
        """Process a NIfTI file through the full pipeline."""
        if scan_id is None:
            scan_id = os.path.splitext(os.path.basename(nifti_path))[0]
            scan_id = scan_id.replace(".nii", "")

        scan_dir = os.path.join(self.output_root, scan_id)
        os.makedirs(scan_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing: {scan_id}")
        print(f"{'='*60}")

        t0 = time.time()

        # Load image
        print("\n[LOAD] Reading NIfTI...")
        image = sitk.ReadImage(nifti_path)
        print(f"  Size: {image.GetSize()}, Spacing: {image.GetSpacing()}")

        # Segment
        print("\n[SEGMENT] Running trachea segmentation...")
        seg_result = self.segmentor.segment(image)

        # Save segmentation mask
        mask_path = os.path.join(scan_dir, "trachea_mask.nii.gz")
        sitk.WriteImage(seg_result["trachea_mask"], mask_path)
        print(f"  Saved mask: {mask_path}")

        # Reconstruct healthy
        print("\n[RECONSTRUCT] Predicting healthy trachea...")
        recon_result = self.predictor.predict(seg_result)

        # Save healthy mask
        healthy_path = os.path.join(scan_dir, "trachea_healthy.nii.gz")
        sitk.WriteImage(recon_result["healthy_mask"], healthy_path)

        # Generate meshes
        print("\n[MESH] Generating 3D meshes...")
        meshes_dir = os.path.join(scan_dir, "meshes")
        os.makedirs(meshes_dir, exist_ok=True)

        # Diseased mesh
        diseased_mesh = self.mesh_gen.mask_to_mesh(
            seg_result["resampled_mask"], smooth_sigma=1.5
        )
        self.mesh_gen.export_glb(
            diseased_mesh,
            os.path.join(meshes_dir, "diseased.glb"),
            color=(0.9, 0.25, 0.2),
        )

        # Healthy mesh
        healthy_mesh = self.mesh_gen.mask_to_mesh(
            recon_result["healthy_mask"], smooth_sigma=1.5
        )
        self.mesh_gen.export_glb(
            healthy_mesh,
            os.path.join(meshes_dir, "healthy.glb"),
            color=(0.2, 0.85, 0.5),
        )

        # Morph frames
        morph_dir = os.path.join(meshes_dir, "morph")
        morph_paths = self.mesh_gen.generate_morph_meshes(
            recon_result["morph_frames"], morph_dir
        )

        # Save metadata
        metadata = {
            "scan_id": scan_id,
            "processing_time_s": round(time.time() - t0, 1),
            "anomalies": recon_result["anomalies"],
            "stats": recon_result["stats"],
            "centerline": seg_result["centerline"].tolist()
                if len(seg_result["centerline"]) > 0 else [],
            "cross_sections": seg_result["cross_sections"],
            "meshes": {
                "diseased": "meshes/diseased.glb",
                "healthy": "meshes/healthy.glb",
                "morph_frames": [
                    os.path.relpath(p, scan_dir) for p in morph_paths
                ],
            },
        }

        meta_path = os.path.join(scan_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"DONE: {scan_id} ({elapsed:.1f}s)")
        print(f"  Anomalies found: {len(recon_result['anomalies'])}")
        print(f"  Morph frames: {len(morph_paths)}")
        print(f"  Output: {scan_dir}")
        print(f"{'='*60}\n")

        return metadata

    def process_dicom(self, dicom_folder, scan_id=None):
        """Process a DICOM folder through the full pipeline."""
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_folder) or []

        if not series_ids:
            print(f"No DICOM series in {dicom_folder}")
            return None

        series_files = reader.GetGDCMSeriesFileNames(dicom_folder, series_ids[0])
        reader.SetFileNames(series_files)
        image = reader.Execute()

        if scan_id is None:
            scan_id = os.path.basename(dicom_folder)

        # Save as NIfTI first
        nifti_dir = os.path.join(self.output_root, scan_id)
        os.makedirs(nifti_dir, exist_ok=True)
        nifti_path = os.path.join(nifti_dir, "ct_scan.nii.gz")
        sitk.WriteImage(image, nifti_path)

        return self.process_nifti(nifti_path, scan_id)

    def list_scans(self):
        """List all processed scans."""
        scans = []
        if not os.path.isdir(self.output_root):
            return scans

        for name in sorted(os.listdir(self.output_root)):
            meta_path = os.path.join(self.output_root, name, "metadata.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    scans.append(json.load(f))
        return scans
