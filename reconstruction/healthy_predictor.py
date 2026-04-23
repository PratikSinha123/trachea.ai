"""Healthy trachea predictor — reconstructs the ideal trachea shape.

Takes the segmented (possibly diseased) trachea and produces a reconstructed
healthy version by identifying anomalies and rebuilding those regions using
anatomical priors and interpolation from healthy sections.
"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, binary_fill_holes

from .shape_model import TracheaShapeModel


class HealthyTracheaPredictor:
    """Predicts what the trachea should look like without disease."""

    def __init__(self):
        self.shape_model = TracheaShapeModel()
        self.anomalies = []
        self.stats = {}

    def predict(self, segmentation_result: dict):
        """Generate a healthy trachea reconstruction.

        Args:
            segmentation_result: Output dict from TracheaSegmentor.segment()

        Returns:
            dict with keys:
                - 'healthy_mask': Reconstructed healthy trachea mask (sitk.Image)
                - 'anomalies': List of detected anomaly descriptions
                - 'stats': Comparison statistics
                - 'morph_frames': List of intermediate masks for morphing animation
        """
        mask = segmentation_result["resampled_mask"]
        centerline = segmentation_result["centerline"]
        cross_sections = segmentation_result["cross_sections"]
        image = segmentation_result["resampled_image"]

        # Fit shape model
        print("  Fitting shape model...")
        self.shape_model.fit(centerline, cross_sections)

        # Detect anomalies
        print("  Detecting anomalies...")
        anomaly_idx, self.anomalies = self.shape_model.detect_anomalies()

        # Generate healthy profiles
        print("  Generating healthy profile...")
        z_coords, healthy_diameters = self.shape_model.predict_healthy_profile()
        healthy_centerline = self.shape_model.predict_healthy_centerline()

        if z_coords is None:
            print("  WARNING: Cannot predict healthy shape, returning original")
            return {
                "healthy_mask": mask,
                "anomalies": [],
                "stats": {},
                "morph_frames": [],
            }

        # Reconstruct the healthy trachea volume
        print("  Reconstructing healthy volume...")
        healthy_mask = self._reconstruct_volume(
            mask, cross_sections, healthy_diameters, healthy_centerline
        )

        # Generate morph frames
        print("  Generating morph animation frames...")
        morph_frames = self._generate_morph_frames(mask, healthy_mask, num_frames=20)

        # Calculate comparison stats
        self.stats = self._calculate_stats(
            mask, healthy_mask, cross_sections, healthy_diameters
        )

        return {
            "healthy_mask": healthy_mask,
            "anomalies": self.anomalies,
            "stats": self.stats,
            "morph_frames": morph_frames,
        }

    def _reconstruct_volume(self, original_mask, cross_sections,
                            healthy_diameters, healthy_centerline):
        """Reconstruct a 3D healthy trachea volume slice by slice."""
        arr = sitk.GetArrayFromImage(original_mask)
        spacing = original_mask.GetSpacing()
        healthy = np.zeros_like(arr, dtype=np.uint8)

        cs_by_z = {cs["z_index"]: i for i, cs in enumerate(cross_sections)}

        for z in range(arr.shape[0]):
            if z not in cs_by_z:
                # If this slice had no trachea, check if it should (interpolate)
                continue

            idx = cs_by_z[z]
            if idx >= len(healthy_diameters):
                continue

            target_diameter = healthy_diameters[idx]
            cs = cross_sections[idx]

            # Use centerline position for the center of the healthy cross-section
            if idx < len(healthy_centerline):
                cx_phys = healthy_centerline[idx][0]
                cy_phys = healthy_centerline[idx][1]
            else:
                cx_phys = cs["centroid_x"]
                cy_phys = cs["centroid_y"]

            # Convert to voxel coordinates
            cx_vox = cx_phys / spacing[0]
            cy_vox = cy_phys / spacing[1]

            # Draw circular cross-section
            radius_vox_x = (target_diameter / 2.0) / spacing[0]
            radius_vox_y = (target_diameter / 2.0) / spacing[1]

            yy, xx = np.ogrid[:arr.shape[1], :arr.shape[2]]
            ellipse = ((yy - cy_vox) / radius_vox_y) ** 2 + \
                      ((xx - cx_vox) / radius_vox_x) ** 2

            healthy[z] = (ellipse <= 1.0).astype(np.uint8)

        # Smooth the volume to avoid sharp edges between slices
        healthy_smooth = gaussian_filter(healthy.astype(np.float32), sigma=1.0)
        healthy_binary = (healthy_smooth > 0.3).astype(np.uint8)

        result = sitk.GetImageFromArray(healthy_binary)
        result.CopyInformation(original_mask)
        return result

    def _generate_morph_frames(self, diseased_mask, healthy_mask, num_frames=20):
        """Generate intermediate frames for smooth morphing animation.

        Uses distance-field based interpolation for smooth transitions.
        """
        d_arr = sitk.GetArrayFromImage(diseased_mask).astype(np.float32)
        h_arr = sitk.GetArrayFromImage(healthy_mask).astype(np.float32)

        frames = []
        for i in range(num_frames + 1):
            t = i / num_frames  # 0.0 (diseased) to 1.0 (healthy)

            # Smooth interpolation with easing
            t_ease = t * t * (3 - 2 * t)  # Smoothstep

            blended = d_arr * (1 - t_ease) + h_arr * t_ease
            binary = (blended > 0.5).astype(np.uint8)

            frame = sitk.GetImageFromArray(binary)
            frame.CopyInformation(diseased_mask)
            frames.append(frame)

        return frames

    def _calculate_stats(self, diseased_mask, healthy_mask,
                         cross_sections, healthy_diameters):
        """Calculate comparison statistics between diseased and healthy."""
        d_arr = sitk.GetArrayFromImage(diseased_mask)
        h_arr = sitk.GetArrayFromImage(healthy_mask)
        sp = diseased_mask.GetSpacing()
        voxel_vol = sp[0] * sp[1] * sp[2]

        d_vol = float(d_arr.sum() * voxel_vol)
        h_vol = float(h_arr.sum() * voxel_vol)

        obs_diams = [cs["equiv_diameter_mm"] for cs in cross_sections]

        return {
            "diseased_volume_mm3": d_vol,
            "healthy_volume_mm3": h_vol,
            "volume_change_pct": float((h_vol - d_vol) / max(d_vol, 1) * 100),
            "num_anomalies": len(self.anomalies),
            "mean_diseased_diameter_mm": float(np.mean(obs_diams)) if obs_diams else 0,
            "mean_healthy_diameter_mm": float(np.mean(healthy_diameters)) if len(healthy_diameters) else 0,
            "min_diseased_diameter_mm": float(np.min(obs_diams)) if obs_diams else 0,
            "max_stenosis_pct": max(
                (a["deviation_pct"] for a in self.anomalies if a["type"] == "stenosis"),
                default=0,
            ),
        }
