"""TotalSegmentator-based trachea segmentation — an industry-standard deep learning approach."""

import os
import subprocess
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.ndimage import binary_fill_holes

from .preprocessing import preprocess_ct, resample_mask_like


class TracheaSegmentor:
    """Deep Learning trachea segmentation using TotalSegmentator."""

    def __init__(self, target_spacing=(0.75, 0.75, 0.75), use_unet=True,
                 model_path=None, device="cpu"):
        self.target_spacing = target_spacing
        self.device = device

    def segment(self, image: sitk.Image):
        """Run the full trachea segmentation using TotalSegmentator."""
        print("[1/3] Preprocessing CT scan for TotalSegmentator...")
        resampled, lung_mask, normalized = preprocess_ct(image, self.target_spacing)

        print("[2/3] Running TotalSegmentator AI (This isolates the exact trachea anatomy)...")
        # TotalSegmentator works best via files. We'll write the resampled image.
        temp_in = "temp_ct_scan.nii.gz"
        temp_out = "temp_trachea_mask.nii.gz"
        
        sitk.WriteImage(resampled, temp_in)
        
        cmd = [
            "TotalSegmentator",
            "-i", temp_in,
            "-o", temp_out,
            "--roi_subset", "trachea"
        ]
        
        if self.device in ["cuda", "mps"]:
            # If the user has a GPU or MPS, total segmentator will use PyTorch automatically
            pass
        elif self.device == "cpu":
            cmd.extend(["--device", "cpu"])
            
        print(f"  Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        if not os.path.exists(temp_out):
            raise RuntimeError("TotalSegmentator failed to generate a mask.")
            
        refined_mask = sitk.ReadImage(temp_out)
        
        # Cleanup temp files
        if os.path.exists(temp_in):
            os.remove(temp_in)
        if os.path.exists(temp_out):
            os.remove(temp_out)

        print("[3/3] Extracting centerline and cross-sections...")
        centerline, cross_sections = self._extract_centerline(refined_mask, resampled)

        original_mask = resample_mask_like(refined_mask, image)

        return {
            "trachea_mask": original_mask,
            "centerline": centerline,
            "cross_sections": cross_sections,
            "resampled_image": resampled,
            "resampled_mask": refined_mask,
        }

    def _extract_centerline(self, mask, image):
        arr = sitk.GetArrayFromImage(mask)
        sp = mask.GetSpacing()
        centerline, cross_sections = [], []
        for z in range(arr.shape[0]):
            area = arr[z].sum()
            if area < 5:
                continue
            cy, cx = ndimage.center_of_mass(arr[z])
            phys = [cx * sp[0], cy * sp[1], z * sp[2]]
            centerline.append(phys)
            voxel_area = area * sp[0] * sp[1]
            eq_d = 2 * np.sqrt(voxel_area / np.pi)
            ys, xs = np.where(arr[z] > 0)
            h = (ys.max()-ys.min()+1)*sp[1] if len(ys) else 0
            w = (xs.max()-xs.min()+1)*sp[0] if len(xs) else 0
            cross_sections.append({
                "z_index": z, "z_physical": phys[2],
                "centroid_x": phys[0], "centroid_y": phys[1],
                "area_mm2": float(voxel_area), "equiv_diameter_mm": float(eq_d),
                "width_mm": float(w), "height_mm": float(h), "voxel_count": int(area),
            })
        return np.array(centerline) if centerline else np.empty((0, 3)), cross_sections

    def _run_unet(self, normalized_image):
        import torch
        from .preprocessing import prepare_for_unet, reconstruct_from_patches
        from .unet3d import UNet3D
        if self._unet is None:
            self._unet = UNet3D(1, 1)
            if self.model_path:
                self._unet.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self._unet.to(self.device).eval()
        patches = prepare_for_unet(normalized_image, (64, 128, 128))
        preds = []
        with torch.no_grad():
            for pa, origin in patches:
                t = torch.from_numpy(pa).unsqueeze(0).unsqueeze(0).to(self.device)
                p = torch.sigmoid(self._unet(t)).squeeze().cpu().numpy()
                preds.append((p, origin))
        arr = sitk.GetArrayFromImage(normalized_image)
        out = reconstruct_from_patches(preds, arr.shape, (64, 128, 128))
        binary = (out > 0.5).astype(np.uint8)
        result = sitk.GetImageFromArray(binary)
        result.CopyInformation(normalized_image)
        return result
