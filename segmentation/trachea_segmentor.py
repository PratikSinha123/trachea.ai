"""Hybrid trachea segmentation — combines classical methods with optional U-Net."""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.ndimage import binary_fill_holes

from .preprocessing import HU_AIR_MAX, preprocess_ct, resample_mask_like


class TracheaSegmentor:
    """Multi-stage trachea segmentation engine."""

    def __init__(self, target_spacing=(0.75, 0.75, 0.75), use_unet=False,
                 model_path=None, device="cpu"):
        self.target_spacing = target_spacing
        self.use_unet = use_unet
        self.model_path = model_path
        self.device = device
        self._unet = None

    def segment(self, image: sitk.Image):
        """Run the full trachea segmentation pipeline."""
        print("[1/5] Preprocessing CT scan...")
        resampled, lung_mask, normalized = preprocess_ct(image, self.target_spacing)

        if self.use_unet and self.model_path:
            print("[2/5] Running U-Net segmentation...")
            airway_mask = self._run_unet(normalized)
        else:
            print("[2/5] Threshold-based airway extraction...")
            airway_mask = self._threshold_airways(resampled)

        print("[3/5] Isolating trachea via region growing...")
        trachea_mask = self._isolate_trachea(airway_mask, resampled)

        print("[4/5] Refining segmentation...")
        refined_mask = self._refine_mask(trachea_mask)

        print("[5/5] Extracting centerline...")
        centerline, cross_sections = self._extract_centerline(refined_mask, resampled)

        original_mask = resample_mask_like(refined_mask, image)

        return {
            "trachea_mask": original_mask,
            "centerline": centerline,
            "cross_sections": cross_sections,
            "resampled_image": resampled,
            "resampled_mask": refined_mask,
        }

    def _threshold_airways(self, image):
        arr = sitk.GetArrayFromImage(image)
        air_mask = ((arr >= -1024) & (arr <= HU_AIR_MAX)).astype(np.uint8)

        body_mask = (arr > -500).astype(np.uint8)
        body_filled = np.zeros_like(body_mask)
        for z in range(body_mask.shape[0]):
            body_filled[z] = binary_fill_holes(body_mask[z]).astype(np.uint8)

        internal_air = air_mask * body_filled
        result = sitk.GetImageFromArray(internal_air)
        result.CopyInformation(image)
        return result

    def _isolate_trachea(self, airway_mask, original_image):
        arr = sitk.GetArrayFromImage(airway_mask)
        image_arr = sitk.GetArrayFromImage(original_image)
        top_range = max(1, arr.shape[0] // 5)
        seed_z = seed_y = seed_x = None

        for z in range(top_range):
            sl = arr[z]
            if sl.sum() < 10:
                continue
            labeled, nf = ndimage.label(sl)
            if nf == 0:
                continue
            cy_center, cx_center = sl.shape[0] // 2, sl.shape[1] // 2
            best_dist = float("inf")
            for lid in range(1, nf + 1):
                comp = labeled == lid
                cs = comp.sum()
                if cs < 20 or cs > 5000:
                    continue
                cy, cx = ndimage.center_of_mass(comp)
                d = np.sqrt((cy - cy_center)**2 + (cx - cx_center)**2)
                if d < best_dist:
                    best_dist = d
                    seed_z, seed_y, seed_x = z, int(cy), int(cx)
            if seed_z is not None:
                break

        if seed_z is None:
            print("  WARNING: No trachea seed found, using full airway mask")
            return airway_mask

        print(f"  Trachea seed at ({seed_z}, {seed_y}, {seed_x})")
        trachea = self._region_grow(arr, seed_z, seed_y, seed_x)
        result = sitk.GetImageFromArray(trachea.astype(np.uint8))
        result.CopyInformation(airway_mask)
        return result

    def _region_grow(self, airway_arr, sz, sy, sx, max_cs=3000, rate_limit=2.5):
        output = np.zeros_like(airway_arr, dtype=np.uint8)
        visited = np.zeros_like(airway_arr, dtype=bool)
        queue = [(sz, sy, sx)]
        visited[sz, sy, sx] = True
        output[sz, sy, sx] = 1
        slice_counts = {}

        while queue:
            z, y, x = queue.pop(0)
            for dz, dy, dx in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                nz, ny, nx = z+dz, y+dy, x+dx
                if not (0 <= nz < airway_arr.shape[0] and
                        0 <= ny < airway_arr.shape[1] and
                        0 <= nx < airway_arr.shape[2]):
                    continue
                if visited[nz, ny, nx] or airway_arr[nz, ny, nx] == 0:
                    continue
                visited[nz, ny, nx] = True
                cur = slice_counts.get(nz, 0) + 1
                slice_counts[nz] = cur
                if cur > max_cs:
                    continue
                output[nz, ny, nx] = 1
                queue.append((nz, ny, nx))

        sizes = [(z, output[z].sum()) for z in range(output.shape[0]) if output[z].sum() > 0]
        if sizes:
            med = np.median([s for _, s in sizes])
            for z, s in sizes:
                if s > med * rate_limit and s > 500:
                    output[z] = 0
        return output

    def _refine_mask(self, mask):
        arr = sitk.GetArrayFromImage(mask)
        struct = ndimage.generate_binary_structure(3, 2)
        closed = ndimage.binary_closing(arr, structure=struct, iterations=2)
        for z in range(closed.shape[0]):
            closed[z] = binary_fill_holes(closed[z])
        opened = ndimage.binary_opening(closed, structure=struct, iterations=1)
        labeled, nf = ndimage.label(opened)
        if nf > 1:
            sizes = ndimage.sum(opened, labeled, range(1, nf + 1))
            opened = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
        else:
            opened = opened.astype(np.uint8)
        result = sitk.GetImageFromArray(opened)
        result.CopyInformation(mask)
        return result

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
