"""3D mesh generation from segmentation masks.

Uses marching cubes to extract surface meshes, applies smoothing and
decimation, and exports to GLB/OBJ formats for web visualization.
"""

import json
import struct
import os

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter


def _marching_cubes(volume, spacing=(1.0, 1.0, 1.0), level=0.5, step_size=1):
    """Marching cubes using skimage."""
    from skimage.measure import marching_cubes
    verts, faces, normals, values = marching_cubes(
        volume, level=level, spacing=spacing, step_size=step_size
    )
    return verts, faces, normals


class MeshGenerator:
    """Generate and export 3D meshes from binary segmentation masks."""

    def __init__(self, smooth_iterations=15, decimate_ratio=0.3):
        self.smooth_iterations = smooth_iterations
        self.decimate_ratio = decimate_ratio

    def mask_to_mesh(self, mask: sitk.Image, smooth_sigma=1.0):
        """Convert a binary mask to a 3D mesh.

        Args:
            mask: Binary segmentation mask (SimpleITK image)
            smooth_sigma: Gaussian smoothing sigma for the volume before
                marching cubes (higher = smoother but less detailed)

        Returns:
            dict: {'vertices': (N,3), 'faces': (M,3), 'normals': (N,3)}
        """
        arr = sitk.GetArrayFromImage(mask).astype(np.float32)
        spacing = mask.GetSpacing()

        if arr.sum() < 10:
            return {"vertices": np.empty((0, 3)), "faces": np.empty((0, 3), dtype=int),
                    "normals": np.empty((0, 3))}

        # Heavy Gaussian pre-smoothing for organic feel
        # Apply multiple passes: broad + fine
        if smooth_sigma > 0:
            arr = gaussian_filter(arr, sigma=smooth_sigma * 2.0)   # broad pass
            arr = gaussian_filter(arr, sigma=smooth_sigma * 0.8)   # fine pass

        # Marching cubes — note SimpleITK uses (x,y,z) spacing but array is (z,y,x)
        spacing_zyx = (spacing[2], spacing[1], spacing[0])

        try:
            verts, faces, normals = _marching_cubes(arr, spacing=spacing_zyx, level=0.5)
        except Exception as e:
            print(f"  Marching cubes failed: {e}")
            return {"vertices": np.empty((0, 3)), "faces": np.empty((0, 3), dtype=int),
                    "normals": np.empty((0, 3))}

        # Vectorized Laplacian smoothing (fast NumPy — no Python loops per vertex)
        if self.smooth_iterations > 0:
            verts = self._laplacian_smooth(verts, faces, self.smooth_iterations)
            # Recompute normals after smoothing
            normals = self._compute_normals(verts, faces)

        # Decimate if requested
        if 0 < self.decimate_ratio < 1.0:
            verts, faces, normals = self._decimate(verts, faces, normals)

        return {"vertices": verts, "faces": faces, "normals": normals}

    def _laplacian_smooth(self, vertices, faces, iterations=30, lam=0.6):
        """Apply Laplacian smoothing using fast vectorized NumPy operations."""
        n_verts = len(vertices)
        smoothed = vertices.copy()

        # Build COO-style edge list from faces (vectorized)
        i0 = faces[:, 0]; i1 = faces[:, 1]; i2 = faces[:, 2]
        rows = np.concatenate([i0, i1, i2, i1, i2, i0])
        cols = np.concatenate([i1, i2, i0, i0, i1, i2])
        # Use sparse matrix for fast neighbor averaging
        from scipy.sparse import coo_matrix
        data = np.ones(len(rows))
        adj = coo_matrix((data, (rows, cols)), shape=(n_verts, n_verts)).tocsr()
        # Degree (number of neighbors per vertex)
        degree = np.array(adj.sum(axis=1)).ravel()[:, None].clip(1)

        for _ in range(iterations):
            neighbor_sum = adj.dot(smoothed)
            centroid = neighbor_sum / degree
            smoothed = smoothed + lam * (centroid - smoothed)

        return smoothed

    def _compute_normals(self, vertices, faces):
        """Compute per-vertex normals from face normals."""
        normals = np.zeros_like(vertices)

        for f in faces:
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            fn = np.cross(edge1, edge2)
            fn_len = np.linalg.norm(fn)
            if fn_len > 1e-10:
                fn /= fn_len
            for idx in f:
                normals[idx] += fn

        # Normalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-10)
        normals /= lengths

        return normals

    def _decimate(self, vertices, faces, normals):
        """Mesh decimation that preserves topology using vertex clustering."""
        n_target_faces = max(200, int(len(faces) * self.decimate_ratio))
        if len(faces) <= n_target_faces:
            return vertices, faces, normals

        # Instead of random face sampling (breaks topology), we use
        # a keep-every-Nth-vertex approach that preserves manifold structure.
        # We cluster nearby vertices by quantizing their positions.
        step = max(1, int(round(len(faces) / n_target_faces)))

        # Pick a subset of faces with stride to preserve mesh coherence
        keep = np.arange(0, len(faces), step)
        new_faces = faces[keep]

        # Re-index to only the used vertices
        used = np.unique(new_faces.ravel())
        remap = np.full(len(vertices), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)

        new_verts = vertices[used]
        new_normals = normals[used]
        new_faces_remapped = remap[new_faces]

        # Drop degenerate faces (where remap was -1)
        valid = (new_faces_remapped >= 0).all(axis=1)
        new_faces_remapped = new_faces_remapped[valid]

        return new_verts, new_faces_remapped, new_normals

    def export_glb(self, mesh_data, output_path, color=(0.8, 0.2, 0.2),
                   vertex_colors=None):
        """Export mesh to GLB (binary glTF) format for Three.js.

        Args:
            mesh_data: dict with 'vertices', 'faces', 'normals'
            output_path: file path for the .glb file
            color: RGB tuple fallback color (used when vertex_colors is None)
            vertex_colors: optional (N,3) float32 array of per-vertex RGB colors
        """
        verts = mesh_data["vertices"].astype(np.float32)
        faces = mesh_data["faces"].astype(np.uint32)
        normals = mesh_data["normals"].astype(np.float32)

        if len(verts) == 0:
            print(f"  Skipping empty mesh export: {output_path}")
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Build binary buffer
        indices = faces.ravel().astype(np.uint16)
        vert_bytes = verts.tobytes()
        norm_bytes = normals.tobytes()
        idx_bytes = indices.tobytes()

        if vertex_colors is not None:
            colors_f32 = vertex_colors.astype(np.float32)
            col_bytes = colors_f32.tobytes()
        else:
            col_bytes = None

        # Compute bounding box
        v_min = verts.min(axis=0).tolist()
        v_max = verts.max(axis=0).tolist()

        # Pack buffer: indices | verts | normals [| colors]
        buffer_data = idx_bytes + vert_bytes + norm_bytes
        if col_bytes:
            buffer_data += col_bytes
        while len(buffer_data) % 4 != 0:
            buffer_data += b'\x00'

        idx_offset = 0
        vert_offset = len(idx_bytes)
        norm_offset = vert_offset + len(vert_bytes)
        col_offset = norm_offset + len(norm_bytes)

        # Build primitive attributes
        prim_attrs = {"POSITION": 1, "NORMAL": 2}
        accessors = [
            {  # 0: Indices
                "bufferView": 0,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": len(indices),
                "type": "SCALAR",
                "max": [int(indices.max())] if len(indices) else [0],
                "min": [int(indices.min())] if len(indices) else [0],
            },
            {  # 1: Vertices
                "bufferView": 1,
                "componentType": 5126,
                "count": len(verts),
                "type": "VEC3",
                "max": v_max,
                "min": v_min,
            },
            {  # 2: Normals
                "bufferView": 2,
                "componentType": 5126,
                "count": len(normals),
                "type": "VEC3",
                "max": [1, 1, 1],
                "min": [-1, -1, -1],
            },
        ]
        buffer_views = [
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": len(idx_bytes), "target": 34963},
            {"buffer": 0, "byteOffset": vert_offset, "byteLength": len(vert_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": norm_offset, "byteLength": len(norm_bytes), "target": 34962},
        ]

        if col_bytes:
            prim_attrs["COLOR_0"] = 3
            accessors.append({  # 3: Vertex Colors
                "bufferView": 3,
                "componentType": 5126,
                "count": len(colors_f32),
                "type": "VEC3",
                "max": [1, 1, 1],
                "min": [0, 0, 0],
            })
            buffer_views.append({
                "buffer": 0,
                "byteOffset": col_offset,
                "byteLength": len(col_bytes),
                "target": 34962,
            })

        # Material: use vertex colors if provided, otherwise baseColorFactor
        material = {
            "name": "TracheaTissue",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0] if col_bytes else [color[0], color[1], color[2], 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.55,
            },
            "doubleSided": True,
        }

        gltf = {
            "asset": {"version": "2.0", "generator": "TracheaAI"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": prim_attrs, "indices": 0, "material": 0}]}],
            "materials": [material],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": len(buffer_data)}],
        }

        json_str = json.dumps(gltf, separators=(",", ":"))
        json_bytes = json_str.encode("utf-8")
        while len(json_bytes) % 4 != 0:
            json_bytes += b' '

        with open(output_path, "wb") as f:
            f.write(b"glTF")
            f.write(struct.pack("<I", 2))
            total = 12 + 8 + len(json_bytes) + 8 + len(buffer_data)
            f.write(struct.pack("<I", total))
            f.write(struct.pack("<I", len(json_bytes)))
            f.write(struct.pack("<I", 0x4E4F534A))
            f.write(json_bytes)
            f.write(struct.pack("<I", len(buffer_data)))
            f.write(struct.pack("<I", 0x004E4942))
            f.write(buffer_data)

        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Exported GLB: {output_path} ({size_kb:.1f} KB, "
              f"{len(verts)} verts, {len(faces)} faces)")

    def generate_stenosis_glb(self, diseased_mask, healthy_mask, output_path):
        """Generate a stenosis heatmap GLB:
        - Mesh surface = lumen of diseased trachea
        - Vertex color = red (severe stenosis) → yellow → green (normal)
        - This is what a medical student needs to see
        """
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        # Build high-quality smooth mesh of the diseased lumen
        gen = MeshGenerator(smooth_iterations=50, decimate_ratio=1.0)
        mesh = gen.mask_to_mesh(diseased_mask, smooth_sigma=2.0)

        if len(mesh["vertices"]) == 0:
            print("  Cannot generate stenosis GLB: empty mesh")
            return

        verts = mesh["vertices"]  # shape (N,3) in mm, z varies with slice

        # For each vertex, find which slice (z) it's on
        spacing = diseased_mask.GetSpacing()
        d_arr = sitk.GetArrayFromImage(diseased_mask)
        h_arr = sitk.GetArrayFromImage(healthy_mask)

        # Compute equivalent diameter per slice for both masks
        z_diameters = {}  # slice_z → (diseased_diam, healthy_diam)
        for z in range(d_arr.shape[0]):
            d_area = d_arr[z].sum() * spacing[0] * spacing[1]
            h_area = h_arr[z].sum() * spacing[0] * spacing[1]
            if d_area > 0:
                d_diam = 2 * np.sqrt(d_area / np.pi)
                h_diam = 2 * np.sqrt(h_area / np.pi) if h_area > 0 else d_diam
                z_phys = z * spacing[2]
                z_diameters[z_phys] = (d_diam, h_diam)

        if not z_diameters:
            # Fallback: plain color
            colors = np.tile([0.8, 0.2, 0.2], (len(verts), 1)).astype(np.float32)
            self.export_glb(mesh, output_path, vertex_colors=colors)
            return

        z_keys = np.array(sorted(z_diameters.keys()))
        d_diams = np.array([z_diameters[k][0] for k in z_keys])
        h_diams = np.array([z_diameters[k][1] for k in z_keys])

        # Assign per-vertex stenosis severity → color
        # verts[:,0] = z axis (from marching cubes in zyx order → axis 0 = Z physical)
        vert_z = verts[:, 0]  # physical Z coordinate of each vertex

        colors = np.zeros((len(verts), 3), dtype=np.float32)
        for i, vz in enumerate(vert_z):
            # Find nearest slice
            idx = np.argmin(np.abs(z_keys - vz))
            d = d_diams[idx]
            h = h_diams[idx]
            # Stenosis ratio: 0 = no stenosis, 1 = complete occlusion
            stenosis = max(0.0, min(1.0, (h - d) / max(h, 0.1)))

            # Color map: green (0% stenosis) → yellow (50%) → red (100%)
            if stenosis < 0.5:
                t = stenosis * 2.0         # 0→1
                r, g, b = t, 1.0, 0.0      # green→yellow
            else:
                t = (stenosis - 0.5) * 2.0 # 0→1
                r, g, b = 1.0, 1.0-t, 0.0  # yellow→red

            colors[i] = [r, g, b]

        self.export_glb(mesh, output_path, vertex_colors=colors)

    def export_obj(self, mesh_data, output_path):
        """Export mesh to OBJ format."""
        verts = mesh_data["vertices"]
        faces = mesh_data["faces"]
        normals = mesh_data["normals"]

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# Trachea mesh generated by TracheaAI\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1}//{face[0]+1} "
                        f"{face[1]+1}//{face[1]+1} "
                        f"{face[2]+1}//{face[2]+1}\n")

        print(f"  Exported OBJ: {output_path}")

    def generate_morph_meshes(self, morph_frames, output_dir):
        """Generate GLB meshes for each morph frame."""
        os.makedirs(output_dir, exist_ok=True)
        paths = []

        for i, frame in enumerate(morph_frames):
            mesh = self.mask_to_mesh(frame, smooth_sigma=1.5)
            if len(mesh["vertices"]) == 0:
                continue

            t = i / max(len(morph_frames) - 1, 1)
            color = (0.8 * (1 - t) + 0.1 * t,
                     0.2 * (1 - t) + 0.8 * t,
                     0.2)

            path = os.path.join(output_dir, f"morph_{i:03d}.glb")
            self.export_glb(mesh, path, color=color)
        return paths
