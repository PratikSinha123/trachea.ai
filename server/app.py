"""FastAPI backend server for the Trachea AI viewer.

Serves the frontend, processes scans, and provides REST API for 3D data.
"""

import os
import sys

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.pipeline import Pipeline
from segmentation.unet3d import get_device

# Configuration
OUTPUT_ROOT = os.environ.get(
    "TRACHEA_OUTPUT", os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data")
)
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

app = FastAPI(title="TracheaAI", version="1.0.0",
              description="AI-powered trachea segmentation and healthy reconstruction viewer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = get_device()
pipeline = Pipeline(output_root=OUTPUT_ROOT, device=device)

# Track processing status
processing_status = {}


class ProcessRequest(BaseModel):
    path: str
    scan_id: str | None = None
    input_type: str = "auto"  # "dicom", "nifti", or "auto"


@app.get("/api/scans")
async def list_scans():
    """List all processed scans with metadata."""
    return pipeline.list_scans()


@app.get("/api/scan/{scan_id}")
async def get_scan(scan_id: str):
    """Get metadata for a specific scan."""
    meta_path = os.path.join(OUTPUT_ROOT, scan_id, "metadata.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, f"Scan '{scan_id}' not found")

    import json
    with open(meta_path) as f:
        return json.load(f)


@app.get("/api/scan/{scan_id}/mesh/{mesh_type}")
async def get_mesh(scan_id: str, mesh_type: str):
    """Serve a GLB mesh file."""
    allowed = {"diseased", "healthy", "aorta", "pulmonary_artery", "heart", "body"}
    if mesh_type not in allowed:
        raise HTTPException(400, f"mesh_type must be one of {allowed}")

    path = os.path.join(OUTPUT_ROOT, scan_id, "meshes", f"{mesh_type}.glb")
    if not os.path.isfile(path):
        raise HTTPException(404, f"Mesh not found: {mesh_type}")

    return FileResponse(path, media_type="model/gltf-binary",
                        filename=f"{scan_id}_{mesh_type}.glb")


@app.get("/api/scan/{scan_id}/morph/{frame_index}")
async def get_morph_frame(scan_id: str, frame_index: int):
    """Serve a morph animation frame GLB."""
    path = os.path.join(OUTPUT_ROOT, scan_id, "meshes", "morph", f"morph_{frame_index:03d}.glb")
    if not os.path.isfile(path):
        raise HTTPException(404, f"Morph frame {frame_index} not found")

    return FileResponse(path, media_type="model/gltf-binary")


@app.get("/api/scan/{scan_id}/morph_count")
async def get_morph_count(scan_id: str):
    """Get the number of morph frames available."""
    morph_dir = os.path.join(OUTPUT_ROOT, scan_id, "meshes", "morph")
    if not os.path.isdir(morph_dir):
        return {"count": 0}

    count = len([f for f in os.listdir(morph_dir) if f.endswith(".glb")])
    return {"count": count}


@app.get("/api/scan/{scan_id}/slice/{axis}/{index}")
async def get_slice(scan_id: str, axis: str, index: int):
    """Get a 2D slice from the CT scan as a PNG image with mask overlay info."""
    import numpy as np
    import SimpleITK as sitk

    ct_path = _find_ct_nifti(scan_id)

    if ct_path is None:
        raise HTTPException(404, "CT scan not found")

    image = sitk.ReadImage(ct_path)
    arr = sitk.GetArrayFromImage(image)

    # Get slice based on axis
    if axis == "axial":
        if index >= arr.shape[0]:
            raise HTTPException(400, f"Index {index} out of range (max {arr.shape[0]-1})")
        sl = arr[index, :, :]
    elif axis == "coronal":
        if index >= arr.shape[1]:
            raise HTTPException(400, f"Index {index} out of range")
        sl = arr[:, index, :]
    elif axis == "sagittal":
        if index >= arr.shape[2]:
            raise HTTPException(400, f"Index {index} out of range")
        sl = arr[:, :, index]
    else:
        raise HTTPException(400, "axis must be 'axial', 'coronal', or 'sagittal'")

    # Window/level for soft tissue
    sl = np.clip(sl, -1024, 600)
    sl = ((sl + 1024) / 1624 * 255).astype(np.uint8)

    # Convert to RGB so we can overlay color
    import cv2
    rgb_sl = cv2.cvtColor(sl, cv2.COLOR_GRAY2RGB)

    # Try to load and overlay the mask
    mask_path = os.path.join(OUTPUT_ROOT, scan_id, "trachea_mask.nii.gz")
    if os.path.isfile(mask_path):
        try:
            mask_image = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask_image)
            if mask_arr.shape == arr.shape:
                if axis == "axial":
                    m_sl = mask_arr[index, :, :]
                elif axis == "coronal":
                    m_sl = mask_arr[:, index, :]
                elif axis == "sagittal":
                    m_sl = mask_arr[:, :, index]
                
                # Overlay cyan/red color where mask > 0
                overlay = rgb_sl.copy()
                overlay[m_sl > 0] = [52, 211, 153] # Emerald green/cyan for trachea
                
                # Blend with original
                cv2.addWeighted(overlay, 0.35, rgb_sl, 0.65, 0, rgb_sl)
        except Exception as e:
            print(f"Warning: Failed to overlay mask: {e}")

    # Convert to PNG
    from io import BytesIO
    from PIL import Image
    img = Image.fromarray(rgb_sl, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/png")


def _find_ct_nifti(scan_id):
    """Search for the CT NIfTI file across the output directory."""
    skip = {"mask", "healthy", "trachea"}
    scan_dir = os.path.join(OUTPUT_ROOT, scan_id)

    # Check inside the scan subdirectory first
    if os.path.isdir(scan_dir):
        for f in sorted(os.listdir(scan_dir)):
            if f.endswith((".nii.gz", ".nii")) and not any(s in f for s in skip):
                return os.path.join(scan_dir, f)

    # Check the parent output directory for files that look related
    if os.path.isdir(OUTPUT_ROOT):
        for f in sorted(os.listdir(OUTPUT_ROOT)):
            fp = os.path.join(OUTPUT_ROOT, f)
            if os.path.isfile(fp) and f.endswith((".nii.gz", ".nii")) and not any(s in f for s in skip):
                return fp

    return None


@app.get("/api/scan/{scan_id}/dimensions")
async def get_dimensions(scan_id: str):
    """Get the CT scan dimensions for the slice viewer."""
    import SimpleITK as sitk

    ct_path = _find_ct_nifti(scan_id)
    if ct_path is None or not os.path.isfile(ct_path):
        raise HTTPException(404, "CT scan not found")

    image = sitk.ReadImage(ct_path)
    size = image.GetSize()

    return {
        "axial": size[2],
        "coronal": size[1],
        "sagittal": size[0],
    }


def _process_background(path, scan_id, input_type):
    """Background task for scan processing."""
    processing_status[scan_id] = {"status": "processing", "progress": 0}
    try:
        if input_type == "nifti" or path.endswith((".nii", ".nii.gz")):
            result = pipeline.process_nifti(path, scan_id)
        else:
            result = pipeline.process_dicom(path, scan_id)
        processing_status[scan_id] = {"status": "done", "result": result}
    except Exception as e:
        processing_status[scan_id] = {"status": "error", "error": str(e)}


@app.post("/api/process")
async def process_scan(req: ProcessRequest, background_tasks: BackgroundTasks):
    """Start processing a DICOM folder or NIfTI file."""
    if not os.path.exists(req.path):
        raise HTTPException(404, f"Path not found: {req.path}")

    scan_id = req.scan_id or os.path.basename(req.path).replace(".nii.gz", "").replace(".nii", "")

    background_tasks.add_task(_process_background, req.path, scan_id, req.input_type)

    return {"message": "Processing started", "scan_id": scan_id}


@app.get("/api/status/{scan_id}")
async def get_status(scan_id: str):
    """Check processing status."""
    return processing_status.get(scan_id, {"status": "unknown"})


# Serve frontend
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def start_server(host="0.0.0.0", port=8000):
    """Start the server."""
    import uvicorn
    print(f"\n🫁 TracheaAI Server starting on http://localhost:{port}")
    print(f"   Device: {device}")
    print(f"   Output: {OUTPUT_ROOT}")
    print(f"   Frontend: {FRONTEND_DIR}\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
