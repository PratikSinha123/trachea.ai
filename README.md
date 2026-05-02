# 🫁 TracheaAI

**AI-Powered Trachea Segmentation, Reconstruction, & Interactive 3D Web Viewer**

TracheaAI is an advanced medical imaging pipeline designed to process DICOM/NIfTI CT scans, intelligently segment the trachea, predict its healthy anatomical state, detect structural anomalies (like stenosis or dilation), and present everything in a premium, browser-based 3D interactive viewer.

---

## 🌟 Key Features

1. **AI Segmentation Engine**: Hybrid rule-based & region-growing segmentor with controlled leakage prevention. (Ready for nnU-Net deep learning integration).
2. **Healthy State Reconstruction**: Uses anatomical shape priors and spline-based interpolation to predict what a diseased/stenotic trachea would look like in a healthy state.
3. **Automated Anomaly Detection**: Automatically flags narrowing (stenosis) and widening (dilation) along the centerline, computing percentage deviations.
4. **Premium 3D Web Viewer**:
   - Built with **Three.js** and **FastAPI**.
   - Real-time morphing animations (Diseased ↔ Healthy).
   - Side-by-side 3D rendering.
   - Interactive 2D CT slice viewer with synchronized cross-section diameter profiles.
5. **High-Performance Pipeline**: Written in Python, utilizing PyTorch with Apple Silicon (MPS) and CUDA support for rapid local or server-based processing.

---

## 🛠️ Architecture Overview

```mermaid
graph LR
    A["DICOM/NIfTI"] --> B["Preprocessing<br>(Resample, Normalize)"]
    B --> C["Segmentation<br>(Hybrid or nnU-Net)"]
    C --> D["Shape Analysis<br>(Anomaly Detection)"]
    D --> E["Reconstruction<br>(Spline Fitting)"]
    E --> F["Mesh Gen<br>(Marching Cubes)"]
    F --> G["FastAPI Backend"]
    G --> H["Three.js Frontend"]
```

---

## 💻 Installation

### Prerequisites
- Python 3.10+
- Git

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/PratikSinha123/trachea.ai.git
   cd trachea.ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Process a New Scan
You can process a raw DICOM folder or an already-converted NIfTI file.

**From DICOM:**
```bash
python3 auto_pipeline.py /path/to/dicom_folder --ai --scan-id patient_001
```

**From NIfTI:**
```bash
python3 auto_pipeline.py --ai --process-nifti /path/to/ct_scan.nii.gz --scan-id patient_001
```

### 2. Launch the 3D Viewer
To start the FastAPI web server and explore the results:
```bash
# Start server only
python3 auto_pipeline.py --server-only --port 8000

# Or process and serve in one command:
python3 auto_pipeline.py /path/to/dicom_folder --ai --serve
```
Open **`http://localhost:8000`** in your browser.

---

## 🌐 Deployment

This project should be deployed as two pieces:

1. **Frontend on Vercel**: Vercel serves the static viewer from `frontend/` plus a tiny `/api/config` endpoint.
2. **AI backend on a Python/container host**: run the FastAPI app on a machine/container that can install PyTorch, SimpleITK, and TotalSegmentator.

### Backend with Docker

Build the backend image:
```bash
docker build -t trachea-ai-backend .
```

If you are building on Apple Silicon for a typical cloud Linux server, build for `linux/amd64`:
```bash
docker build --platform linux/amd64 -t trachea-ai-backend .
```

Run it locally:
```bash
docker run --rm -p 8000:8000 \
  -e TRACHEA_OUTPUT=/app/processed_data \
  -v "$PWD/processed_data:/app/processed_data" \
  -v "$PWD/input_data:/data:ro" \
  -v "$PWD/nnunet_workspace:/app/nnunet_workspace" \
  -v trachea_model_cache:/home/appuser \
  trachea-ai-backend
```

The backend will be available at:
```bash
http://localhost:8000/api/scans
```

If Docker Compose is installed, you can also use:
```bash
docker compose up --build
```

To process local scans through the Docker backend, put files in `input_data/` and use container paths in the app, for example:
```bash
/data/patient_ct.nii.gz
```

For a cloud server, push the image to a registry and run it on the server:
```bash
docker build -t your-registry/trachea-ai-backend:latest .
docker push your-registry/trachea-ai-backend:latest

docker run -d --name trachea-ai-backend \
  --restart unless-stopped \
  -p 8000:8000 \
  -e TRACHEA_OUTPUT=/app/processed_data \
  -v trachea_processed:/app/processed_data \
  -v trachea_model_cache:/home/appuser \
  your-registry/trachea-ai-backend:latest
```

Put HTTPS in front of the backend before connecting it to the Vercel frontend. A Vercel page is served over HTTPS, so browser requests to a plain `http://` backend may be blocked as mixed content.

Use persistent storage for both `/app/processed_data` and `/home/appuser` on the backend host. The first stores generated scan outputs; the second lets model downloads and caches survive container restarts.

Non-Docker backend start command:
```bash
uvicorn server.app:app --host 0.0.0.0 --port $PORT
```

After the backend is live, set this Vercel environment variable:
```bash
TRACHEA_API_BASE_URL=https://your-backend-domain.example.com
```

Then redeploy Vercel. The browser app will call the external backend for `/api/scans`, meshes, slices, and AI processing.

Do not deploy patient scans, generated NIfTI files, GLB outputs, or training workspaces to Vercel. They are ignored by `.vercelignore` and should live on the backend host or object storage.

---

## 🧠 Training the Deep Learning Model (nnU-Net)

TracheaAI is set up to transition from its hybrid segmentor to a state-of-the-art **nnU-Net** deep learning model.

1. **Prepare the Dataset**: Convert your processed data into nnU-Net format.
   ```bash
   python3 data_preparation/nnunet_dataset.py
   ```
2. **Train the Model** (Requires GPU or Apple MPS):
   ```bash
   python3 training/run_nnunet_training.py --fold 0 --epochs 500
   ```

---

## 📂 Repository Structure

- `auto_pipeline.py`: The main CLI entry point.
- `segmentation/`: Preprocessing, hybrid segmentor, and 3D U-Net structures.
- `reconstruction/`: Shape models, anomaly detection, and healthy volume generation.
- `visualization/`: Marching cubes mesh generation and GLB exporting.
- `server/`: FastAPI application and pipeline orchestrator.
- `frontend/`: HTML, CSS, and Three.js JavaScript for the 3D viewer.
- `data_preparation/`: Scripts for converting data to nnU-Net training format.
- `training/`: Wrapper scripts for training AI models.

---

## 📝 License

This project is intended for research and educational purposes. Ensure you have the appropriate permissions and anonymization procedures in place when working with real patient DICOM data.
