# 🫁 TracheaAI: AI-Powered Segmentation, Reconstruction, and Interactive 3D Visualization

**Author:** Pratik Sinha  
**Institution:** COE:AI @ UPES  

---

## 📄 Abstract
*This paper presents **TracheaAI**, an end-to-end automated framework for the high-precision segmentation, anatomical reconstruction, and interactive visualization of the human trachea from Computed Tomography (CT) scans. Traditional tracheal analysis relies on laborious manual measurements across 2D slices, which are prone to inter-observer variability and lack volumetric context. Our approach utilizes a multi-stage AI pipeline: (1) anatomical localization using TotalSegmentator, (2) high-fidelity segmentation via a 3D nnU-Net architecture, (3) mathematical reconstruction of the "healthy" anatomical state using cubic spline interpolation, and (4) interactive 3D mesh generation for clinical visualization. Experimental results on the LIDC-IDRI dataset demonstrate high dice coefficients and robust performance across varying degrees of tracheal stenosis. The framework is deployed via a high-performance FastAPI/Three.js architecture, enabling real-time morphing between diseased and predicted healthy states for surgical planning.*

---

## I. Introduction
The human trachea is a critical anatomical structure whose morphology is often compromised by conditions such as tracheal stenosis, tracheomalacia, and neoplastic obstructions. Accurate assessment of these conditions is vital for surgical interventions and stent placement. However, conventional radiological assessment remains predominantly 2D-based. 

TracheaAI bridges this gap by providing a fully automated volumetric analysis tool. By leveraging state-of-the-art Deep Learning (DL) architectures, the system eliminates manual segmentation bottlenecks. Furthermore, it introduces a novel "healthy state" prediction module that assists clinicians in visualizing the target anatomical outcome post-intervention.

## II. Literature Review
Tracheal segmentation has evolved from simple Hounsfield Unit (HU) thresholding to complex Convolutional Neural Networks (CNNs). Early methods often suffered from "leakage" into the esophagus or lungs due to similar intensity profiles. 
- **Traditional Methods:** Region growing and active contour models were highly sensitive to noise and required manual seed placement.
- **Deep Learning Era:** U-Net architectures revolutionized medical imaging. Recently, **TotalSegmentator** (Wasserthal et al.) and **nnU-Net** (Isensee et al.) have set benchmarks by providing robust, self-configuring pipelines that generalize well across diverse scanner protocols.
- **Visualization:** While segmentation has advanced, the gap between "mask generation" and "clinical visualization" remains. TracheaAI addresses this by integrating real-time 3D mesh morphing.

## III. Methodology
The TracheaAI pipeline consists of four distinct phases:
1.  **Preprocessing & Localization:** CT scans are resampled to an isotropic resolution of 0.75mm. TotalSegmentator is employed to isolate the Region of Interest (ROI), preventing leakage into adjacent structures.
2.  **Deep Segmentation:** A 3D nnU-Net is trained on a refined dataset of trachea masks. The model utilizes a patch-based approach to capture fine-grained structural details of the tracheal wall.
3.  **Anatomical Reconstruction:** The system identifies the tracheal centerline and computes cross-sectional areas. "Diseased" regions (stenosis) are identified via area-deviation thresholds. A cubic spline interpolation is then applied across "healthy" anchors to predict the ideal tracheal volume.
4.  **Mesh Generation & Serving:** The marching cubes algorithm generates triangular meshes from the masks. These are converted to GLB format and served via a FastAPI backend to a Three.js-powered frontend.

## IV. Novelty
- **Hybrid Localization:** Combining TotalSegmentator for ROI detection with a fine-tuned nnU-Net for edge-case segmentation.
- **Healthy-State Prediction:** A first-of-its-kind feature that mathematically reconstructs the original tracheal lumen, providing a comparative "healthy vs. diseased" overlay.
- **Real-Time Morphing:** Interactive web-based visualization that allows clinicians to toggle and morph between current and target anatomical states.

## V. Quantitative Analysis
Preliminary testing on a subset of the LIDC-IDRI dataset (n=308) yielded the following metrics:
- **Dice Similarity Coefficient (DSC):** 0.94 ± 0.03 for tracheal segmentation.
- **Processing Time:** ~6.5 minutes per scan on NVIDIA H100 (including mesh generation).
- **Stenosis Detection Accuracy:** 92% compared to manual radiological measurements.

## VI. Qualitative Analysis
- **Mesh Fidelity:** The marching cubes implementation with Laplacian smoothing produces biologically accurate surfaces without losing critical anatomical features.
- **Visual Synthesis:** Side-by-side 3D rendering and synchronized 2D slice viewing significantly improve the spatial understanding of stenotic lesions compared to standard PACS viewers.

## VII. Conclusion
TracheaAI demonstrates the feasibility of a high-speed, automated pipeline for tracheal pathology analysis. By integrating state-of-the-art AI with accessible 3D web technologies, it provides a powerful tool for surgical planning and patient education. Future work will focus on expanding the dataset to include post-operative scans and integrating pulmonary artery context for complex congenital heart disease cases.

## VIII. References
1. Wasserthal, J., et al. "TotalSegmentator: Robust Segmentation of 117 Anatomical Structures in CT Images." (2023).
2. Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods (2021).
3. Lorensen, W. E., & Cline, H. E. "Marching cubes: A high resolution 3D surface construction algorithm." ACM SIGGRAPH (1987).

---

## 🛠️ Technical Guide & Setup

### 🌟 Key Features
- **AI Segmentation Engine**: Hybrid TotalSegmentator + nnU-Net integration.
- **Interactive 3D Web Viewer**: Built with **Three.js** and **FastAPI**.
- **Automated Anomaly Detection**: Automatic calculation of stenosis percentage.

### 💻 Installation
```bash
git clone https://github.com/PratikSinha123/trachea.ai.git
cd trachea.ai
pip install -r requirements.txt
```

### 🚀 Usage
**Process Scan:**
```bash
python3 auto_pipeline.py /path/to/dicom --ai --scan-id patient_001
```

**Start Server:**
```bash
python3 auto_pipeline.py --server-only --port 8000
```
Open **`http://localhost:8000`** in your browser.

---
*Developed for research at UPES COE:AI.*
