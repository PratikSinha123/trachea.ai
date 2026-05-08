# 🧠 TracheaAI: Model Usage Guide

Congratulations! The **nnU-Net 3D Deep Learning model** has successfully completed its training on the LIDC-IDRI dataset with an excellent Mean Validation Dice Score of **~0.88**. 

This guide explains how to use your trained AI to predict and analyze the trachea on *new, unseen* CT scans.

---

## 1. Web Viewer Inference (Recommended)

The easiest way to use the AI is directly through the TracheaAI web interface. 

1. **Start the Web Server**:
   Make sure the backend is running.
   ```bash
   cd trachea.ai
   python3 -m server.app
   ```
2. **Open the Dashboard**: Navigate to `http://localhost:8000`.
3. **Run a Prediction**:
   - Click the **"🧠 AI Predict"** button.
   - Enter a **Scan ID** (e.g., `Patient_999`).
   - Provide the **absolute path** to the raw CT scan on the server (e.g., `/home/pratiksinha1064/dataset/LIDC-IDRI-0099/scan.nii.gz` or a DICOM folder).
4. **View Results**: The system will run inference in the background, reconstruct the 3D meshes, and automatically load the results into the 3D viewer when finished.

---

## 2. Command Line: Single Scan Prediction

If you are working via SSH or writing batch scripts, you can use the CLI to run predictions. This method takes a raw CT NIfTI file, runs the AI, and builds the full pipeline (including stenosis calculations and 3D meshes).

**Command**:
```bash
cd /home/pratiksinha1064/trachea.ai
python3 training/predict.py --input /path/to/raw_ct.nii.gz --scan-id MyNewPatient
```

**What happens?**
1. The script stages the input for nnU-Net.
2. It runs `nnUNetv2_predict` using your trained weights (`checkpoint_best.pth`).
3. It saves the resulting mask to `processed_data/MyNewPatient/trachea_mask.nii.gz`.
4. It triggers the `auto_pipeline.py` to generate the healthy reconstruction, anomalies (stenosis), and `.glb` mesh files.
5. The result is immediately available in the web viewer dropdown.

*Note: If you only want the mask and want to skip the 3D mesh generation, add the `--no-pipeline` flag.*

---

## 3. Command Line: Auto-Pipeline (DICOM Folders)

If you have a raw DICOM folder and want the pipeline to automatically convert it to NIfTI and run the AI segmentation:

**Command**:
```bash
cd /home/pratiksinha1064/trachea.ai
python3 auto_pipeline.py /path/to/dicom_folder --ai --scan-id DICOM_Patient
```

---

## 📂 Where are the Model Weights?

Your trained AI model ("the brains") is stored in the nnU-Net workspace. If you need to back up the model, deploy it to a different server, or share it, you need to copy this file:

**Best Checkpoint**:
`/home/pratiksinha1064/trachea.ai/nnunet_workspace/nnUNet_results/Dataset001_Trachea/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth`

**Model Configuration (Required alongside the checkpoint)**:
`/home/pratiksinha1064/trachea.ai/nnunet_workspace/nnUNet_preprocessed/Dataset001_Trachea/nnUNetPlans.json`

---

## ⚙️ Environment Variables

The inference scripts automatically locate the model using the default workspace. However, if you move the `nnunet_workspace` folder, ensure these environment variables are set before running predictions:

```bash
export nnUNet_raw="/path/to/nnunet_workspace/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnunet_workspace/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnunet_workspace/nnUNet_results"
```