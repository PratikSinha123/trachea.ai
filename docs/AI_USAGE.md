# 🧠 AI Inference & Usage Guide

This guide explains how to use your trained 3D nnU-Net model to predict and analyze the trachea on *new, unseen* CT scans.

## 1. Web Viewer Inference (Recommended)

The easiest way to use the AI is directly through the TracheaAI web interface.

1. **Start the Web Server**:
   Ensure you are in the `trachea.ai` directory and run:
   ```bash
   python3 -m server.app
   ```
2. **Open the Dashboard**: Navigate to `http://localhost:8000`.
3. **Run a Prediction**:
   - Click the **"NEW ANALYSIS"** button in the top right.
   - Enter the **absolute path** to the raw CT scan on the server (e.g., `/home/pratiksinha1064/dataset/LIDC-IDRI-0099/` or the path to a `.nii.gz` file).
   - Click "Start Analysis".
4. **View Results**: The system will run inference in the background, reconstruct the 3D meshes, and load the results into the 3D viewer.

## 2. Command Line: Single Scan Prediction

If you are working via SSH or writing batch scripts, you can use the CLI to run predictions.

**Command**:
```bash
python3 training/predict.py --input /path/to/raw_ct.nii.gz --scan-id MyNewPatient
```

**What happens under the hood?**
1. The script stages the input for nnU-Net.
2. It runs `nnUNetv2_predict` using your trained weights (`checkpoint_best.pth`).
3. It saves the resulting mask to `processed_data/MyNewPatient/trachea_mask.nii.gz`.
4. It triggers `auto_pipeline.py` to generate the healthy reconstruction, anomalies (stenosis), and `.glb` mesh files.

## 3. Command Line: Auto-Pipeline (DICOM Folders)

If you have a raw DICOM folder and want the pipeline to automatically convert it to NIfTI and run the AI segmentation:

**Command**:
```bash
python3 auto_pipeline.py /path/to/dicom_folder --ai --scan-id DICOM_Patient
```

## 📂 Model Weights Location

Your trained AI model is stored in the nnU-Net workspace.

- **Best Checkpoint**: `nnunet_workspace/nnUNet_results/Dataset001_Trachea/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth`
- **Model Configuration**: `nnunet_workspace/nnUNet_preprocessed/Dataset001_Trachea/nnUNetPlans.json`

Ensure these paths are accessible via your environment variables (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) when running predictions.
