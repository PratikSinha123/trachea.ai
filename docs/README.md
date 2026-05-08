# 🫁 TracheaAI: Complete Documentation

Welcome to the official documentation for **TracheaAI**. This directory contains all the information you need to understand, run, and scale the AI pipeline.

## 📚 Table of Contents

1. **[AI Usage & Inference Guide](AI_USAGE.md)** - How to run predictions on new patient scans using the CLI or the Web Viewer.
2. **[Architecture Overview](#architecture-overview)** - A high-level view of how the system works.
3. **[Dataset Management](#dataset-management)** - How to add more data to the model.

---

## Architecture Overview

TracheaAI is an end-to-end framework consisting of three main components:

1. **The Data Pipeline (`auto_pipeline.py`)**: Converts raw DICOMs to NIfTI, handles preprocessing, and generates 3D meshes using the marching cubes algorithm.
2. **The AI Engine (`nnU-Net`)**: A state-of-the-art 3D convolutional neural network that segments the trachea with high precision.
3. **The Web Dashboard**: A FastAPI backend and Three.js frontend that allows clinicians to interact with the 3D models, calculate stenosis, and view the predicted "healthy" state.

## Dataset Management

Currently, the model is trained on a 60GB subset of the LIDC-IDRI dataset. To scale to the full 250GB dataset:

1. Transfer the remaining `.nii.gz` scans to the HPC environment.
2. Place them in the `nnUNet_raw/Dataset001_Trachea/imagesTr` and `labelsTr` directories.
3. Update the `dataset.json` file.
4. Rerun the nnU-Net preprocessing and training scripts on the GPU queue.
