# ExecuTorch MLOps Haptic Mat — Quantized Person Detection Pipeline

An end-to-end **MLOps framework** for deploying **quantized person-detection models** on an **Android-based haptic mat system** using **STM32 pressure sensing** and **ExecuTorch**.

This repo extends the original ExecuTorch MobileNet demo with a production-ready pipeline:
**Python PTQ + MLflow + MinIO + Jenkins CI/CD + Android ExecuTorch inference**.

---

## Project Title
**MLOps Framework for Quantized Person Detection Models in Android-Based Haptic Mat Systems with STM32 Pressure Sensing**

## Abstract

This project builds an automated MLOps system for delivering and evaluating **on-device AI models** used in a haptic mat capable of detecting if a user is present, sitting, fully lying, or partially lying.

Pressure data from STM32 bladders is processed by a **quantized person-detection model**, enabling:

* private, offline inference
* reduced operational cost
* automatic app shutdown when no user is detected
* real-time responsiveness on mobile devices

Using **post-training quantization (PTQ)**, the system compresses models, evaluates them via MLflow, stores versions in MinIO, deploys them through Jenkins, and loads them inside an Android ExecuTorch app.

**References:**
* https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202402461
* https://arxiv.org/abs/1712.05877
* https://github.com/google/XNNPACK
* https://www.vulkan.org/nkins, and loads them inside an Android ExecuTorch app.

##  Project Structure & Key Files

Here is an overview of the project directory structure and the location of the most critical files for the MLOps pipeline.

```bash
executorch-mlops-hapticmat/
├── deployment_pipeline.groovy # Jenkins CI/CD pipeline definition script
├── scripts/
│   ├── log_model_to_mlflow.py # Python script to log models/artifacts to MLflow
│   ├── docker-compose.yml     # Docker Compose file to spin up the MLOps stack
│   ├── mv2_xnnpack_build.py   # Script to export/quantize the ExecuTorch model
│   └── ...
├── app/                       # Android application source code
├── models/                    # Folder containing raw/quantized .pte models
├── README.md                  # This documentation file
└── ...
```
## TECH STACK
MLOps Core: MLflow, MinIO, Jenkins, MySQL, Docker

AI/ML Frameworks: PyTorch, ExecuTorch, XNNPACK (CPU backend), Python

Mobile Development: Kotlin, Jetpack Compose, Android SDK

Hardware: STM32 Pressure Sensors

## MLOPS WORKFLOW & PROOF OF CONCEPT

# Phase 1: MLOps Foundation & Tracking (Python & MLflow)
The pipeline begins with a Python script (scripts/log_model_to_mlflow.py) that:

Loads a pre-trained quantized model (.pte).

Calculates its SHA-256 hash for security.

Logs all metadata, key metrics (e.g., latency, size), and the model artifacts to the MLflow Tracking Server.

# Phase 2: CI/CD Pipeline & Model Promotion (Jenkins)
A Jenkins pipeline (deployment_pipeline.groovy) is triggered to deploy the latest successful model run from MLflow. It fetches the artifacts, performs a critical SHA-256 verification to ensure data integrity, and promotes the model to the production bucket.

# Phase 3: Edge Deployment & Over-The-Air (OTA) Update (Android)
The verified model and a newly generated latest.json manifest are stored in the Production Bucket (mlops-test). This bucket is read-only accessible to the Android app.

# KEY FEATURES & PERFORMANCE
100% On-Device Inference: Ensures complete privacy and offline capability.

Real-time Quantized AI: Optimized for mobile CPUs using PTQ and XNNPACK, achieving sub-5ms latency.

Agile OTA Updates: Decoupled model updates from app releases, enabling continuous improvement.

Automated CI/CD Pipeline: Streamlined deployment with Jenkins and SHA-256 security.

Reproducible Setup: Entire backend infrastructure is Dockerized.!

[project_flow.png](graph/project_flow.png)

## HOW TO RUN THE MLOPS PIPELINE
Follow these steps to stand up the entire MLOps infrastructure and run the project locally.

1. Prerequisites
   Docker Desktop (Installed and running)

Python 3.8+

Android Studio (For running the mobile app)

2. Start the Backend Infrastructure (Docker)
   Spin up the MLOps stack (MLflow, MinIO, MySQL, Jenkins) using Docker Compose. This might take a few minutes the first time.


# Navigate to the scripts directory
cd scripts

# Start all services in detached mode
docker compose up -d
Once the command completes, you can access the services at:

MLflow: http://localhost:5001

MinIO Console: http://localhost:9001 (User: minioadmin, Pass: minioadmin)

Jenkins: http://localhost:8080

3. Set up Python Environment
   Create a virtual environment and install the required Python packages for the scripts.

# From the root project directory
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install mlflow boto3 requests torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install executorch

4. Run the MLOps 
Step 1: Log a Model to MLflow (Phase 1) This script simulates the end of a training pipeline, logging a quantized model to MLflow.
python scripts/log_model_to_mlflow.py
Check the MLflow dashboard (http://localhost:5001) to see the new run and artifacts.

Step 2: Trigger Jenkins Deployment (Phase 2)
Go to Jenkins (http://localhost:8080).
Create a new Pipeline job and point it to the deployment_pipeline.groovy script in the repo.
Build the job. Watch it download artifacts from MLflow, validate them, and push them to the MinIO production bucket (mlops-test).

Step 3: Run the Android App (Phase 3)
Open the app/ folder in Android Studio.
Sync Gradle and run the app on an emulator or physical device.
Tap "Load Model". The app will check MinIO, download the new model, and be ready for inference.


