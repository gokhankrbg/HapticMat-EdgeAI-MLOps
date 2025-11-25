# ExecuTorch MLOps Haptic Mat — Quantized Person Detection Pipeline

An end-to-end **MLOps framework** for deploying **quantized person-detection models** on an **Android-based haptic mat system** using **STM32 pressure sensing** and **ExecuTorch**.

This repo extends the original ExecuTorch MobileNet demo with a production-ready pipeline:  
**Python PTQ + MLflow + MinIO + Jenkins CI/CD + Android ExecuTorch inference**.

---

## Project Title  
**MLOps Framework for Quantized Person Detection Models in Android-Based Haptic Mat Systems with STM32 Pressure Sensing**

```mermaid
flowchart LR

    %% ========== COMPONENTS ==========
    Script["Python Script<br/>(Model Dev & Log)"]
    MLflow["MLflow Tracking Server"]
    MySQL["MySQL Backend Store"]
    Minio["MinIO Object Storage"]
    Jenkins["Jenkins CI/CD Pipeline"]
    Android["Android App<br/>ExecuTorch Runtime"]

    %% ========== PIPELINES ==========

    %% Script → MLflow
    Script -->|1. Train, Quantize & Log<br/>"(Artifacts & Metrics)"| MLflow

    %% MLflow → Backend Stores
    MLflow -->|runs, params, metrics| MySQL
    MLflow -->|artifacts<br/>"(Staging Bucket)"| Minio

    %% Jenkins → MLflow → MinIO
    Jenkins -->|2. Trigger & Fetch latest run| MLflow
    Jenkins -->|Download artifacts<br/>"(from Staging)"| Minio
    Jenkins -->|3. SHA-256 Verification| Jenkins
    Jenkins -->|4. Promote to Production<br/>"(model.pte + latest.json)"| Minio

    %% Android App → MinIO
    Android -->|5. Check for Updates<br/>"(GET latest.json)"| Minio
    Android -->|Download new model.pte<br/>"(if SHA mismatch)"| Minio
    Android -->|6. On-device Inference<br/>ExecuTorch XNNPACK| Android

    %% STYLES
    classDef server fill:#e7f0ff,stroke:#4a90e2,stroke-width:1px;
    classDef storage fill:#fff7e6,stroke:#e6a500,stroke-width:1px;
    classDef mobile fill:#e6fff2,stroke:#00a86b,stroke-width:1px;
    classDef script fill:#f6f8fa,stroke:#999,stroke-width:1px;
    classDef jenkins fill:#ffecec,stroke:#ff5757,stroke-width:1px;

    class Script script;
    class MLflow server;
    class Jenkins jenkins;
    class MySQL,Minio storage;
    class Android mobile;
```
