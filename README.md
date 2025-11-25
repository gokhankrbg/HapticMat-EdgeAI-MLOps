# HapticMat-EdgeAI-MLOps: End-to-End Quantized Person Detection Pipeline

An end-to-end **MLOps framework** for deploying **quantized person-detection models** on an **Android-based haptic mat system** using **STM32 pressure sensing** and **ExecuTorch**.

This repository demonstrates a complete, production-ready pipeline connecting PyTorch model development with on-device Android runtime:
**Data Collection → PyTorch Training → PTQ Quantization → MLflow Tracking → MinIO Storage → Jenkins CI/CD → Android ExecuTorch Inference (XNNPACK)**.

Developed in collaboration with **Seroton GmbH**.

---

## Table of Contents

1.  [Introduction & Abstract](#1-introduction--abstract)
2.  [System Architecture & Tech Stack](#2-system-architecture--tech-stack)
3.  [Model Development Process](#3-model-development-process)
4.  [MLOps Workflow & Proof of Concept](#4-mlops-workflow--proof-of-concept)
5.  [Android Application Integration](#5-android-application-integration)
6.  [How to Run the Pipeline](#6-how-to-run-the-pipeline)
7.  [Conclusion & Future Work](#7-conclusion--future-work)
8.  [References](#8-references)

---

## 1. Introduction & Abstract

This project builds an automated MLOps system for delivering and evaluating **on-device AI models** used in a haptic mat capable of detecting user states (present, sitting, lying).

Pressure data from STM32 sensor bladders is processed by a **quantized person-detection model (MobileNetV2)** running directly on an Android device.

**Key Benefits:**
* ✅ **Private, Offline Inference:** No data leaves the device.
* ✅ **Real-time Responsiveness:** Sub-30ms latency on mobile CPUs.
* ✅ **Reduced Operational Cost:** No reliance on cloud servers for inference.
* ✅ **Automated Lifecycle:** Continuous integration and deployment via Jenkins.

Using **Post-Training Quantization (PTQ)**, the system compresses models, evaluates them via MLflow, stores versions in MinIO, deploys them through a secure Jenkins pipeline, and loads them dynamically inside an Android ExecuTorch app via Over-The-Air (OTA) updates.

---

## 2. System Architecture & Tech Stack

### High-Level Architecture

This diagram illustrates the complete end-to-end MLOps flow, from the initial Python script to the final on-device inference on Android.

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
    Script -->|1. Train, Quantize & Log<br/>(Artifacts & Metrics)| MLflow

    %% MLflow → Backend Stores
    MLflow -->|runs, params, metrics| MySQL
    MLflow -->|artifacts<br/>(Staging Bucket)| Minio

    %% Jenkins → MLflow → MinIO
    Jenkins -->|2. Trigger & Fetch latest run| MLflow
    Jenkins -->|Download artifacts<br/>(from Staging)| Minio
    Jenkins -->|3. SHA-256 Verification| Jenkins
    Jenkins -->|4. Promote to Production<br/>(model.pte + latest.json)| Minio

    %% Android App → MinIO
    Android -->|5. Check for Updates<br/>(GET latest.json)| Minio
    Android -->|Download new model.pte<br/>(if SHA mismatch)| Minio
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
