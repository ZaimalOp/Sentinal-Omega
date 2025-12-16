# Sentinal-Omega

# 🦅 SENTINEL OMEGA: Autonomous AI Surveillance System

![System Status](https://img.shields.io/badge/System-ONLINE-brightgreen?style=for-the-badge)
![AI Model](https://img.shields.io/badge/Model-YOLOv8_Medium-blueviolet?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/mAP-78.2%25-success?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Powered_By-DeepFace_%7C_EasyOCR_%7C_Streamlit-blue?style=for-the-badge)

> **"Transforming Passive CCTV into Active Intelligence."**

---

## 📜 Executive Summary
**Sentinel Omega** is an enterprise-grade Computer Vision system designed for real-time forensic analysis. Unlike traditional object detection models, Sentinel Omega integrates **Biometrics (Face ID)**, **Automatic Number Plate Recognition (ANPR)**, and **Geospatial Analytics (Heatmaps)** into a single unified dashboard.

Built on the **YOLOv8** architecture and accelerated by **NVIDIA T4 Tensor Cores**, this system is engineered to detect threats, identify unauthorized personnel, and generate forensic PDF reports automatically.

---

## 🧠 Key Capabilities

### 🛡️ Phase 1: Visual Intelligence
* **Real-Time Object Tracking:** Utilizes **ByteTrack** algorithm to assign persistent IDs to moving targets (counting unique people, not just frames).
* **Thermal Heatmaps:** Generates dynamic heatmaps to visualize high-traffic zones in real-time.
* **Movement Tracers:** Draws trajectory lines to track the path of vehicles and individuals.

### 🕵️ Phase 2: Forensic Intelligence
* **Biometric Access Control:** Integrated **DeepFace (VGG-Face)** engine to recognize registered personnel vs. unknown intruders.
* **Threat Detection:** Specialized logic to flag high-risk objects (Knives, Firearms, Backpacks) with **RED** alert boxes.
* **ANPR / OCR:** Automatic Number Plate Recognition using **EasyOCR** to extract text from vehicle license plates.

---

## 🛠️ Technical Architecture

| Component | Technology Used | Purpose |
| :--- | :--- | :--- |
| **Neural Core** | Ultralytics YOLOv8 (Medium) | High-speed Object Detection (80+ Classes) |
| **Biometrics** | Meta DeepFace | Face Recognition & Identity Verification |
| **Tracking** | ByteTrack / Supervision | Persistent Object ID Tracking |
| **OCR Engine** | EasyOCR (GPU Accelerated) | License Plate Reading |
| **Frontend** | Streamlit | Cyberpunk-styled Mission Dashboard |
| **Reporting** | FPDF | Auto-generation of Forensic PDF Reports |

---

## 📊 Performance Audit
The system was benchmarked against the **COCO128** validation dataset.

* **mAP @ 0.50:** `78.28%` (High Reliability)
* **Precision:** `71.10%` (Trust Score)
* **Recall:** `73.16%` (Coverage Score)
* **Inference Speed:** `~12ms` per frame on T4 GPU.

---

## 🚀 Installation & Setup

### Option 1: Run in Google Colab (Recommended for Demo)
This project is optimized for Cloud GPU execution.
1. Open the notebook.
2. Run the **Installation Cell** to setup the environment.
3. Run the **Launch Protocol** cell.
4. Click the **Cloudflare Tunnel Link** to access the dashboard globally.

### Option 2: Local Installation
**Prerequisites:** Python 3.9+, NVIDIA GPU (Recommended)

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/sentinel-omega.git](https://github.com/your-username/sentinel-omega.git)
cd sentinel-omega

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Dashboard
streamlit run app.py
