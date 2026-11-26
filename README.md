# OmniSense: Human Detection and Activity Recognition (HDAR) System

> ⚠️ **Work in Progress:** Additional architectural diagrams and documentation updates are actively being finalized. Core codebase, firmware, and pretrained weights are available below.

![OmniSense Main Dashboard](screenshots/dashboard.jpeg)

## Overview

**OmniSense HDAR** is a dual-modal, non-invasive disaster triage and fall detection platform. It operates by fusing spatial data from an optical camera (powered by **YOLOv8-Pose** estimation) with **Channel State Information (CSI)** extracted from 2.4 GHz Wi-Fi radio waves.

By analyzing subcarrier amplitude and phase variance across **52 active OFDM subcarriers** between two **ESP32-S3** microcontrollers operating in a peer-to-peer network, the system detects macro-movements (such as walking or falling) and isolates micro-movements (such as human respiration rates via Digital Signal Processing). This fused architecture enables the system to differentiate living subjects from inanimate objects, maintain telemetry in visual blind spots, and detect concealed subjects during entrapment scenarios where standalone optical cameras fail.

---

## Key Features

* **Dual-Modal Late Fusion:** Combines RF subcarrier pattern classification (`1D-CNN`) with optical skeletal tracking (`YOLOv8-Pose`) to verify incidents and minimize single-sensor false alarms.
* **Through-Wall & Non-Line-of-Sight Sensing:** Extracts 802.11n CSI packets from dual-core ESP32-S3 nodes at **921,600 baud** to monitor movement through physical obstructions.
* **Respiration Rate Monitoring:** Applies a **0.1–0.5 Hz Butterworth bandpass filter** to stationary RF phase streams to detect chest-wall displacement and estimate respirations per minute (RPM).
* **Real-Time Telemetry & HUD:** Interactive **Streamlit** tactical dashboard displaying live camera overlays, CSI variance waveforms, raw serial packet streams, and automated **MQTT (HiveMQ)** emergency alerts.

---

## System Requirements

### Hardware & Peripherals
* **Deployment Machine:** Windows 10/11, macOS (M-series recommended), or Ubuntu Linux
* **Processor:** Quad-core CPU or better (Intel i5 / AMD Ryzen 5 equivalent or higher)
* **Memory:** Minimum 8 GB RAM (16 GB recommended)
* **Microcontrollers:** 2x ESP32-S3 modules (`ESP32_Transmitter` and `ESP32_Receiver`)
* **Camera:** 1x USB Webcam (e.g., A4Tech PK-910H 1080p) or integrated laptop camera
* **Cables:** 2x High-quality USB-C data cables

### Software Prerequisites
* **Python:** Version 3.9, 3.10, or 3.11
* **IDE:** Visual Studio Code (VS Code) with the **PlatformIO IDE** extension

---

## Repository Structure

```text
OmniSense/
├── dataset/                 # Labeled time-series CSI (.csv) datasets
├── ESP32_Receiver/          # C++ firmware for the ESP32-S3 CSI receiver node
├── ESP32_Transmitter/       # C++ firmware for the ESP32-S3 packet transmitter node
├── fusion_app.py            # Main Streamlit dashboard & live multimodal inference engine
├── record.py                # Audio-guided CSI dataset collection and labeling utility
├── train_local.py           # PyTorch training pipeline for the Rescue1DCNN model
├── rescue_wifi_brain.pth    # Pretrained 1D-CNN model weights
└── yolov8n-pose.pt          # Pretrained YOLOv8-Pose optical tracking weights
```

---

## Installation & Setup

### 1. Firmware Flashing (VS Code & PlatformIO)
1. Install **Visual Studio Code** and add the **PlatformIO IDE** extension (`Ctrl+Shift+X`).
2. Open the `ESP32_Transmitter` and `ESP32_Receiver` project folders in PlatformIO.
3. Verify that any network/broker configurations match your local setup, then compile and flash each firmware to its respective ESP32-S3 board.

![PlatformIO Setup](screenshots/platformio.jpeg)

### 2. Python Virtual Environment
Open a terminal in the project root directory and create an isolated virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install PyTorch, Streamlit, SciPy, Ultralytics, and supporting libraries:

```bash
pip install -r requirements.txt
```

---

## Deployment & Usage

### Hardware Positioning
1. Mount the **ESP32 Transmitter** and **ESP32 Receiver** approximately **90–100 cm** off the ground, spaced **1 to 3 meters** apart.
2. Position the USB webcam directly adjacent to the ESP32 Receiver with an unobstructed view of the primary line-of-sight target area.
3. Ensure the host machine and nodes are connected to the designated Local Area Network (LAN).

### Launching the Application
Run the main Streamlit fusion engine from your activated virtual environment:

```bash
streamlit run fusion_app.py
```

Once the dashboard opens at `http://localhost:8501`:
1. Open the **`[ CONFIG // SYSTEM ]`** sidebar panel.
2. Set **`CAMERA_INDEX`** (default `0` for primary webcam).
3. Enter your active **`COM_PORT`** (e.g., `COM6`).
4. Verify **`BAUD_RATE`** is set to **`921600`** to prevent serial buffer bottlenecks.
5. Click **`[ INITIATE DATA COLLECTION_ ]`** to initialize sensor streams and begin live inference.

![System Configuration Panel](screenshots/configPanel.jpeg)

---

## Dashboard Interface Overview

* **Vision Status Panel (`[ OPTICAL DIAGNOSTICS ]`):** Displays real-time camera tracking and keypoint classifications (e.g., `WALKING`, `LYING DOWN`, `DISTRESS WAVING`).
* **Wi-Fi Status Panel (`[ RF OVERSEER ]`):** Tracks live 75th-percentile subcarrier variance to distinguish empty rooms from active movement.
* **Fusion Alert Panel (`[ GEO. COMMS ]`):** Outputs the combined multimodal emergency probability. Events exceeding the $\ge 60\%$ threshold trigger a `FUSED CRITICAL ALERT` and dispatch MQTT telemetry.
* **CSI Visualizer & Raw Stream (`[ CSI VARIANCE VISUALIZER ]`):** Plots real-time RF scalar fluctuations alongside raw 52-subcarrier packet arrays.

![Dashboard Telemetry Panels](screenshots/dashboardTelemetry.jpeg)

---

## Dataset Collection & Model Training

### Recording Specific Activity Samples (`record.py`)
The data collection pipeline is structured around a **fixed set of predefined target classes** designed for triage and fall-detection scenarios (such as *Empty/Safe*, *Walking*, and *Fall*). Rather than acting as an open-ended gesture recorder, `record.py` standardizes sample capture specifically for these trained states:

1. Run `python record.py` in your terminal.
2. Select the target action label corresponding to the predefined class you are recording (e.g., `1` for Empty/Safe, `2` for Walking, `3` for Fall).
3. Follow the automated audio cues:
   * **Countdown:** Prepare in the target zone.
   * **Long Low Note:** Execute the designated movement or posture immediately.
   * **High Chirp:** Capture complete; the 52-subcarrier time-series sample is saved to the `dataset/` directory.

### Retraining the Classifier (`train_local.py`)
`train_local.py` is specifically configured to train and fine-tune the **`Rescue1DCNN`** architecture on our predefined activity classes using collected 52-subcarrier CSV sequences:

1. Ensure all recorded `.csv` class files are placed inside the `/dataset` directory.
2. Execute the training script:
   ```bash
   python train_local.py
   ```
3. The script validates packet dimensions, normalizes the subcarrier sequences, and optimizes the 1D-CNN weights across **60 epochs**.
4. Once training finishes, the updated weights overwrite `rescue_wifi_brain.pth` and are automatically loaded by `fusion_app.py` upon restart.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
