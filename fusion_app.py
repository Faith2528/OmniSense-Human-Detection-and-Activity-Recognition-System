import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import torch
import torch.nn as nn
from ultralytics import YOLO
import serial
import threading
from collections import deque
import time
import paho.mqtt.client as mqtt
import json
import ssl
from datetime import datetime

# --- 1. MQTT CLOUD SETUP (THE TOWER) ---
MQTT_BROKER = "701845da75c342f3841d057778cdc110.s1.eu.hivemq.cloud"  
MQTT_PORT = 8883
MQTT_USER = "wifi_sensing_camera_vision"                 
MQTT_PASS = "WiFi*Sensing*Camera*Vision*./.*64"                 
MQTT_TOPIC = "rescue/system/status"

# We use a global state to prevent reconnecting constantly in Streamlit
if "mqtt_client" not in st.session_state:
    print("🌐 Connecting to Cloud Broker...")
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.username_pw_set(MQTT_USER, MQTT_PASS)
        client.tls_set(tls_version=ssl.PROTOCOL_TLS)
        client.connect(MQTT_BROKER, MQTT_PORT)
        client.loop_start()
        st.session_state.mqtt_client = client
        st.session_state.last_published_status = ""
        print("✅ Cloud Connected.")
    except Exception as e:
        print(f"⚠️ Cloud Connection Failed. Bypassing to OFFLINE MODE: {e}")
        st.session_state.mqtt_client = None
        st.session_state.last_published_status = ""

# --- 2. WIFI MODEL ARCHITECTURE (PyTorch) ---
class Rescue1DCNN(nn.Module):
    def __init__(self):
        super(Rescue1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(52, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 3) 
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --- 3. CONFIG & GLOBALS ---
WINDOW_SIZE = 400
PRESENCE_THRESHOLD = 0.8  
MOTION_THRESHOLD = 1.8    
active_idx = [i for i in range(6, 32)] + [i for i in range(33, 59)]
classes = ['SAFE', 'WALKING', 'FALL_DETECTED']
# --- DSP: RESPIRATION EXTRACTION ---
def extract_respiration_rate(csi_window, fps=10):
    """
    Applies a Butterworth Bandpass Filter to isolate breathing frequencies (0.1 - 0.5 Hz)
    from the raw CSI amplitude matrix.
    """
    try:
        # 1. Average the 52 subcarriers to get a single unified room waveform
        waveform = np.mean(csi_window, axis=1)
        
        # 2. Filter setup (Nyquist is half the sample rate)
        nyquist = 0.5 * fps
        low = 0.1 / nyquist   # Breathing lower bound
        high = 0.5 / nyquist  # Breathing upper bound
        b, a = butter(3, [low, high], btype='band')
        
        # 3. Apply the filter to strip away static and noise
        clean_breath_signal = filtfilt(b, a, waveform)
        
        # 4. Count the wave peaks
        peaks, _ = find_peaks(clean_breath_signal)
        
        # 5. Calculate Respirations Per Minute (RPM)
        window_seconds = len(csi_window) / fps
        rpm = (len(peaks) / window_seconds) * 60
        
        return int(rpm)
    except Exception:
        return 0 # Failsafe if buffer is momentarily corrupted
    
@st.cache_resource
def load_models():
    print("🧠 Loading Models...")
    yolo_model = YOLO('yolov8n-pose.pt') 
    wifi_model = Rescue1DCNN()
    try:
        wifi_model.load_state_dict(torch.load('rescue_wifi_brain.pth', map_location=torch.device('cpu'), weights_only=True))
        wifi_model.eval()
        wifi_loaded = True
        print("✅ Models Loaded.")
    except Exception as e:
        print(f"Error loading wifi model: {e}")
        wifi_loaded = False
    return yolo_model, wifi_model, wifi_loaded

yolo_model, wifi_model, wifi_loaded = load_models()

# Globals for threading
csi_buffer = deque(maxlen=WINDOW_SIZE)
raw_serial_log = deque(maxlen=1)

def serial_worker(com_port, baud_rate):
    """Background thread to read ESP32 Serial Data"""
    try:
        ser = serial.Serial(com_port, baud_rate, timeout=1) 
        raw_serial_log.append("Connected to port successfully. Waiting for data...")
        
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                raw_serial_log.append(line) 
                
            if line.startswith("[CSI],"):
                parts = line.split(",")
                if len(parts) >= 64:
                    raw_amps = [float(x) for x in parts[-64:]]
                    csi_buffer.append([raw_amps[i] for i in active_idx])
    except Exception as e:
        raw_serial_log.append(f"SERIAL ERROR: {e}")

# --- 4. STREAMLIT UI SETUP ---
st.set_page_config(page_title="OMNI SENSE", layout="wide", page_icon="🎛️")

# INJECT CYBERPUNK / HUD CSS
st.markdown("""
    <style>
    /* Dark theme & font */
    body, .stApp {
        background-color: #030303 !important;
        background-image: radial-gradient(#1a1a1a 1px, transparent 1px);
        background-size: 20px 20px;
        font-family: 'Courier New', Courier, monospace !important;
        color: #dcdcdc !important;
    }
    
    /* Modify headers to match tactical HUD */
    h1, h2, h3, h4, h5, p {
        color: #dcdcdc !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 2px !important;
    }
    h1 {
        border-bottom: 1px solid #444;
        padding-bottom: 5px;
        color: #fff !important;
    }
    h3 {
        font-size: 14px !important;
        border-bottom: 1px dotted #555;
        padding-bottom: 5px;
        color: #888 !important;
    }
    
    /* Metrics and Data boxes */
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: bold;
        font-size: 24px !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #777 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 10px !important;
    }
    .stMetric {
        background-color: rgba(10, 10, 10, 0.9); 
        padding: 5px 10px; 
        border: 1px solid #222; 
        border-left: 2px solid #555;
    }
    
    /* Alert Banners (HUD Style) */
    div[data-testid="stAlert"] {
        background-color: rgba(0, 30, 30, 0.8) !important;
        border: 1px solid #00ffcc !important;
        border-left: 5px solid #00ffcc !important;
        border-radius: 0px !important;
        color: #fff !important;
    }
    div[data-testid="stAlert"] div[role="alert"] {
        color: #fff !important;
        font-weight: bold;
        letter-spacing: 1px;
    }
    div[data-testid="stAlert"]:has(.st-dz) {
        border-color: #ff3333 !important;
        background-color: rgba(30, 0, 0, 0.8) !important;
    }
    
    /* Markdown / Text */
    .stMarkdown code {
        background-color: rgba(0, 40, 40, 0.6) !important;
        color: #00ffcc !important;
        border: 1px solid #005555;
        border-radius: 0px !important;
        padding: 4px;
        font-size: 14px !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #0d0d0d !important;
        border: 1px solid #333 !important;
        color: #aaa !important;
        border-radius: 0px !important;
        font-size: 10px !important;
        letter-spacing: 2px;
        padding: 5px 20px !important;
    }
    .stButton button:hover {
        border-color: #ff3333 !important;
        color: #ff3333 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px dotted #333;
    }
    
    /* Tactical box wrapper */
    .tactical-box {
        border: 1px solid #222;
        padding: 10px;
        background: rgba(10, 10, 10, 0.8);
        position: relative;
    }
    .tactical-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 10px; height: 10px;
        border-top: 1px solid #888;
        border-left: 1px solid #888;
    }
    .tactical-box::after {
        content: '';
        position: absolute;
        bottom: 0; right: 0;
        width: 10px; height: 10px;
        border-bottom: 1px solid #888;
        border-right: 1px solid #888;
    }
    .red-text {
        color: #ff3333 !important;
    }
    .grey-text {
        color: #777 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### `[ CONFIG // SYSTEM ]`")
cam_index = st.sidebar.selectbox("CAMERA_INDEX", [0, 1, 2, 3], index=0)
com_port = st.sidebar.text_input("COM_PORT", value="COM6")
baud_rate = st.sidebar.number_input("BAUD_RATE", value=921600)
st.sidebar.markdown("<br><p class='grey-text'>// END CONFIG</p>", unsafe_allow_html=True)

st.markdown("""
<div style='display: flex; justify-content: space-between; border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 15px;'>
    <div>
        <h1 style='border: none; padding: 0; margin: 0; font-size: 24px;'> OMNI SENSE <span style='font-size: 12px; color: #555;'>// TELEREMETRY NETWORK</span></h1>
        <span style='color: #666; font-size: 10px;'>CSI_RF_VISION_FUSION // v4.2.1</span>
    </div>
    <div style='text-align: right;'>
        <div style='color: #ff3333; font-weight: bold;'>00. 00. 00</div>
        <div style='color: #555; font-size: 10px;'>TIMER _ SYS_ACTIVE</div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.6, 1])

with col1:
    st.markdown("### `[ LIVE CAMERA FEED_1 ]`")
    st.markdown("<div class='tactical-box'>", unsafe_allow_html=True)
    camera_feed = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### `[ OPTICAL DIAGNOSTICS ]`")
        vision_status = st.empty()
    with c2:
        st.markdown("### `[ RF OVERSEER ]`")
        wifi_status = st.empty()
    with c3:
        st.markdown("### `[ GEO. COMMS ]`")
        fusion_alert = st.empty()

with col2:
    st.markdown("### `[ BIOMETRIC / RF DOSSIER ]`")
    
    st.markdown("<div class='tactical-box'>", unsafe_allow_html=True)
    
    # --- THE NEW NODE ARRAY ---
    n1, n2 = st.columns(2)
    with n1:
        node_metric_1 = st.empty()
    with n2:
        node_metric_2 = st.empty()
        
    st.markdown("<hr style='margin: 10px 0; border-color: #333;'>", unsafe_allow_html=True)
    
    # --- THE ORIGINAL 4 METRICS ---
    m1, m2 = st.columns(2)
    with m1:
        rf_metric_1 = st.empty()
    with m2:
        rf_metric_2 = st.empty()
        
    m3, m4 = st.columns(2)
    with m3:
        rf_metric_3 = st.empty()
    with m4:
        rf_metric_4 = st.empty()
    st.markdown("</div><br>", unsafe_allow_html=True)
    
    st.markdown("### `[ TARGET TELEMETRY ]`")
    st.markdown("<div class='tactical-box'>", unsafe_allow_html=True)
    hud_info = st.empty()
    st.markdown("</div><br>", unsafe_allow_html=True)
    
    st.markdown("### `[ CSI VARIANCE VISUALIZER ]`")
    st.caption(">> LIVE_RF_SCALAR")
    csi_chart = st.empty()
    
    st.markdown("<br>### `[ DB STREAM / RAW PACKETS ]`", unsafe_allow_html=True)
    st.markdown("<div class='tactical-box' style='height: 150px; overflow: hidden;'>", unsafe_allow_html=True)

    serial_debug_text = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    
    if not wifi_loaded:
        st.error("ERR_404: rescue_wifi_brain.pth OFFLINE")

# --- 5. THE MAIN LOOP ---
if st.button("[ INITIATE DATA COLLECTION_ ]", type="primary"):
    
    threading.Thread(target=serial_worker, args=(com_port, int(baud_rate)), daemon=True).start()
    cap = cv2.VideoCapture(cam_index)
    
    # --- START CONTINUOUS INFERENCE LOOP ---
    last_predict_time = time.time()
    continuous_motion_frames = 0
    idle_start_time = None # NEW: Tracks when the room enters PRESENCE state
    
    # Define these OUTSIDE the loop so they persist across camera frames!
    wifi_state = "BUFFERING"
    vision_state = "CLEAR"
    signal_variance = 0.0
    conf_pct = 100.0
    final_status = "SAFE"
    
    fused_fall_prob = 0.0
    wifi_prob_fall = 0.0
    vis_prob_fall = 0.0
    
    live_variance_log = deque(maxlen=60)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if len(raw_serial_log) > 0:
            serial_debug_text.code(raw_serial_log[-1])
            
        # -- A. VISION PIPELINE (From app.py) --
        # We only process vision every other frame to save heavy YOLO processing time
        if int(time.time() * 30) % 2 == 0:
            vision_state = "CLEAR"
            results = yolo_model(frame, verbose=False)
            
            for r in results:
                if r.keypoints is not None and len(r.keypoints.xy) > 0:
                    keypoints = r.keypoints.xy[0].cpu().numpy()
                    
                    if len(keypoints) >= 13:
                        nose_y = keypoints[0][1]
                        hip_y = (keypoints[11][1] + keypoints[12][1]) / 2.0
                        wrist_l_y = keypoints[9][1]
                        wrist_r_y = keypoints[10][1]
                        
                        valid_x = [kp[0] for kp in keypoints if kp[0] > 0]
                        valid_y = [kp[1] for kp in keypoints if kp[1] > 0]
                        skel_width = max(valid_x) - min(valid_x) if valid_x else 0
                        skel_height = max(valid_y) - min(valid_y) if valid_y else 0
                        
                        if wrist_l_y < nose_y and wrist_r_y < nose_y and wrist_l_y > 0 and wrist_r_y > 0:
                            vision_state = "DISTRESS WAVING"
                        elif nose_y > hip_y and nose_y > 0 and hip_y > 0:
                            vision_state = "FALL DETECTED"
                        elif skel_width > (skel_height * 1.3) and skel_width > 0:
                            vision_state = "LYING DOWN"
            
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            camera_feed.image(annotated_frame, channels="RGB", use_container_width=True)
            
        # -- B. LIVE WIFI PIPELINE (From main_app.py) --
        # We NO LONGER initialize wifi_state = "BUFFERING" here, because the camera loop runs 
        # 30 times a second, but prediction runs twice a second. Redefining it here caused the flash!
        
        # Calculate these so the UI always has something, even if BUFFERING
        
        if len(csi_buffer) == WINDOW_SIZE:
            data = np.array(csi_buffer)
            
            # The Math Fix (75th Percentile)
            subcarrier_variances = np.var(data, axis=0)
            signal_variance = np.percentile(subcarrier_variances, 75)
            live_variance_log.append(signal_variance)
            
            if (time.time() - last_predict_time) > 0.5:
                # Gate Logic & Prolonged Idle Timer
                if signal_variance < PRESENCE_THRESHOLD:
                    continuous_motion_frames = 0 
                    wifi_state = "SAFE"
                    idle_start_time = None # Reset timer
                    
                elif signal_variance < MOTION_THRESHOLD:
                    continuous_motion_frames = 0 
                    
                    if idle_start_time is None:
                        idle_start_time = time.time()
                    
                    idle_duration = time.time() - idle_start_time
                    if idle_duration > 15.0: # 15-second trigger
                        wifi_state = "PROLONGED_IDLE"
                    else:
                        wifi_state = "PRESENCE"
                else: # AI Control
                    continuous_motion_frames += 1
                    idle_start_time = None # Reset timer because they are moving
                    data_norm = (data - np.mean(data)) / (np.std(data) + 1e-8) 
                    data_tensor = torch.tensor(data_norm.T, dtype=torch.float32).unsqueeze(0) 
                    
                    with torch.no_grad():
                        output = wifi_model(data_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        confidence, predicted_class = torch.max(probabilities, 1)
                        
                        idx = predicted_class.item()
                        conf_pct = confidence.item() * 100
                        
                        if continuous_motion_frames < 3:
                            wifi_state = "TRACKING MOTION"
                        else:
                            wifi_state = classes[idx] # SAFE, WALKING, or FALL_DETECTED
                            
                last_predict_time = time.time()
                
            # -- C. LATE FUSION & SMART MQTT LOGGING --
            # Decide final system status (Soft Voting Fusion)
            final_status = wifi_state
            
            wifi_prob_fall = 0.0
            vis_prob_fall = 0.0

            # 1. Extract Wi-Fi Fall Probability
            if wifi_state == "FALL_DETECTED":
                wifi_prob_fall = conf_pct / 100.0  # Real AI confidence (e.g., 0.95)
            elif wifi_state in ["SAFE", "PRESENCE", "WALKING", "TRACKING MOTION"]:
                wifi_prob_fall = 0.05 # Low baseline probability

            # 2. Synthesize Vision Fall Probability (Heuristic)
            if vision_state == "FALL DETECTED":
                vis_prob_fall = 0.95 # Highly confident heuristic
            elif vision_state in ["LYING DOWN", "DISTRESS WAVING"]:
                vis_prob_fall = 0.60 # Ambiguous distress
            elif vision_state == "CLEAR":
                vis_prob_fall = 0.05 # Baseline safe
                
            # 3. Soft Voting Formula (Weighted Average)
            # Wi-Fi (0.6) + Vision (0.4) = Fused Fall Probability
            fused_fall_prob = (wifi_prob_fall * 0.6) + (vis_prob_fall * 0.4)
            
            if fused_fall_prob >= 0.60:
                final_status = "FALL_DETECTED"
            
            # Check Edge-Cloud Gatekeeper rules (Only publish on change or periodic fall spam)
            last_status = st.session_state.last_published_status
            
            # Create a rate-limiter for continuous fall alerts (e.g., every 3 seconds)
            if "last_fall_spam_time" not in st.session_state:
                st.session_state.last_fall_spam_time = 0.0
                
            is_new_status = (final_status != last_status)
            is_periodic_fall_spam = (final_status == "FALL_DETECTED" and (time.time() - st.session_state.last_fall_spam_time) >= 3.0)
            
            if final_status and (is_new_status or is_periodic_fall_spam):
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Update spam timer if it's an ongoing fall
                if final_status == "FALL_DETECTED":
                    st.session_state.last_fall_spam_time = time.time()
                
                payload = {
                    "device_id": "Room_1_Node",
                    "timestamp": current_time,
                    "status": final_status, 
                    "vision_status": vision_state,
                    "wifi_status": wifi_state,
                    "fused_probability": round(fused_fall_prob * 100, 2),
                    "variance": round(signal_variance, 2)
                }
                if st.session_state.mqtt_client is not None:
                    st.session_state.mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
                st.session_state.last_published_status = final_status
        else:
            pct = (len(csi_buffer) / WINDOW_SIZE) * 100
            
            # Show buffering state
            node_metric_1.metric(label="N1: MAIN LOS", value="WAIT")
            node_metric_2.metric(label="N2: DEADSPOT", value="WAIT")
            rf_metric_1.metric(label="SUB_VARIANCE", value="WAIT")
            rf_metric_2.metric(label="WIFI_PROB", value="BUFF")
            rf_metric_3.metric(label="VIS_PROB", value="BUFF")
            rf_metric_4.metric(label="FUSED_PROB", value=f"{pct:.1f}%")

        # UI Updates
        # Update metrics for valid state
        if int(time.time() * 10) % 2 == 0:
            if len(csi_buffer) == WINDOW_SIZE:
                simulated_deadspot = signal_variance * 0.3 if signal_variance > 5 else 1.2
                    
                # Update the new Node metrics
                node_metric_1.metric(label="N1: MAIN LOS", value=f"{signal_variance:.2f}")
                node_metric_2.metric(label="N2: DEADSPOT", value=f"{simulated_deadspot:.2f}")
                
                # Restore the original 4 metrics
                rf_metric_1.metric(label="SUB_VARIANCE", value=f"{signal_variance:.2f}")
                rf_metric_2.metric(label="WIFI_PROB", value=f"{wifi_prob_fall * 100:.1f}%")
                rf_metric_3.metric(label="VIS_PROB", value=f"{vis_prob_fall * 100:.1f}%")
                rf_metric_4.metric(label="FUSED_PROB", value=f"{fused_fall_prob * 100:.1f}%")
                
                # Simple line chart for CSI Variance
                csi_chart.line_chart(list(live_variance_log), height=120)

            # --- DYNAMIC HUD & CONTINUOUS TRIAGE LOGIC ---
            critical_states = ["FALL DETECTED", "LYING DOWN", "DISTRESS WAVING"]
            is_emergency_event = final_status == "FALL_DETECTED" or vision_state in critical_states
            
            real_rpm = 0
            real_rpm_display = "0 RPM"
            MIN_BREATHING_VAR = 0.15 # Minimum variance to trust the DSP (filters out empty room static)
            
            # 1. CONSTANT RADAR SWEEP
            if signal_variance >= MOTION_THRESHOLD:
                # Subject is actively moving/squirming. Too noisy for DSP, but movement = life.
                real_rpm = -1 
            elif signal_variance >= MIN_BREATHING_VAR:
                # Room is quiet enough for DSP, but active enough to contain actual breathing
                extracted_rpm = extract_respiration_rate(data, fps=10)
                if 10 <= extracted_rpm <= 35: # Gate it to realistic human breathing rates
                    real_rpm = extracted_rpm
            else:
                # Absolute dead silence. Room is genuinely empty or object is inanimate.
                real_rpm = 0

            # 2. DETERMINE TACTICAL STATE
            if real_rpm == -1:
                # --- STATE: LIVING HUMAN (MACRO-MOTION) ---
                bio_class = "LIVING HUMAN (ACTIVE MOTION)"
                bio_color = "#00ffcc"
                real_rpm_display = "N/A (MOVING)"
                
                if is_emergency_event:
                    box_color = "#ff3333"
                    bg_color = "rgba(30, 0, 0, 0.8)"
                    header_text = "🫀 CRITICAL TRIAGE: CONSCIOUS PATIENT DETECTED"
                else:
                    box_color = "#00ffcc"
                    bg_color = "rgba(0, 30, 30, 0.8)"
                    header_text = "🫀 LIFE SIGNS: TARGET IN MOTION"

            elif real_rpm > 0:
                # --- STATE: LIVING HUMAN (IDLE/UNCONSCIOUS) ---
                bio_class = "LIVING HUMAN (STABLE)"
                bio_color = "#00ffcc"
                real_rpm_display = f"{real_rpm} RPM"
                
                if is_emergency_event:
                    box_color = "#ff3333"
                    bg_color = "rgba(30, 0, 0, 0.8)"
                    header_text = "🫀 CRITICAL TRIAGE: BREATHING DETECTED"
                elif vision_state == "CLEAR" and wifi_state == "SAFE":
                    box_color = "#ff9900" 
                    bg_color = "rgba(30, 15, 0, 0.8)"
                    header_text = "🫀 WARNING: CONCEALED HUMAN / BLINDSPOT"
                else:
                    box_color = "#00ffcc"
                    bg_color = "rgba(0, 30, 30, 0.8)"
                    header_text = "🫀 LIFE SIGNS: NOMINAL (IDLE)"
            
            else:
                # --- STATE: NO LIFE SIGNS ---
                bio_class = "NO LIFE SIGNS"
                bio_color = "#777777"
                real_rpm_display = "0 RPM"
                
                if is_emergency_event:
                    box_color = "#555555"
                    bg_color = "rgba(10, 10, 10, 0.8)"
                    header_text = "🫀 SYSTEM ALERT: INANIMATE OBJECT DROPPED"
                else:
                    box_color = "#333333"
                    bg_color = "rgba(5, 5, 5, 0.8)"
                    header_text = "🫀 RADAR SWEEP: CLEAR"

            # Render the combined HUD
            hud_info.markdown(f"""
            `ID:` **NK_CAM_9** <br>
            `WIFI_STATE:` **{wifi_state}** <br>
            `LENS_STATE:` **{vision_state}** <br>
            `SYS_DECISION:` **{final_status}** <br>
            
            <div style='border: 1px solid {box_color}; padding: 10px; background: {bg_color}; margin-top: 10px; transition: border-color 0.3s ease;'>
                <div style='color: {box_color}; font-weight: bold; margin-bottom: 5px;'>{header_text}</div>
                `RESPIRATION:` **{real_rpm_display}**<br>
                `BIO_CLASS  :` <b style='color: {bio_color};'>{bio_class}</b>
            </div>
            """, unsafe_allow_html=True)
            
            
            # --- UI ALERTS UPDATES ---
            if vision_state in ["FALL DETECTED", "DISTRESS WAVING", "LYING DOWN"]:
                vision_status.error(f"👁️ VISION: {vision_state}")
            else:
                vision_status.success("👁️ VISION: CLEAR")
                
            if wifi_state == "BUFFERING":
                wifi_status.markdown(f"📡 `BUFFERING`")
            elif wifi_state == "SAFE":
                wifi_status.markdown(f"🟩 `EMPTY ROOM (Var: {signal_variance:.1f})`")
            elif wifi_state == "PRESENCE":
                wifi_status.markdown(f"🟦 `PRESENCE DETECTED (Idle | Var: {signal_variance:.1f})`")
            elif wifi_state in ["WALKING", "TRACKING MOTION"]:
                wifi_status.markdown(f"📊 🟧 `WALKING / MOVEMENT ({conf_pct:.1f}% | Var: {signal_variance:.1f})`")
            elif wifi_state == "FALL_DETECTED":
                wifi_status.markdown(f"🚨 🟥 `FALL DETECTED ({conf_pct:.1f}% | Var: {signal_variance:.1f})`")
                
            if final_status == "FALL_DETECTED":
                fusion_alert.error(f"🚨 FUSED CRITICAL ALERT 🚨  \nPROB: {fused_fall_prob*100:.1f}% | UPLINK SENT.")
            else:
                try:
                    current_fuse = fused_fall_prob * 100
                    fusion_alert.success(f"[+] GEO COMMS SECURE  \nPROB: {current_fuse:.1f}%")
                except NameError:
                    fusion_alert.success("[+] GEO COMMS SECURE  \nPROB: 0.0%")
                    
            time.sleep(0.01) # Small sleep to prevent CPU hogging in while loop