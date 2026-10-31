import serial
import time
import os
import pandas as pd
import winsound  # Built-in Windows sound module

# --- CONFIG ---
COM_PORT = 'COM6' 
BAUD_RATE = 921600 
PACKETS_NEEDED = 400 
SAMPLES_TO_RUN = 20  
SCRAM_TIME = 5 
active_idx = [i for i in range(6, 32)] + [i for i in range(33, 59)]

def play_sound(type="success"):
    if type == "countdown":
        winsound.Beep(600, 100)  # Short blip for 3-2-1
    elif type == "start":
        winsound.Beep(440, 800)  # Long Low Note: "ACTION!"
    elif type == "success":
        winsound.Beep(1200, 200) # High Chirp
        winsound.Beep(1200, 200) # Double beep for success

def record_action(ser, label, count):
    os.makedirs("dataset", exist_ok=True)
    print(f"\n🚀 {label.upper()} - {count} SAMPLES")
    
    for n in range(1, count + 1):
        # --- THE PAUSE ---
        if label == "fall":
            input(f"\n[Sample {n}/{count}] Get on your feet. Press ENTER when standing...")
            print("Stand still! Countdown starting...")
            time.sleep(1) 
        else:
            print(f"\nSample {n}/{count}: Starting in {SCRAM_TIME}s...", end=" ", flush=True)

        # --- THE 3-2-1 COUNTDOWN ---
        for i in range(3, 0, -1):
            print(f"{i}...", end=" ", flush=True)
            play_sound("countdown")
            time.sleep(1)
        
        # --- THE START ---
        play_sound("start") 
        print("RECORDING! 🔴")
        
        csi_window = []
        ser.reset_input_buffer()
        
        while len(csi_window) < PACKETS_NEEDED:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("[CSI],"):
                parts = line.split(",")
                if len(parts) >= 64:
                    raw_amps = [float(x) for x in parts[-64:]]
                    csi_window.append([raw_amps[i] for i in active_idx])
                    if len(csi_window) % 40 == 0: print("█", end="", flush=True)
        
        # --- THE SAVE ---
        pd.DataFrame(csi_window).to_csv(f"dataset/{label}_{int(time.time())}.csv", header=False, index=False)
        play_sound("success")
        print(" ✅ Saved!")

    print(f"\n🎉 ALL {count} SAMPLES OF {label.upper()} COMPLETED!")

try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
    while True:
        print("\n1. EMPTY | 2. WALK | 3. FALL | 4. EXIT")
        choice = input("Choice: ")
        if choice == '1': record_action(ser, "empty", SAMPLES_TO_RUN)
        elif choice == '2': record_action(ser, "walk", SAMPLES_TO_RUN)
        elif choice == '3': record_action(ser, "fall", SAMPLES_TO_RUN)
        elif choice == '4': break
    ser.close()
except Exception as e:
    print(f"❌ Error: {e}")