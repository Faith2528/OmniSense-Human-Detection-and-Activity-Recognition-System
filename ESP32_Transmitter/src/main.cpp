#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>

// ============================================================================
// Configuration Constants
// ============================================================================
#define WIFI_CHANNEL 6
#define TX_POWER 15
#define PING_INTERVAL_MS 100

// Access Point credentials
#define AP_SSID "CSI_TX_NODE_A"
#define AP_PASSWORD "csi_transmitter_2026"
#define AP_IP IPAddress(192, 168, 1, 1)
#define AP_GATEWAY IPAddress(192, 168, 1, 1)
#define AP_SUBNET IPAddress(255, 255, 255, 0)

// UDP unicast to receiver (forces 802.11n HT-LTF instead of broadcast legacy rates)
#define RECEIVER_IP IPAddress(192, 168, 1, 2)
#define UDP_PORT 9999
#define UDP_PAYLOAD_SIZE 64

// ============================================================================
// Global Variables
// ============================================================================
static uint32_t packet_counter = 0;
static unsigned long last_tx_time = 0;
WiFiUDP udp;

// ============================================================================
// Forward Declarations
// ============================================================================
void send_udp_broadcast();

// ============================================================================
// SETUP: Initialize ESP32 as AP and UDP broadcaster
// ============================================================================
void setup() {
    Serial.begin(921600);
    delay(2000);
    
    Serial.println("\n\n================================");
    Serial.println("ESP32-S3 CSI TX Node (Transmitter)");
    Serial.println("================================");
    Serial.flush();
    
    // Log system information
    Serial.printf("[SYS] ESP32-S3 Core Count: %d\n", ESP.getChipCores());
    Serial.printf("[SYS] CPU Frequency: %u MHz\n", ESP.getCpuFreqMHz());
    Serial.printf("[SYS] Free Heap: %u bytes\n", ESP.getFreeHeap());
    Serial.printf("[SYS] PSRAM Free: %u bytes\n", ESP.getFreePsram());
    Serial.flush();
    
    // Stop any existing WiFi
    WiFi.mode(WIFI_OFF);
    delay(100);
    
    // Set up WiFi as AP
    WiFi.mode(WIFI_AP);
    delay(100);
    
    // Configure AP
    Serial.println("[WIFI] Configuring AP mode...");
    WiFi.softAPConfig(AP_IP, AP_GATEWAY, AP_SUBNET);
    WiFi.softAP(AP_SSID, AP_PASSWORD, WIFI_CHANNEL);
    
    delay(500);
    
    // Log AP info
    Serial.println("[WIFI] AP Mode configured:");
    Serial.printf("[WIFI] SSID: %s\n", AP_SSID);
    Serial.printf("[WIFI] Channel: %d\n", WIFI_CHANNEL);
    Serial.printf("[WIFI] IP: %s\n", WiFi.softAPIP().toString().c_str());
    Serial.flush();
    
    // Set TX power
    esp_wifi_set_max_tx_power(TX_POWER * 4);
    
    // Wait for serial monitor to connect before printing protocol info
    Serial.println("\n[WAIT] Waiting 3 seconds for serial monitor to connect...\n");
    Serial.flush();
    delay(3000);
    
    // Explicitly enable 802.11n (HT) to generate HT-LTF for CSI capture
    uint8_t proto = WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N;
    esp_err_t ret = esp_wifi_set_protocol(WIFI_IF_AP, proto);
    Serial.printf("[WIFI] Set protocol to 11b/g/n result: %d (ESP_OK=0)\n", ret);
    Serial.flush();
    
    // Verify protocol was set
    uint8_t configured_proto = 0;
    esp_wifi_get_protocol(WIFI_IF_AP, &configured_proto);
    Serial.printf("[WIFI] Configured protocol bitmap: 0x%02X\n", configured_proto);
    Serial.flush();
    Serial.printf("[WIFI] 11N enabled: %s\n", (configured_proto & WIFI_PROTOCOL_11N) ? "YES" : "NO");
    Serial.flush();
    
    // Start UDP
    if (udp.begin(UDP_PORT)) {
        Serial.printf("[UDP] Sending unicast to %s:%d (forces 802.11n HT-LTF)\n", RECEIVER_IP.toString().c_str(), UDP_PORT);
    } else {
        Serial.println("[UDP] Failed to start UDP!");
    }
    
    Serial.println("\n[INIT] Transmitter initialized successfully");
    Serial.println("[INFO] Sending unicast UDP packets every 100ms (802.11n HT-LTF)...\n");
    Serial.flush();
}

// ============================================================================
// MAIN LOOP: Continuous UDP broadcast
// ============================================================================
void loop() {
    unsigned long current_time = millis();
    
    // Broadcast UDP packet at regular intervals
    if (current_time - last_tx_time >= PING_INTERVAL_MS) {
        send_udp_broadcast();
        last_tx_time = current_time;
    }
    
    // Print protocol diagnostics every 5 seconds (visible when monitor connects)
    static unsigned long last_proto_check = 0;
    if (current_time - last_proto_check >= 5000) {
        last_proto_check = current_time;
        
        // --- Read the internal temperature sensor ---
        float internal_temp = temperatureRead();
        
        uint8_t configured_proto = 0;
        esp_wifi_get_protocol(WIFI_IF_AP, &configured_proto);
        
        // --- Print the temperature alongside the protocol info ---
        Serial.printf("[PROTO] AP Protocol bitmap: 0x%02X | 11N: %s | Packets sent: %lu | Temp: %.2f C\n", 
                      configured_proto, 
                      (configured_proto & WIFI_PROTOCOL_11N) ? "YES" : "NO",
                      packet_counter,
                      internal_temp);
    }
    
    delay(1);
}

// ============================================================================
// Send UDP broadcast packet
// ============================================================================
void send_udp_broadcast() {
    char payload[UDP_PAYLOAD_SIZE];
    
    // Create payload with packet ID and timestamp
    uint32_t timestamp_ms = millis();
    snprintf(payload, UDP_PAYLOAD_SIZE,
        "{\"id\":%lu,\"ts\":%lu}",
        packet_counter,
        timestamp_ms);
    
    // Send UDP unicast to receiver IP (forces 802.11n MCS rates with HT-LTF)
    udp.beginPacket(RECEIVER_IP, UDP_PORT);
    udp.write((uint8_t*)payload, strlen(payload));
    udp.endPacket();
    
    packet_counter++;
    
    // Log every 10 packets
    if (packet_counter % 10 == 0) {
        Serial.printf("[TX] Packets sent: %lu | Uptime: %lu ms\n", 
                      packet_counter, millis());

    }
}