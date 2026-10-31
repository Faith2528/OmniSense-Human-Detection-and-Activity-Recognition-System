#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Configuration Constants
// ============================================================================
#define WIFI_CHANNEL 6
#define RX_BUFFER_SIZE 100
#define JSON_BUFFER_SIZE 2048
#define CSI_SUBCARRIER_COUNT 64

#define TX_SSID "CSI_TX_NODE_A"
#define TX_PASSWORD "csi_transmitter_2026"
#define UDP_PORT 9999

// ============================================================================
// CSI Data Structure
// ============================================================================
typedef struct {
    uint32_t timestamp_ms;
    uint32_t packet_id;
    int8_t rssi;                          // Signal strength (dBm)
    uint8_t data_rate;                    // PHY data rate
    uint16_t channel_width;               // 20, 40, 80 MHz
    uint16_t secondary_channel;           // 0 = none
    uint16_t noise_floor;
    uint8_t num_tones;                    // Number of subcarriers
    uint8_t stbc;                         // STBC info
    uint8_t fec_coding;                   // LDPC FEC coding
    uint8_t sgi;                          // Short GI
    uint16_t subcarriers[CSI_SUBCARRIER_COUNT];  // CSI amplitude for each subcarrier
    uint16_t subcarrier_count;            // Actual number of valid subcarriers extracted
} csi_data_t;

// ============================================================================
// Global Variables
// ============================================================================
static csi_data_t csi_buffer[RX_BUFFER_SIZE];
static uint32_t csi_index = 0;
static uint32_t csi_count = 0;
static uint32_t callback_call_count = 0;
static uint32_t udp_packets_received = 0;
WiFiUDP udp;

// ============================================================================
// CSI Callback - Called when frame is received with CSI data
// ============================================================================
static void wifi_csi_callback(void *ctx, wifi_csi_info_t *csi) {
    if (csi == NULL || csi->buf == NULL) {
        return;
    }
    
    callback_call_count++;
    
    // Extract packet info from PHY header
    wifi_pkt_rx_ctrl_t *rx_ctrl = &csi->rx_ctrl;
    
    // Store CSI data
    csi_data_t *current_csi = &csi_buffer[csi_index % RX_BUFFER_SIZE];
    
    current_csi->timestamp_ms = millis();
    current_csi->rssi = rx_ctrl->rssi;
    current_csi->data_rate = rx_ctrl->rate;
    current_csi->noise_floor = rx_ctrl->noise_floor;
    current_csi->num_tones = rx_ctrl->sig_len;
    current_csi->channel_width = rx_ctrl->cwb;
    current_csi->stbc = rx_ctrl->stbc;
    current_csi->fec_coding = rx_ctrl->fec_coding;
    current_csi->sgi = rx_ctrl->sgi;
    current_csi->packet_id = csi_count;  // Use counter as ID
    
    // Extract I/Q subcarrier data and calculate amplitude
    // csi->buf contains int8_t alternating I, Q values
    // Use csi->len as strict boundary to prevent overflow
    uint16_t subcarrier_idx = 0;
    for (uint16_t i = 0; i < csi->len && subcarrier_idx < CSI_SUBCARRIER_COUNT; i += 2) {
        int16_t i_val = (int16_t)csi->buf[i];      // In-phase
        int16_t q_val = (i + 1 < csi->len) ? (int16_t)csi->buf[i + 1] : 0;  // Quadrature
        
        // Calculate amplitude: sqrt(I^2 + Q^2)
        uint16_t amplitude = (uint16_t)sqrt((float)(i_val * i_val + q_val * q_val));
        current_csi->subcarriers[subcarrier_idx] = amplitude;
        subcarrier_idx++;
    }
    current_csi->subcarrier_count = subcarrier_idx;
    
    csi_index++;
    csi_count++;
}



// ============================================================================
// Output CSI data as barebones CSV
// ============================================================================
void output_csi_csv() {
    // Check for incoming UDP packets
    uint8_t packet_buffer[256];
    int len = udp.parsePacket();
    if (len > 0) {
        udp.read(packet_buffer, len);
        packet_buffer[len] = '\0';
        udp_packets_received++;
    }
    
    // Periodic status report every 5 seconds
    static unsigned long last_status = 0;
    if (millis() - last_status >= 5000) {
        last_status = millis();
        Serial.printf("[STATUS] CSI frames: %lu | UDP packets: %lu | CSI callbacks: %lu\n",
                      csi_count, udp_packets_received, callback_call_count);
        
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[WARN] WiFi connection lost!");
        }
    }
    
    if (csi_count == 0) return;
    
    // Get the most recent CSI entry
    uint32_t idx = (csi_index > 0) ? (csi_index - 1) % RX_BUFFER_SIZE : 0;
    csi_data_t *csi = &csi_buffer[idx];
    
    // Print the metadata separated by commas
    Serial.printf("[CSI],%lu,%d,%u,%u,%u,%lu",
        csi->timestamp_ms,
        csi->rssi,
        csi->noise_floor,
        csi->num_tones,
        csi->data_rate,
        csi_count
    );
    
    // Append the subcarriers, each preceded by a comma
    for (uint16_t i = 0; i < csi->subcarrier_count; i++) {
        Serial.printf(",%u", csi->subcarriers[i]);
    }
    
    // ONE newline at the very end
    Serial.println();
}

// ============================================================================
// SETUP: Initialize ESP32-S3 for CSI Reception as WiFi Client
// ============================================================================
void setup() {
    // Initialize Serial
    Serial.begin(921600);
    delay(500);
    
    Serial.println("\n\n================================");
    Serial.println("ESP32-S3 CSI RX Node (Receiver)");
    Serial.println("================================");
    Serial.flush();
    
    // Log system information
    Serial.printf("[SYS] ESP32-S3 Core Count: %d\n", ESP.getChipCores());
    Serial.printf("[SYS] CPU Frequency: %u MHz\n", ESP.getCpuFreqMHz());
    Serial.printf("[SYS] Free Heap: %u bytes\n", ESP.getFreeHeap());
    Serial.printf("[SYS] PSRAM Free: %u bytes\n", ESP.getFreePsram());
    Serial.flush();
    
    // Turn off WiFi
    WiFi.mode(WIFI_OFF);
    delay(100);
    
    // Initialize WiFi in Station mode
    WiFi.mode(WIFI_STA);
    delay(100);
    
    // Disconnect from any APs
    WiFi.disconnect();
    delay(100);
    
    // Attempt to connect to transmitter's AP
    Serial.println("[CONN] Attempting to connect to transmitter AP...");
    WiFi.begin(TX_SSID, TX_PASSWORD);
    
    unsigned long conn_start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - conn_start < 10000) {
        delay(500);
        Serial.print(".");
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[CONN] Connected to transmitter AP!");
        Serial.printf("[CONN] IP: %s | RSSI: %d dBm\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
        
        // Debug channel and BSSID info
        delay(1000);
        Serial.printf("[DEBUG] Station MAC: %s\n", WiFi.macAddress().c_str());
        Serial.printf("[DEBUG] BSSID: %s\n", WiFi.BSSIDstr().c_str());
        Serial.printf("[DEBUG] Channel: %d\n", WiFi.channel());
        Serial.flush();
    } else {
        Serial.println("\n[CONN] Failed to connect to transmitter AP!");
        return;
    }
    delay(100);
    
    // Start listening for UDP packets
    if (udp.begin(UDP_PORT)) {
        Serial.printf("[UDP] Listening on port %d\n", UDP_PORT);
    } else {
        Serial.println("[UDP] Failed to start UDP listener!");
    }
    delay(100);
    
    // Configure CSI BEFORE registering callback or enabling
    wifi_csi_config_t csi_config;
    memset(&csi_config, 0, sizeof(csi_config));  // Zero-initialize to prevent undefined MAC filter behavior
    csi_config.lltf_en = 1;
    csi_config.htltf_en = 1;
    csi_config.stbc_htltf2_en = 1;
    csi_config.ltf_merge_en = 1;
    csi_config.channel_filter_en = 0;
    csi_config.manu_scale = 0;
    esp_err_t ret1 = esp_wifi_set_csi_config(&csi_config);
    Serial.printf("[CSI] CSI config result: %d (ESP_OK=0)\n", ret1);

    // Register CSI callback BEFORE enabling CSI
    esp_err_t ret2 = esp_wifi_set_csi_rx_cb(wifi_csi_callback, NULL);
    Serial.printf("[CSI] CSI callback register result: %d (ESP_OK=0)\n", ret2);

    // THEN enable CSI capture (must be last)
    esp_err_t ret3 = esp_wifi_set_csi(true);
    Serial.printf("[CSI] CSI enable result: %d (ESP_OK=0)\n", ret3);
    
    Serial.println("[INIT] Receiver initialized successfully");
    Serial.println("[INIT] Waiting for CSI events from transmitter...\n");
    Serial.flush();
}

// ============================================================================
// MAIN LOOP: Continuous CSI Output
// ============================================================================
void loop() {
    static unsigned long last_output = 0;
    unsigned long current_time = millis();
    
    // Output CSI data every 10ms (100 Hz)
    if (current_time - last_output >= 10) {
        output_csi_csv();
        last_output = current_time;
    }
    
    delay(10);  // Yield to WiFi processing
}