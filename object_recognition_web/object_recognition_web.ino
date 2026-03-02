/*
 * ============================================================================
 *  ESP32-S3 Few-Shot Object Recognition — Embedded Firmware
 * ============================================================================
 *
 *  Overview:
 *    This firmware runs on an ESP32-S3 microcontroller and implements a
 *    few-shot object recognition system. The pipeline is:
 *
 *      1. Capture image via ArduCAM OV2640 camera (RGB565, 160×120)
 *      2. Crop center 96×96 region, convert RGB565 → RGB888
 *      3. Feed into MobileNetV2-based TFLite Micro model (INT8 quantized)
 *      4. Extract 128-dimensional embedding vector
 *      5. Compare embeddings using cosine similarity
 *
 *    Communication is over a Wi-Fi HTTP API (JSON responses).
 *    A separate web dashboard connects to this API for the user interface.
 *
 *  Hardware:
 *    - ESP32-S3 with 8MB PSRAM
 *    - ArduCAM OV2640 camera module (SPI interface)
 *
 *  Model:
 *    - MobileNetV2 (α=0.35), trained with few-shot embedding loss
 *    - INT8 fully quantized for microcontroller deployment
 *    - Input: 96×96×3 (uint8) → Output: 128-dim embedding (int8, dequantized)
 *
 *  API Endpoints:
 *    POST /capture  — Capture reference image, extract & store embedding
 *    POST /test     — Capture test image, compare against stored references
 *    POST /reset    — Clear all stored references
 *    GET  /status   — Get current reference count
 *    GET  /metrics  — Get system diagnostics (memory, timing, WiFi, etc.)
 *    GET  /         — API info
 *
 * ============================================================================
 */

// ============================================================================
//  Section 1: INCLUDES
// ============================================================================

#include "embedding_model_int8_esp32_model.h" // INT8 TFLite model as C array
#include "memorysaver.h"                      // ArduCAM memory config
#include <ArduCAM.h>                          // Camera driver
#include <mbedtls/base64.h> // Base64 encoding for image streaming

// Resolve macro conflict between ArduCAM and TFLite
#ifdef swap
#undef swap
#endif

#include "TFLiteMicro_ArduinoESP32S3.h" // TFLite Micro for ESP32-S3
#include <SPI.h>                        // SPI communication (camera)
#include <WebServer.h>                  // HTTP server
#include <WiFi.h>                       // Wi-Fi connectivity
#include <Wire.h>                       // I2C (camera config registers)

// ============================================================================
//  Section 2: CONFIGURATION CONSTANTS
// ============================================================================

// Wi-Fi credentials
const char *WIFI_SSID = "Harsh's Galaxy S23";
const char *WIFI_PASSWORD = "Harsh13261";

// Camera hardware
#define CAMERA_CS_PIN 10  // SPI chip-select pin for ArduCAM
#define CAMERA_WIDTH 160  // Native camera resolution (width)
#define CAMERA_HEIGHT 120 // Native camera resolution (height)

// Model input/output dimensions
#define MODEL_INPUT_SIZE 96 // Model expects 96×96 RGB image
#define MODEL_INPUT_BYTES (MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3) // 27,648
#define EMBEDDING_DIM 128 // Output embedding vector length

// Memory buffers
#define RAW_BUFFER_SIZE 42240 // Camera FIFO buffer (160×120×2 + safety margin)
#define ARENA_SIZE_BYTES (2 * 1024 * 1024) // 2 MB TFLite tensor arena (PSRAM)

// Recognition parameters
#define SIMILARITY_THRESHOLD 0.75f // Cosine similarity threshold for "match"
#define MAX_REFERENCES 3           // Maximum number of reference images

// ============================================================================
//  Section 3: GLOBAL VARIABLES
// ============================================================================

// ----- Hardware objects -----
ArduCAM camera(OV2640, CAMERA_CS_PIN);
WebServer server(80);

// ----- TFLite Micro objects -----
const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *inputTensor = nullptr;
TfLiteTensor *outputTensor = nullptr;
uint8_t *tensorArena = nullptr; // Allocated in PSRAM

// ----- Image buffers (allocated in PSRAM) -----
uint8_t *croppedImageRGB = nullptr; // 96×96×3 = 27,648 bytes (model input)
uint8_t *cameraRawBuffer = nullptr; // Raw FIFO data from camera

// ----- Reference embeddings storage -----
float referenceEmbeddings[MAX_REFERENCES][EMBEDDING_DIM];
int referenceCount = 0;

// ----- Performance metrics -----
unsigned long lastCaptureMs = 0;
unsigned long lastInferenceMs = 0;
unsigned long totalInferences = 0;

// ============================================================================
//  Section 4: UTILITY FUNCTIONS
// ============================================================================

/**
 * Compute cosine similarity between two vectors.
 * Returns value in [-1, 1]; higher = more similar.
 *
 *                    a · b
 *   cos(θ) = ─────────────────
 *              ‖a‖ × ‖b‖
 */
float cosineSimilarity(const float *vecA, const float *vecB, int length) {
  float dotProduct = 0.0f, normA = 0.0f, normB = 0.0f;
  for (int i = 0; i < length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  float denominator = sqrtf(normA) * sqrtf(normB);
  return (denominator < 1e-8f) ? 0.0f : dotProduct / denominator;
}

/**
 * Serialize a float array to a JSON array string.
 * Example: [0.1234, -0.5678, ...]
 */
String floatArrayToJSON(const float *data, int length) {
  String json = "[";
  for (int i = 0; i < length; i++) {
    json += String(data[i], 4);
    if (i < length - 1)
      json += ",";
  }
  json += "]";
  return json;
}

/**
 * Stream the cropped RGB888 image as Base64 over HTTP.
 * Encodes row-by-row to avoid large memory allocations.
 * Each row = 96 pixels × 3 channels = 288 bytes → ~384 Base64 chars.
 */
void streamImageAsBase64() {
  const int bytesPerRow = MODEL_INPUT_SIZE * 3;
  size_t encodedLength = 0;
  unsigned char base64Buffer[512];

  for (int row = 0; row < MODEL_INPUT_SIZE; row++) {
    int offset = row * bytesPerRow;
    mbedtls_base64_encode(base64Buffer, sizeof(base64Buffer), &encodedLength,
                          &croppedImageRGB[offset], bytesPerRow);
    base64Buffer[encodedLength] = '\0';
    server.sendContent((char *)base64Buffer);
  }
}

/**
 * Send CORS headers to allow cross-origin requests from the web dashboard.
 */
void sendCORSHeaders() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}

// ============================================================================
//  Section 5: CAMERA CAPTURE PIPELINE
// ============================================================================

/**
 * Captures an image from the ArduCAM and processes it for the model.
 *
 * Pipeline:
 *   1. Trigger hardware capture into camera's internal FIFO
 *   2. Read raw RGB565 data from FIFO via SPI (byte-by-byte)
 *   3. Crop the center 96×96 region from the 160×120 frame
 *   4. Convert each pixel from RGB565 (16-bit) to RGB888 (24-bit)
 *   5. Store result in croppedImageRGB buffer
 *
 * Returns: true if all 9,216 pixels (96×96) were successfully captured.
 */
bool captureAndProcessImage() {
  Serial.println("[Camera] Starting capture...");

  // Step 1: Trigger camera hardware capture
  camera.flush_fifo();
  camera.clear_fifo_flag();
  camera.start_capture();

  // Wait for capture completion (timeout after 5 seconds)
  uint32_t waitMs = 0;
  while (!camera.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    delay(1);
    if (++waitMs > 5000) {
      Serial.println("[Camera] ERROR: Capture timeout");
      return false;
    }
  }

  // Step 2: Read raw data from camera FIFO via SPI
  uint32_t fifoLength = camera.read_fifo_length();
  Serial.printf("[Camera] FIFO length: %u bytes\n", fifoLength);

  if (fifoLength == 0 || fifoLength > 100000) {
    Serial.println("[Camera] ERROR: Invalid FIFO length");
    return false;
  }

  uint32_t readSize = min(fifoLength, (uint32_t)RAW_BUFFER_SIZE);

  // SPI byte-by-byte read (safe, no burst/DMA issues)
  camera.CS_LOW();
  camera.set_fifo_burst();
  for (uint32_t i = 0; i < readSize; i++) {
    cameraRawBuffer[i] = SPI.transfer(0x00);
  }
  camera.CS_HIGH();

  Serial.printf("[Camera] Read %u bytes from FIFO\n", readSize);

  // Step 3 & 4: Crop center region and convert RGB565 → RGB888
  // Camera frame is 160×120, we extract the center 96×96
  int cropOffsetX = (CAMERA_WIDTH - MODEL_INPUT_SIZE) / 2;  // = 32
  int cropOffsetY = (CAMERA_HEIGHT - MODEL_INPUT_SIZE) / 2; // = 12
  uint32_t pixelsCropped = 0;

  for (int y = 0; y < CAMERA_HEIGHT; y++) {
    for (int x = 0; x < CAMERA_WIDTH; x++) {
      // Check if this pixel falls within the center crop region
      if (x < cropOffsetX || x >= cropOffsetX + MODEL_INPUT_SIZE ||
          y < cropOffsetY || y >= cropOffsetY + MODEL_INPUT_SIZE)
        continue;

      // Read 2-byte RGB565 pixel from raw buffer
      uint32_t rawIndex = (y * CAMERA_WIDTH + x) * 2;
      if (rawIndex + 1 >= readSize)
        break;

      uint8_t highByte = cameraRawBuffer[rawIndex];
      uint8_t lowByte = cameraRawBuffer[rawIndex + 1];

      // RGB565 → RGB888 conversion
      //   RGB565 layout: RRRRRGGG GGGBBBBB (5-6-5 bits)
      uint8_t red = ((highByte >> 3) & 0x1F) * 255 / 31;
      uint8_t green =
          (((highByte & 0x07) << 3) | ((lowByte >> 5) & 0x07)) * 255 / 63;
      uint8_t blue = (lowByte & 0x1F) * 255 / 31;

      // Write to output buffer at the cropped coordinate
      int outIndex =
          ((y - cropOffsetY) * MODEL_INPUT_SIZE + (x - cropOffsetX)) * 3;
      croppedImageRGB[outIndex] = red;
      croppedImageRGB[outIndex + 1] = green;
      croppedImageRGB[outIndex + 2] = blue;
      pixelsCropped++;
    }
  }

  Serial.printf("[Camera] Cropped %u / %d pixels\n", pixelsCropped,
                MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
  return (pixelsCropped == (uint32_t)(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE));
}

// ============================================================================
//  Section 6: TFLITE INFERENCE PIPELINE
// ============================================================================

/**
 * Run the TFLite Micro model on the captured image and extract embedding.
 *
 * Pipeline:
 *   1. Preprocess: Normalize RGB888 pixels and quantize into model input tensor
 *      - MobileNetV2 expects input scaled to [-1, 1]
 *      - INT8 model: apply scale/zero_point quantization parameters
 *   2. Invoke: Run the neural network forward pass
 *   3. Postprocess: Dequantize INT8 output to float32 embedding
 *
 * @param embedding  Output array to store the 128-dim float embedding
 * @return           true if inference succeeded
 */
bool runInference(float *embedding) {
  Serial.println("[Model] Running inference...");

  // Step 1: Fill the model's input tensor with preprocessed image data
  if (inputTensor->type == kTfLiteUInt8) {
    // UINT8 model — raw pixel bytes can be copied directly
    memcpy(inputTensor->data.uint8, croppedImageRGB, MODEL_INPUT_BYTES);

  } else if (inputTensor->type == kTfLiteInt8) {
    // INT8 model — normalize to [-1,1] then quantize using model parameters
    float scale = inputTensor->params.scale;
    int zeroPoint = inputTensor->params.zero_point;
    for (int i = 0; i < MODEL_INPUT_BYTES; i++) {
      float normalized = (croppedImageRGB[i] - 127.5f) / 127.5f; // → [-1, 1]
      inputTensor->data.int8[i] = (int8_t)(normalized / scale + zeroPoint);
    }

  } else {
    // Float32 model — just normalize
    for (int i = 0; i < MODEL_INPUT_BYTES; i++) {
      inputTensor->data.f[i] = (croppedImageRGB[i] - 127.5f) / 127.5f;
    }
  }

  // Step 2: Run the model
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[Model] ERROR: Invoke failed");
    return false;
  }

  // Step 3: Read and dequantize the output embedding
  if (outputTensor->type == kTfLiteInt8) {
    float scale = outputTensor->params.scale;
    int zeroPoint = outputTensor->params.zero_point;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
      embedding[i] = (outputTensor->data.int8[i] - zeroPoint) * scale;
    }
  } else {
    memcpy(embedding, outputTensor->data.f, EMBEDDING_DIM * sizeof(float));
  }

  Serial.println("[Model] Inference complete");
  return true;
}

// ============================================================================
//  Section 7: SETUP — Hardware & System Initialization
// ============================================================================

void setup() {
  Serial.begin(921600);
  delay(2000);
  Serial.println("\n=== ESP32-S3 Few-Shot Object Recognition ===\n");

  // ------------------------------------------------------------------
  //  7a. Camera Initialization (ArduCAM OV2640 over SPI)
  // ------------------------------------------------------------------
  Wire.begin();
  SPI.begin(13, 11, 12, CAMERA_CS_PIN); // SCK, MISO, MOSI, CS
  SPI.beginTransaction(SPISettings(8000000, MSBFIRST, SPI_MODE0));
  pinMode(CAMERA_CS_PIN, OUTPUT);
  digitalWrite(CAMERA_CS_PIN, HIGH);

  // Verify SPI link by writing and reading a test register
  camera.write_reg(ARDUCHIP_TEST1, 0x55);
  if (camera.read_reg(ARDUCHIP_TEST1) != 0x55) {
    Serial.println("[Init] FATAL: Camera SPI communication failed");
    while (1)
      delay(1000);
  }

  // Configure camera: BMP format (raw RGB565), 160×120 resolution
  camera.set_format(BMP);
  camera.InitCAM();
  camera.OV2640_set_JPEG_size(OV2640_160x120);
  delay(1000);
  Serial.println("[Init] Camera ready");

  // ------------------------------------------------------------------
  //  7b. PSRAM & Buffer Allocation
  // ------------------------------------------------------------------
  if (!psramInit() && ESP.getPsramSize() == 0) {
    Serial.println("[Init] FATAL: No PSRAM detected");
    while (1)
      delay(1000);
  }

  tensorArena = (uint8_t *)ps_malloc(ARENA_SIZE_BYTES);
  croppedImageRGB = (uint8_t *)ps_malloc(MODEL_INPUT_BYTES);
  cameraRawBuffer = (uint8_t *)ps_malloc(RAW_BUFFER_SIZE);

  if (!tensorArena || !croppedImageRGB || !cameraRawBuffer) {
    Serial.println("[Init] FATAL: PSRAM allocation failed");
    while (1)
      delay(1000);
  }
  Serial.printf(
      "[Init] PSRAM allocated — Arena: %d KB, Image: %d B, Raw: %d B\n",
      ARENA_SIZE_BYTES / 1024, MODEL_INPUT_BYTES, RAW_BUFFER_SIZE);

  // ------------------------------------------------------------------
  //  7c. TFLite Micro Model Initialization
  // ------------------------------------------------------------------

  // Load the model from the C header array
  tflModel = tflite::GetModel(TFLM_embedding_model_int8_esp32_model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[Init] FATAL: TFLite model schema version mismatch");
    while (1)
      delay(1000);
  }

  // Create the op resolver (lists all operators used by the model)
  static auto opResolver = TFLMgetResolver();

  // Create the interpreter with the tensor arena in PSRAM
  static tflite::MicroInterpreter staticInterpreter(
      tflModel, opResolver, tensorArena, ARENA_SIZE_BYTES);
  interpreter = &staticInterpreter;

  // Allocate memory for all model tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[Init] FATAL: AllocateTensors failed");
    while (1)
      delay(1000);
  }

  // Cache input/output tensor pointers
  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);

  Serial.printf("[Init] TFLite model loaded\n");
  Serial.printf("       Input:  [%d, %d, %d, %d]  dtype=%d\n",
                inputTensor->dims->data[0], inputTensor->dims->data[1],
                inputTensor->dims->data[2], inputTensor->dims->data[3],
                inputTensor->type);
  Serial.printf("       Output: [%d, %d]  dtype=%d\n",
                outputTensor->dims->data[0], outputTensor->dims->data[1],
                outputTensor->type);

  // ------------------------------------------------------------------
  //  7d. Wi-Fi Connection
  // ------------------------------------------------------------------
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[Init] Connecting to WiFi");

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (++attempts > 60) {
      Serial.println("\n[Init] FATAL: WiFi connection timeout");
      ESP.restart();
    }
  }
  Serial.printf("\n[Init] WiFi connected — http://%s\n",
                WiFi.localIP().toString().c_str());

  // ------------------------------------------------------------------
  //  7e. HTTP API Route Registration
  // ------------------------------------------------------------------

  // -- CORS preflight handlers (OPTIONS requests) --
  auto corsHandler = []() {
    sendCORSHeaders();
    server.send(204);
  };
  server.on("/", HTTP_OPTIONS, corsHandler);
  server.on("/status", HTTP_OPTIONS, corsHandler);
  server.on("/metrics", HTTP_OPTIONS, corsHandler);
  server.on("/capture", HTTP_OPTIONS, corsHandler);
  server.on("/test", HTTP_OPTIONS, corsHandler);
  server.on("/reset", HTTP_OPTIONS, corsHandler);

  // -- GET / — API information --
  server.on("/", HTTP_GET, []() {
    sendCORSHeaders();
    server.send(200, "application/json",
                "{\"device\":\"ESP32-S3 Object Recognition\","
                "\"version\":\"2.0\","
                "\"endpoints\":[\"/status\",\"/metrics\",\"/capture\",\"/"
                "test\",\"/reset\"]}");
  });

  // -- GET /status — Reference count --
  server.on("/status", HTTP_GET, []() {
    sendCORSHeaders();
    server.send(200, "application/json",
                "{\"num_references\":" + String(referenceCount) + "}");
  });

  // -- GET /metrics — System diagnostics for dashboard --
  server.on("/metrics", HTTP_GET, []() {
    sendCORSHeaders();
    String json = "{";
    json += "\"free_heap\":" + String(ESP.getFreeHeap()) + ",";
    json += "\"psram_free\":" + String(ESP.getFreePsram()) + ",";
    json += "\"psram_total\":" + String(ESP.getPsramSize()) + ",";
    json += "\"wifi_rssi\":" + String(WiFi.RSSI()) + ",";
    json += "\"cpu_freq_mhz\":" + String(ESP.getCpuFreqMHz()) + ",";
    json += "\"last_inference_ms\":" + String(lastInferenceMs) + ",";
    json += "\"last_capture_ms\":" + String(lastCaptureMs) + ",";
    json += "\"total_inferences\":" + String(totalInferences) + ",";
    json += "\"uptime_sec\":" + String(millis() / 1000) + ",";
    json += "\"num_references\":" + String(referenceCount) + ",";
    json += "\"model_size_kb\":" +
            String(sizeof(TFLM_embedding_model_int8_esp32_model) / 1024) + ",";
    json += "\"arena_size_kb\":" + String(ARENA_SIZE_BYTES / 1024);
    json += "}";
    server.send(200, "application/json", json);
  });

  // -- POST /capture — Capture reference image & store embedding --
  server.on("/capture", HTTP_POST, []() {
    sendCORSHeaders();

    // Guard: check reference limit
    if (referenceCount >= MAX_REFERENCES) {
      server.send(400, "application/json",
                  "{\"success\":false,\"message\":\"Max references reached\"}");
      return;
    }

    // Step 1: Capture image
    unsigned long captureStart = millis();
    if (!captureAndProcessImage()) {
      server.send(500, "application/json",
                  "{\"success\":false,\"message\":\"Capture failed\"}");
      return;
    }
    lastCaptureMs = millis() - captureStart;

    // Step 2: Run inference to get embedding
    float embedding[EMBEDDING_DIM];
    unsigned long inferenceStart = millis();
    if (!runInference(embedding)) {
      server.send(500, "application/json",
                  "{\"success\":false,\"message\":\"Inference failed\"}");
      return;
    }
    lastInferenceMs = millis() - inferenceStart;
    totalInferences++;

    // Step 3: Store the embedding as a reference
    memcpy(referenceEmbeddings[referenceCount], embedding,
           EMBEDDING_DIM * sizeof(float));
    referenceCount++;

    Serial.printf(
        "[Capture] Reference %d saved (capture=%lu ms, inference=%lu ms)\n",
        referenceCount, lastCaptureMs, lastInferenceMs);

    // Step 4: Stream JSON response with embedded Base64 image
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, "application/json", "");

    server.sendContent(
        "{\"success\":true,\"num_references\":" + String(referenceCount) +
        ",\"capture_ms\":" + String(lastCaptureMs) +
        ",\"inference_ms\":" + String(lastInferenceMs) + ",\"image\":\"");
    streamImageAsBase64();
    server.sendContent(
        "\",\"embedding\":" + floatArrayToJSON(embedding, EMBEDDING_DIM) + "}");
  });

  // -- POST /test — Capture test image & compare against references --
  server.on("/test", HTTP_POST, []() {
    sendCORSHeaders();

    // Guard: need at least one reference
    if (referenceCount == 0) {
      server.send(400, "application/json",
                  "{\"success\":false,\"message\":\"No references captured\"}");
      return;
    }

    // Step 1: Capture image
    unsigned long captureStart = millis();
    if (!captureAndProcessImage()) {
      server.send(500, "application/json",
                  "{\"success\":false,\"message\":\"Capture failed\"}");
      return;
    }
    lastCaptureMs = millis() - captureStart;

    // Step 2: Run inference to get test embedding
    float embedding[EMBEDDING_DIM];
    unsigned long inferenceStart = millis();
    if (!runInference(embedding)) {
      server.send(500, "application/json",
                  "{\"success\":false,\"message\":\"Inference failed\"}");
      return;
    }
    lastInferenceMs = millis() - inferenceStart;
    totalInferences++;

    // Step 3: Compare test embedding against each reference embedding
    float similarities[MAX_REFERENCES];
    float sumSimilarity = 0.0f;
    for (int i = 0; i < referenceCount; i++) {
      similarities[i] =
          cosineSimilarity(embedding, referenceEmbeddings[i], EMBEDDING_DIM);
      Serial.printf("  vs Reference %d: %.3f\n", i + 1, similarities[i]);
      sumSimilarity += similarities[i];
    }

    // Compute aggregate statistics
    float avgSimilarity = sumSimilarity / referenceCount;
    float maxSimilarity = similarities[0], minSimilarity = similarities[0];
    for (int i = 1; i < referenceCount; i++) {
      if (similarities[i] > maxSimilarity)
        maxSimilarity = similarities[i];
      if (similarities[i] < minSimilarity)
        minSimilarity = similarities[i];
    }
    bool isMatch = avgSimilarity >= SIMILARITY_THRESHOLD;

    Serial.printf("[Test] Average similarity: %.3f → %s (capture=%lu ms, "
                  "inference=%lu ms)\n",
                  avgSimilarity, isMatch ? "MATCH" : "NO MATCH", lastCaptureMs,
                  lastInferenceMs);

    // Step 4: Stream JSON response with image and all metrics
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, "application/json", "");

    String json = "{\"success\":true";
    json += ",\"similarity\":" + String(avgSimilarity, 4);
    json += ",\"max_similarity\":" + String(maxSimilarity, 4);
    json += ",\"min_similarity\":" + String(minSimilarity, 4);
    json += ",\"match\":" + String(isMatch ? "true" : "false");
    json += ",\"threshold\":" + String(SIMILARITY_THRESHOLD, 2);
    json += ",\"capture_ms\":" + String(lastCaptureMs);
    json += ",\"inference_ms\":" + String(lastInferenceMs);
    json += ",\"total_ms\":" + String(lastCaptureMs + lastInferenceMs);
    json += ",\"image\":\"";
    server.sendContent(json);

    streamImageAsBase64();

    json = "\",\"embedding\":" + floatArrayToJSON(embedding, EMBEDDING_DIM);
    json += ",\"ref_embeddings\":[";
    for (int i = 0; i < referenceCount; i++) {
      json += floatArrayToJSON(referenceEmbeddings[i], EMBEDDING_DIM);
      if (i < referenceCount - 1)
        json += ",";
    }
    json += "]}";
    server.sendContent(json);
  });

  // -- POST /reset — Clear all references --
  server.on("/reset", HTTP_POST, []() {
    sendCORSHeaders();
    referenceCount = 0;
    totalInferences = 0;
    lastInferenceMs = 0;
    lastCaptureMs = 0;
    server.send(200, "application/json", "{\"success\":true}");
  });

  // Start the HTTP server
  server.begin();

  Serial.println("\n========== SYSTEM READY ==========");
  Serial.printf("API: http://%s\n", WiFi.localIP().toString().c_str());
  Serial.println("Endpoints: /status /metrics /capture /test /reset");
  Serial.println("==================================\n");
}

// ============================================================================
//  Section 8: MAIN LOOP — Process incoming HTTP requests
// ============================================================================

void loop() {
  server.handleClient();
  delay(2); // Yield to background Wi-Fi stack
}
