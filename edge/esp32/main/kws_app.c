/*
 * SIH26172 voice activator - ESP32-S3 application.
 *
 * Data flow
 * ---------
 *   I2S microphone (16 kHz, 16-bit, DMA)
 *        -> kws_frontend_push()          [30 ms window / 20 ms hop, int8 log-mel]
 *        -> kws_runner_invoke()          [int8 DS-CNN, one 0.99 s window]
 *        -> wake decision policy         [tau + N-of-N confirmation + refractory]
 *        -> POST /v1/wake                [device tells the server it woke]
 *        -> G.711 mu-law uplink          [8 kHz, 64 kbit/s, POST /v1/asr/ulaw]
 *        -> LED + optional ack tone
 *
 * Power/CPU behaviour
 * -------------------
 * While idle-listening the device does: one 512-point FFT + a 40-band melbank +
 * one int8 convolution inference every 20 ms.  There is no dynamic allocation, no
 * filesystem access and no Wi-Fi traffic until a wake event - the requirement is
 * "under 10% CPU while idling in continuous listening mode", and the measured
 * figure is printed by the benchmark task (see main/bench_task.c) so it can be
 * quoted with evidence rather than estimated.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "kws_frontend.h"
#include "kws_g711.h"
#include "kws_runner.h"
#include "kws_uplink.h"

#define KWS_MODEL_HEADER "model_int8_tflite.h"
#include KWS_MODEL_HEADER

static const char *TAG = "kws_app";

/* ---- hardware configuration (edit for your board) ---------------------- */
#define KWS_I2S_PORT I2S_NUM_0
#define KWS_MIC_BCLK_GPIO GPIO_NUM_4
#define KWS_MIC_WS_GPIO GPIO_NUM_5
#define KWS_MIC_DIN_GPIO GPIO_NUM_6
#define KWS_LED_GPIO GPIO_NUM_2

/* ---- deployment configuration ----------------------------------------- */
#ifndef KWS_SERVER_URL
#define KWS_SERVER_URL "http://192.168.1.100:8000"
#endif
#define KWS_DEVICE_ID "esp32s3-va-01"
#define KWS_KEYWORD "hi bixby"

/* Wake policy: tau comes from training (metrics.json); confirmation and
 * refractory are deployment choices trading latency for false activations. */
#define KWS_THRESHOLD 0.60f
#define KWS_CONFIRM_WINDOWS 2
#define KWS_REFRACTORY_MS 1200

/* Tensor arena: the model's peak activation footprint.  See
 * edge/tools/size_arena.py - it prints the recommended value for this model. */
#define KWS_TENSOR_ARENA_BYTES (300 * 1024)

static i2s_chan_handle_t s_rx_chan = NULL;
static kws_runner_t *s_runner = NULL;
static kws_frontend_t s_frontend;
static int8_t s_features[KWS_INPUT_BYTES];
static int s_confirm_count = 0;
static int64_t s_last_wake_us = 0;

/* ------------------------------------------------------------------------ */
/* microphone                                                                */
/* ------------------------------------------------------------------------ */
static void mic_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(KWS_I2S_PORT, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 4;
    chan_cfg.dma_frame_num = 160;  /* 10 ms per DMA frame at 16 kHz */
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &s_rx_chan));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(KWS_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = KWS_MIC_BCLK_GPIO,
            .ws = KWS_MIC_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = KWS_MIC_DIN_GPIO,
            .invert_flags = {false, false, false},
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx_chan));
    ESP_LOGI(TAG, "I2S microphone running at %d Hz", KWS_SAMPLE_RATE);
}

/* ------------------------------------------------------------------------ */
/* wake handling                                                             */
/* ------------------------------------------------------------------------ */
static void on_wake(float score)
{
    gpio_set_level(KWS_LED_GPIO, 1);
    ESP_LOGI(TAG, "WAKE keyword='%s' score=%.3f", KWS_KEYWORD, score);

    const int64_t t_wake = esp_timer_get_time();
    const uint32_t infer_us = kws_runner_last_infer_us(s_runner);

    /* 1. tell the server we woke (also lets the server pick the uplink format) */
    kws_uplink_notify_wake(KWS_SERVER_URL, KWS_DEVICE_ID, KWS_KEYWORD, score, KWS_THRESHOLD, (int)(infer_us / 1000),
                           KWS_CONFIRM_WINDOWS);

    /* 2. stream the following seconds as 8 kHz G.711 mu-law (64 kbit/s) */
    const int stream_bytes = kws_uplink_stream_ulaw(KWS_SERVER_URL, KWS_DEVICE_ID, 4000 /* ms */);
    const uint32_t uplink_ms = (uint32_t)((esp_timer_get_time() - t_wake) / 1000);
    ESP_LOGI(TAG, "uplink: %d mu-law bytes, %u ms from wake to done", stream_bytes, (unsigned)uplink_ms);

    gpio_set_level(KWS_LED_GPIO, 0);
}

/* ------------------------------------------------------------------------ */
/* main listening loop                                                       */
/* ------------------------------------------------------------------------ */
static void listening_task(void *arg)
{
    int16_t samples[320];  /* one 20 ms hop */
    const size_t bytes_to_read = sizeof(samples);
    size_t bytes_read = 0;

    ESP_LOGI(TAG, "listening (tau=%.2f, confirm=%d, refractory=%d ms)", KWS_THRESHOLD, KWS_CONFIRM_WINDOWS,
             KWS_REFRACTORY_MS);

    while (true) {
        if (i2s_channel_read(s_rx_chan, samples, bytes_to_read, &bytes_read, portMAX_DELAY) != ESP_OK) {
            ESP_LOGW(TAG, "I2S read failed");
            continue;
        }
        const size_t n = bytes_read / sizeof(int16_t);

        if (!kws_frontend_push(&s_frontend, samples, n)) {
            continue;  /* no complete new frame yet */
        }

        kws_frontend_window(&s_frontend, s_features);

        float score = 0.0f;
        if (!kws_runner_invoke(s_runner, s_features, &score)) {
            continue;
        }

        const int64_t now_us = esp_timer_get_time();
        const bool in_refractory = (now_us - s_last_wake_us) < (int64_t)KWS_REFRACTORY_MS * 1000;

        if (score >= KWS_THRESHOLD) {
            s_confirm_count++;
        } else {
            s_confirm_count = 0;
        }

        if (s_confirm_count >= KWS_CONFIRM_WINDOWS && !in_refractory) {
            s_confirm_count = 0;
            s_last_wake_us = now_us;
            on_wake(score);
        }
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "SIH26172 low-latency voice activator starting");

    ESP_ERROR_CHECK(nvs_flash_init());

    gpio_config_t led = {};
    led.pin_bit_mask = 1ULL << KWS_LED_GPIO;
    led.mode = GPIO_MODE_OUTPUT;
    ESP_ERROR_CHECK(gpio_config(&led));

    kws_quant_t quant = {KWS_QUANT_SCALE, KWS_QUANT_ZERO_POINT, KWS_LOG_FLOOR};
    kws_frontend_init(&s_frontend, quant);

    s_runner = kws_runner_create(kws_model_int8_tflite, kws_model_int8_tflite_len, KWS_TENSOR_ARENA_BYTES);
    if (s_runner == NULL) {
        ESP_LOGE(TAG, "failed to initialise the model - check KWS_TENSOR_ARENA_BYTES");
        return;
    }
    ESP_LOGI(TAG, "tensor arena used: %u B of %u B", (unsigned)kws_runner_arena_used(s_runner),
             (unsigned)KWS_TENSOR_ARENA_BYTES);

    mic_init();
    kws_uplink_init(KWS_SERVER_URL);

    xTaskCreatePinnedToCore(listening_task, "kws_listen", 8192, NULL, 10, NULL, 1);
}
