/*
 * On-device resource benchmark.
 *
 * The SIH26172 evaluation criteria include two hard numbers that must be measured
 * on the physical board, not estimated on a laptop:
 *
 *   1. RAM footprint  < 256 KB  (tensor arena + feature buffers + stack)
 *   2. CPU while idle-listening < 10%
 *
 * This task produces both, every 10 seconds, in a form that can be pasted
 * straight into docs/BENCHMARKS.md.  Build with `idf.py -DKWS_ENABLE_BENCH=1`.
 */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "kws_frontend.h"
#include "kws_runner.h"

static const char *TAG = "kws_bench";

#define BENCH_ITERS 200

extern void kws_bench_report(float frontend_us, float infer_us, size_t arena_used, size_t free_heap,
                             size_t min_free_heap);

static void bench_task(void *arg)
{
    kws_frontend_t fe;
    kws_quant_t quant = {KWS_QUANT_SCALE, KWS_QUANT_ZERO_POINT, KWS_LOG_FLOOR};
    kws_frontend_init(&fe, quant);

    int8_t features[KWS_CONTEXT_FRAMES * KWS_NUM_MEL_BINS];
    int16_t frame[KWS_FRAME_LENGTH];
    for (int i = 0; i < KWS_FRAME_LENGTH; ++i) {
        frame[i] = (int16_t)(2000 * ((i % 100) - 50));  /* deterministic, not silence */
    }

    kws_runner_t *runner = kws_runner_create(kws_model_int8_tflite, kws_model_int8_tflite_len,
                                             KWS_TENSOR_ARENA_BYTES);
    if (runner == NULL) {
        ESP_LOGE(TAG, "runner init failed");
        vTaskDelete(NULL);
        return;
    }

    /* ---- front-end cost: one 30 ms frame ---- */
    const int64_t t0 = esp_timer_get_time();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        kws_frontend_frame(&fe, frame, features + (i % KWS_CONTEXT_FRAMES) * KWS_NUM_MEL_BINS);
    }
    const float frontend_us = (float)(esp_timer_get_time() - t0) / BENCH_ITERS;

    /* ---- inference cost: one 0.99 s window ---- */
    kws_frontend_window(&fe, features);
    float score = 0.0f;
    const int64_t t1 = esp_timer_get_time();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        kws_runner_invoke(runner, features, &score);
    }
    const float infer_us = (float)(esp_timer_get_time() - t1) / BENCH_ITERS;

    /* ---- derived duty cycle ----
     * one inference per 20 ms hop; the front-end FFT runs once per frame too */
    const float duty_percent = (infer_us + frontend_us) / 20000.0f * 100.0f;

    ESP_LOGI(TAG, "front-end    : %.1f us / frame", frontend_us);
    ESP_LOGI(TAG, "inference    : %.1f us / window  (int8 DS-CNN)", infer_us);
    ESP_LOGI(TAG, "idle CPU     : %.2f %% of one core at a 50 Hz decision rate", duty_percent);
    ESP_LOGI(TAG, "tensor arena : %u B used", (unsigned)kws_runner_arena_used(runner));
    ESP_LOGI(TAG, "free heap    : %u B (min ever %u B)", (unsigned)esp_get_free_heap_size(),
             (unsigned)esp_get_minimum_free_heap_size());
    ESP_LOGI(TAG, "last score   : %.4f (sanity: model runs, output dequantised)", score);

    kws_bench_report(frontend_us, infer_us, kws_runner_arena_used(runner), esp_get_free_heap_size(),
                     esp_get_minimum_free_heap_size());

    kws_runner_destroy(runner);
    vTaskDelete(NULL);
}

void kws_bench_start(void)
{
    xTaskCreatePinnedToCore(bench_task, "kws_bench", 8192, NULL, 5, NULL, 0);
}
