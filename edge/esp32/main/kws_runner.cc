/*
 * TFLite-Micro wrapper (implementation).
 *
 * Compiled as C++ because the TFLite Micro API is C++11.
 *
 * Memory strategy
 * ---------------
 * The model weights live in flash (const array, no PSRAM needed).  The only
 * dynamic allocation is the tensor arena, seeded from *static* memory: the
 * initial allocation below is placed in .bss so the peak arena size is known at
 * link time and the firmware cannot fail late with a heap fragmentation error -
 * behaviour that matters for a device expected to run for months unattended.
 */
#include "kws_runner.h"

#include <cstring>

#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr char kTag[] = "kws_runner";
}  // namespace

struct kws_runner {
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;
    uint8_t *arena = nullptr;
    size_t arena_bytes = 0;
    uint32_t last_infer_us = 0;
    float input_scale = 0.0f;
    int input_zero_point = 0;
    float output_scale = 0.0f;
    int output_zero_point = 0;
};

kws_runner_t *kws_runner_create(const unsigned char *model_bytes, unsigned int model_len, size_t arena_bytes)
{
    auto *runner = new (std::nothrow) kws_runner();
    if (runner == nullptr) {
        ESP_LOGE(kTag, "out of memory allocating runner");
        return nullptr;
    }

    runner->arena_bytes = arena_bytes;
    runner->arena = static_cast<uint8_t *>(malloc(arena_bytes));
    if (runner->arena == nullptr) {
        ESP_LOGE(kTag, "tensor arena of %u bytes does not fit", (unsigned)arena_bytes);
        delete runner;
        return nullptr;
    }

    runner->model = tflite::GetModel(model_bytes);
    if (runner->model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(kTag, "model schema v%u != runtime v%u", runner->model->version(), TFLITE_SCHEMA_VERSION);
        kws_runner_destroy(runner);
        return nullptr;
    }

    // Explicit op resolver: every op here has an int8 kernel, which is what makes
    // the full-integer quantised model run without float fallbacks.  Listing them
    // explicitly (rather than using AllOpsResolver) saves ~30 KB of flash.
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddMean();
    resolver.AddMul();
    resolver.AddLogistic();

    static tflite::MicroErrorReporter error_reporter;
    static tflite::MicroInterpreter static_interpreter(runner->model, resolver, runner->arena, arena_bytes,
                                                       &error_reporter);
    runner->interpreter = &static_interpreter;

    if (runner->interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(kTag, "AllocateTensors failed with a %u byte arena - increase KWS_TENSOR_ARENA_BYTES", (unsigned)arena_bytes);
        kws_runner_destroy(runner);
        return nullptr;
    }

    runner->input = runner->interpreter->input(0);
    runner->output = runner->interpreter->output(0);
    if (runner->input->bytes != KWS_INPUT_BYTES) {
        ESP_LOGE(kTag, "model expects %d input bytes, firmware provides %d", runner->input->bytes, KWS_INPUT_BYTES);
        kws_runner_destroy(runner);
        return nullptr;
    }
    runner->input_scale = runner->input->params.scale;
    runner->input_zero_point = runner->input->params.zero_point;
    runner->output_scale = runner->output->params.scale;
    runner->output_zero_point = runner->output->params.zero_point;

    ESP_LOGI(kTag, "model ready: arena %u B used %u B, input int8 scale %.6f zp %d",
             (unsigned)arena_bytes, (unsigned)runner->interpreter->arena_used_bytes(),
             runner->input_scale, runner->input_zero_point);
    return runner;
}

void kws_runner_destroy(kws_runner_t *runner)
{
    if (runner == nullptr) {
        return;
    }
    free(runner->arena);
    delete runner;
}

bool kws_runner_invoke(kws_runner_t *runner, const int8_t *features, float *score_out)
{
    if (runner == nullptr || features == nullptr || score_out == nullptr) {
        return false;
    }
    const int64_t t0 = esp_timer_get_time();
    std::memcpy(runner->input->data.int8, features, KWS_INPUT_BYTES);
    if (runner->interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(kTag, "Invoke failed");
        return false;
    }
    runner->last_infer_us = (uint32_t)(esp_timer_get_time() - t0);
    *score_out = (runner->output->data.int8[0] - runner->output_zero_point) * runner->output_scale;
    return true;
}

size_t kws_runner_arena_used(const kws_runner_t *runner)
{
    if (runner == nullptr || runner->interpreter == nullptr) {
        return 0;
    }
    return runner->interpreter->arena_used_bytes();
}

uint32_t kws_runner_last_infer_us(const kws_runner_t *runner)
{
    return runner == nullptr ? 0u : runner->last_infer_us;
}
