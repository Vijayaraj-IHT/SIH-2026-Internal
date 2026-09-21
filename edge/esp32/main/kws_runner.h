/*
 * TFLite-Micro wrapper around the exported int8 keyword-spotting model.
 *
 * Owns exactly one thing: the interpreter, its tensor arena and the
 * per-window inference call.  All feature extraction lives in
 * ../common/kws_frontend.c so the same numbers are produced on the host.
 */
#ifndef KWS_RUNNER_H
#define KWS_RUNNER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model geometry is fixed by the training pipeline; keep in sync via
 * edge/common/kws_frontend.h (KWS_CONTEXT_FRAMES x KWS_NUM_MEL_BINS). */
#define KWS_INPUT_FRAMES 49
#define KWS_INPUT_BINS 40
#define KWS_INPUT_BYTES (KWS_INPUT_FRAMES * KWS_INPUT_BINS)

typedef struct kws_runner kws_runner_t;

/**
 * Initialise the interpreter over the model blob compiled into flash.
 *
 * @param arena_bytes  size of the tensor arena to allocate (see
 *                     edge/tools/size_arena.py for how to pick it; the value
 *                     used by the firmware is KWS_TENSOR_ARENA_BYTES).
 * @return NULL on failure (out of memory / model rejected).
 */
kws_runner_t *kws_runner_create(const unsigned char *model, unsigned int model_len, size_t arena_bytes);

void kws_runner_destroy(kws_runner_t *runner);

/**
 * Run one inference on a KWS_CONTEXT_FRAMES x KWS_NUM_MEL_BINS int8 window.
 *
 * @param features  row-major int8 log-mel window (the front-end's output)
 * @param score_out wake probability in [0, 1], dequantised from the int8 output
 * @return true on success
 */
bool kws_runner_invoke(kws_runner_t *runner, const int8_t *features, float *score_out);

/** Bytes of the tensor arena currently in use (for the RAM budget report). */
size_t kws_runner_arena_used(const kws_runner_t *runner);

/** Microseconds spent in the last invoke (on-device timing, no host involved). */
uint32_t kws_runner_last_infer_us(const kws_runner_t *runner);

#ifdef __cplusplus
}
#endif

#endif /* KWS_RUNNER_H */
