/*
 * Uplink: the device half of the HTTP contract with the ASR server.
 *
 * Two calls, deliberately tiny:
 *   POST /v1/wake       JSON  -> the server returns the codec it wants
 *   POST /v1/asr/ulaw   body  -> raw G.711 mu-law bytes, 8 kHz
 *
 * Why raw mu-law in the body instead of multipart WAV:
 *   * no container, no base64, no boundary strings - the body *is* the audio,
 *   * 64 kbit/s instead of 256 kbit/s, which is the difference between a link
 *     that works on a weak rural connection and one that does not,
 *   * and the encoder costs a handful of cycles per sample (see kws_g711.h).
 */
#ifndef KWS_UPLINK_H
#define KWS_UPLINK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char base_url[128];
    char wake_path[32];
    char asr_path[32];
    bool wifi_ready;
} kws_uplink_config_t;

/** Initialise networking (Wi-Fi station) and the HTTP client. */
void kws_uplink_init(const char *base_url);

/**
 * POST /v1/wake - announce a local activation.
 * Fire-and-forget from the caller's perspective; returns true if the server
 * acknowledged.
 */
bool kws_uplink_notify_wake(const char *base_url, const char *device_id, const char *keyword, float score,
                            float threshold, int latency_ms, int confirm_windows);

/**
 * Capture the next `duration_ms` of audio, decimate to 8 kHz, compand to mu-law
 * and POST it to /v1/asr/ulaw in small chunks.
 *
 * Streaming in chunks (rather than buffering the whole utterance) is what keeps
 * the time from "user stops speaking" to "transcript" low: the server can begin
 * decoding before the device has finished sending.
 *
 * @return number of mu-law bytes sent, or -1 on failure.
 */
int kws_uplink_stream_ulaw(const char *base_url, const char *device_id, int duration_ms);

#ifdef __cplusplus
}
#endif

#endif /* KWS_UPLINK_H */
