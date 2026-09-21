/*
 * G.711 mu-law companding - the uplink codec.
 *
 * Rationale for the SIH26172 "minimal data overhead" requirement:
 *   16-bit / 16 kHz linear PCM = 256 kbit/s
 *   8-bit  /  8 kHz G.711 mu-law =  64 kbit/s   (4x smaller)
 * while remaining the format every telephony-trained ASR back-end expects.
 *
 * The encoder below is bit-exact with the Python/NumPy reference in
 * ml/common/audio.py (`ulaw_encode`), which tests/test_g711.py verifies against
 * the ITU-T G.711 definition over the full 16-bit input range.
 */
#ifndef KWS_G711_H
#define KWS_G711_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_ULAW_BIAS 0x84  /* 132 */
#define KWS_ULAW_CLIP 32635

/** Encode one signed 16-bit sample to G.711 mu-law. */
static inline uint8_t kws_ulaw_encode_sample(int16_t sample)
{
    int32_t s = sample;
    uint8_t sign = 0;
    if (s < 0) {
        sign = 0x80;
        s = -s;
    }
    if (s > KWS_ULAW_CLIP) s = KWS_ULAW_CLIP;
    s += KWS_ULAW_BIAS;

    int exponent = 7;
    for (int mask = 0x4000; (s & mask) == 0 && exponent > 0; mask >>= 1) {
        exponent--;
    }
    const int mantissa = (s >> (exponent + 3)) & 0x0F;
    return (uint8_t)(~(sign | (exponent << 4) | mantissa));
}

/** Encode a block of int16 PCM into `out` (same length as `n`). */
static inline void kws_ulaw_encode(const int16_t *pcm, size_t n, uint8_t *out)
{
    for (size_t i = 0; i < n; ++i) {
        out[i] = kws_ulaw_encode_sample(pcm[i]);
    }
}

/** Decode one mu-law byte back to a signed 16-bit sample (for loopback tests). */
static inline int16_t kws_ulaw_decode_sample(uint8_t ulaw)
{
    uint8_t u = (uint8_t)(~ulaw);
    const int sign = u & 0x80;
    const int exponent = (u >> 4) & 0x07;
    const int mantissa = u & 0x0F;
    int sample = ((mantissa << 3) + KWS_ULAW_BIAS) << exponent;
    sample -= KWS_ULAW_BIAS;
    return (int16_t)(sign ? -sample : sample);
}

/**
 * Decimate 16 kHz PCM to 8 kHz before companding (2:1 with a short FIR).
 * Uses a 15-tap half-band filter stored in the generated tables header.
 */
void kws_downsample_16k_to_8k(const int16_t *in, size_t n_in, int16_t *out, size_t *n_out);

#ifdef __cplusplus
}
#endif

#endif /* KWS_G711_H */
