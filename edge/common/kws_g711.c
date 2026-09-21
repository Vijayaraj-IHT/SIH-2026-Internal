/*
 * 16 kHz -> 8 kHz decimation for the G.711 uplink.
 *
 * A half-band FIR followed by 2:1 decimation.  Doing this properly matters for
 * ASR accuracy: naive sample-dropping aliases everything above 4 kHz back into
 * the speech band, which is exactly where the recogniser expects to find
 * fricatives.  The coefficients live in the generated tables header so they are
 * the ones the host-side parity test uses.
 */
#include "kws_g711.h"

#include "kws_frontend_tables.h"

void kws_downsample_16k_to_8k(const int16_t *in, size_t n_in, int16_t *out, size_t *n_out)
{
    const int taps = KWS_DECIM_FIR_TAPS;
    const int half = taps / 2;
    size_t produced = 0;

    for (size_t i = 0; i < n_in; i += 2) {
        float acc = 0.0f;
        for (int t = 0; t < taps; ++t) {
            const long idx = (long)i - half + t;
            float sample = 0.0f;
            if (idx >= 0 && idx < (long)n_in) {
                sample = (float)in[idx] / 32768.0f;
            }
            acc += sample * KWS_DECIM_FIR[t];
        }
        float v = acc * 32768.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        out[produced++] = (int16_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
    }
    *n_out = produced;
}
