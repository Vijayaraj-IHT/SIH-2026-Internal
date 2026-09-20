/*
 * Fixed-size sliding-window log-mel front-end (see kws_frontend.h).
 *
 * Numerical notes
 * ---------------
 * * The FFT is a plain radix-2 float32 implementation.  The ESP32-S3 has a
 *   single-precision FPU, and 512-point FFTs at 50 frames/s cost well under 1%
 *   of the CPU, so a float FFT is the right engineering trade here: it keeps
 *   the code short enough to audit and matches the training-time spectrum to
 *   ~1e-6.  The *model* is full int8 - that is where the memory and MAC budget
 *   matters.
 * * The mel filterbank and Hann window are compiled in as constants generated
 *   from the exact Python objects used in training (bit-identical).
 * * log() is taken in float and then affine-quantised to int8 with the
 *   scale/zero-point recorded at export time, so no per-device calibration is
 *   needed and the server sees the same numbers.
 */
#include "kws_frontend.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------------ */
/* radix-2 in-place FFT                                                      */
/* ------------------------------------------------------------------------ */
static void fft_radix2(float *re, float *im, const uint32_t *bit_reverse, int n)
{
    for (int i = 0; i < n; ++i) {
        uint32_t j = bit_reverse[i];
        if (i < (int)j) {
            float tr = re[i], ti = im[i];
            re[i] = re[j];
            im[i] = im[j];
            re[j] = tr;
            im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * (float)M_PI / (float)len;
        const float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; ++k) {
                const int a = i + k, b = i + k + len / 2;
                const float tr = cr * re[b] - ci * im[b];
                const float ti = cr * im[b] + ci * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                const float ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}

/* ------------------------------------------------------------------------ */
/* public API                                                                */
/* ------------------------------------------------------------------------ */
void kws_frontend_init(kws_frontend_t *fe, kws_quant_t quant)
{
    memset(fe, 0, sizeof(*fe));
    fe->quant = quant;

    /* bit-reversal permutation table */
    int bits = 0;
    while ((1 << bits) < KWS_FFT_SIZE) ++bits;
    for (int i = 0; i < KWS_FFT_SIZE; ++i) {
        uint32_t r = 0;
        for (int b = 0; b < bits; ++b) {
            if (i & (1 << b)) r |= 1u << (bits - 1 - b);
        }
        fe->bit_reverse[i] = r;
    }
}

void kws_frontend_frame(kws_frontend_t *fe, const int16_t *frame, int8_t out[KWS_NUM_MEL_BINS])
{
    /* 1. windowed frame -> float, zero-padded to the FFT size */
    for (int i = 0; i < KWS_FRAME_LENGTH; ++i) {
        fe->fft_re[i] = ((float)frame[i] / 32768.0f) * KWS_HANN_WINDOW[i];
    }
    for (int i = KWS_FRAME_LENGTH; i < KWS_FFT_SIZE; ++i) {
        fe->fft_re[i] = 0.0f;
    }
    memset(fe->fft_im, 0, sizeof(fe->fft_im));

    /* 2. real spectrum via a complex FFT of the real signal (imag == 0) */
    fft_radix2(fe->fft_re, fe->fft_im, fe->bit_reverse, KWS_FFT_SIZE);

    /* 3. power spectrum (only the first n/2+1 bins are used by the filterbank) */
    float *power = fe->fft_power;
    for (int i = 0; i <= KWS_FFT_SIZE / 2; ++i) {
        power[i] = fe->fft_re[i] * fe->fft_re[i] + fe->fft_im[i] * fe->fft_im[i];
    }

    /* 4. mel filterbank + log + affine int8 quantisation */
    for (int m = 0; m < KWS_NUM_MEL_BINS; ++m) {
        const float *row = &KWS_MEL_MATRIX[m][0];
        float acc = 0.0f;
        for (int k = 0; k <= KWS_FFT_SIZE / 2; ++k) {
            acc += row[k] * power[k];
        }
        const float fl = logf(acc > fe->quant.log_floor ? acc : fe->quant.log_floor);
        float q = fl / fe->quant.scale + (float)fe->quant.zero_point;
        /* round-half-away-from-zero, then clamp to the int8 range */
        q = (q >= 0.0f) ? floorf(q + 0.5f) : ceilf(q - 0.5f);
        if (q > 127.0f) q = 127.0f;
        if (q < -128.0f) q = -128.0f;
        out[m] = (int8_t)q;
    }
}

bool kws_frontend_push(kws_frontend_t *fe, const int16_t *samples, size_t n)
{
    /* Samples are appended to `sample_ring`, which always holds the samples of the
     * frame being assembled, contiguously:
     *
     *   first frame : [0 .. FRAME_LENGTH)                    <- 480 new samples
     *   afterwards  : [0 .. KEEP) tail of the previous frame <- carried over
     *                 [KEEP .. FRAME_LENGTH) new samples      <- HOP new samples
     *
     * KEEP = FRAME_LENGTH - HOP = 160, so the frames start at 0, HOP, 2*HOP, ...
     * exactly like `frame_signal()` on the Python side.  An earlier version
     * prefilled the ring with KEEP zeros, which put every device frame 160 samples
     * (10 ms) behind the training frames - a shift that is invisible in the feature
     * values and quietly misaligns the keyword.
     */
    const size_t keep = KWS_FRAME_LENGTH - KWS_FRAME_HOP;
    bool produced = false;

    for (size_t i = 0; i < n; ++i) {
        const bool first_frame = (fe->frames_emitted == 0);
        const size_t base = first_frame ? 0 : keep;
        const size_t needed = first_frame ? KWS_FRAME_LENGTH : KWS_FRAME_HOP;

        fe->sample_ring[base + fe->samples_since_frame] = samples[i];
        fe->samples_since_frame++;

        if (fe->samples_since_frame == needed) {
            /* shift the rolling spectrogram, then append the frame straight out of
             * the ring - it is already contiguous, so no staging copy is needed */
            memmove(&fe->frames[0][0], &fe->frames[1][0],
                    (KWS_CONTEXT_FRAMES - 1) * KWS_NUM_MEL_BINS);
            kws_frontend_frame(fe, fe->sample_ring, fe->frames[KWS_CONTEXT_FRAMES - 1]);

            /* carry the tail of this frame into the head of the next one */
            memmove(fe->sample_ring, fe->sample_ring + KWS_FRAME_HOP, keep * sizeof(int16_t));

            fe->frames_emitted++;
            if (fe->frames_filled < KWS_CONTEXT_FRAMES) fe->frames_filled++;
            fe->samples_since_frame = 0;
            produced = true;
        }
    }
    return produced;
}

void kws_frontend_window(const kws_frontend_t *fe, int8_t *out)
{
    /* The padded rows must be the quantised value of ln(log_floor), not of
     * log_floor itself.  The Python side pads with `np.log(params.log_floor)`
     * (= -13.8), and quantising the raw 1e-6 instead gives zero_point, i.e. 70 in
     * the tables this repository generates - a padding row that is 197 int8 steps
     * away from the value the model was trained on, fed to the first frames of
     * every window. */
    const float floor_log = logf(KWS_LOG_FLOOR);
    float fq = floor_log / fe->quant.scale + (float)fe->quant.zero_point;
    fq = (fq >= 0.0f) ? floorf(fq + 0.5f) : ceilf(fq - 0.5f);
    if (fq > 127.0f) fq = 127.0f;
    if (fq < -128.0f) fq = -128.0f;
    const int8_t floor_q = (int8_t)fq;
    const uint32_t filled = fe->frames_filled < KWS_CONTEXT_FRAMES ? fe->frames_filled : KWS_CONTEXT_FRAMES;
    const uint32_t pad = KWS_CONTEXT_FRAMES - filled;

    for (uint32_t i = 0; i < pad; ++i) {
        memset(out + i * KWS_NUM_MEL_BINS, floor_q, KWS_NUM_MEL_BINS);
    }
    memcpy(out + pad * KWS_NUM_MEL_BINS, &fe->frames[KWS_CONTEXT_FRAMES - filled][0],
           filled * KWS_NUM_MEL_BINS);
}

uint32_t kws_frontend_frames(const kws_frontend_t *fe)
{
    return (fe->frames_emitted < KWS_CONTEXT_FRAMES) ? fe->frames_emitted : KWS_CONTEXT_FRAMES;
}

void kws_frontend_reset(kws_frontend_t *fe)
{
    memset(fe->sample_ring, 0, sizeof(fe->sample_ring));
    memset(fe->frames, 0, sizeof(fe->frames));
    fe->samples_since_frame = 0;
    fe->frames_filled = 0;
    fe->frames_emitted = 0;
}

void kws_frontend_waveform_to_features(const int16_t *pcm, size_t n_samples, int8_t *out_features)
{
    kws_frontend_t fe;
    /* the tables header also carries the quantisation constants chosen at export */
    kws_quant_t q = {KWS_QUANT_SCALE, KWS_QUANT_ZERO_POINT, KWS_LOG_FLOOR};
    kws_frontend_init(&fe, q);
    /* Only the first KWS_WINDOW_SAMPLES are consumed: that is exactly the audio
     * needed for CONTEXT_FRAMES frames, and `ml.common.features.features_from_waveform`
     * likewise takes the *first* 49 frames of a clip (`_fix_frames` truncates with
     * `mel[:n]`).  Pushing the whole clip would instead yield the rolling *last* 49
     * frames, which is the streaming contract and a different window. */
    const size_t used = (n_samples < (size_t)KWS_WINDOW_SAMPLES) ? n_samples : (size_t)KWS_WINDOW_SAMPLES;
    kws_frontend_push(&fe, pcm, used);
    kws_frontend_window(&fe, out_features);
}
