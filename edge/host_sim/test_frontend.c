/*
 * Host-side harness for the firmware's acoustic front-end.
 *
 *   1. `--features`  : int16 PCM in, int8 log-mel window out (parity check
 *                      against ml/common/features.py via tests/test_frontend_parity.py)
 *   2. `--bench`     : measure the per-frame and per-window cost of the
 *                      front-end itself (FFT + melbank + log + quantise), which
 *                      is the part of the device CPU budget that is *not* the
 *                      neural network.
 *   3. `--g711`      : encode a PCM stream to mu-law and report the compression
 *
 * Usage:
 *     ./kws_host_sim --features in.pcm out.bin [n_samples]
 *     ./kws_host_sim --bench in.pcm [seconds]
 *     ./kws_host_sim --g711  in.pcm out.ulaw
 */
/* The timing code below uses clock_gettime(CLOCK_MONOTONIC), which POSIX exposes
 * only when _POSIX_C_SOURCE is declared; with plain -std=c99 the compiler stops at
 * "implicit declaration of function 'clock_gettime'". This is host-only code, so
 * asking for the POSIX namespace is the right fix - the firmware does its timing
 * with esp_timer instead. */
#define _POSIX_C_SOURCE 199309L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../common/kws_frontend.h"
#include "../common/kws_g711.h"

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int16_t *read_pcm(const char *path, size_t *n_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        fclose(f);
        return NULL;
    }
    size_t n = (size_t)size / sizeof(int16_t);
    int16_t *buf = (int16_t *)malloc(n * sizeof(int16_t));
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, sizeof(int16_t), n, f) != n) {
        fprintf(stderr, "short read on %s\n", path);
    }
    fclose(f);
    *n_out = n;
    return buf;
}

static int cmd_features(const char *in_path, const char *out_path, size_t limit)
{
    size_t n = 0;
    int16_t *pcm = read_pcm(in_path, &n);
    if (!pcm) return 2;
    if (limit && limit < n) n = limit;

    int8_t features[KWS_CONTEXT_FRAMES * KWS_NUM_MEL_BINS];
    kws_frontend_waveform_to_features(pcm, n, features);

    FILE *f = fopen(out_path, "wb");
    if (!f) {
        free(pcm);
        return 3;
    }
    fwrite(features, 1, sizeof(features), f);
    fclose(f);
    fprintf(stderr, "features: %zu samples -> %zu bytes\n", n, sizeof(features));
    free(pcm);
    return 0;
}

static int cmd_bench(const char *in_path, double seconds)
{
    size_t n = 0;
    int16_t *pcm = read_pcm(in_path, &n);
    if (!pcm) return 2;

    /* loop the input until we have `seconds` of audio */
    const size_t target = (size_t)(seconds * KWS_SAMPLE_RATE);
    if (target > n) {
        int16_t *big = (int16_t *)malloc(target * sizeof(int16_t));
        for (size_t i = 0; i < target; ++i) big[i] = pcm[i % n];
        free(pcm);
        pcm = big;
        n = target;
    }

    kws_frontend_t fe;
    kws_quant_t q = {KWS_QUANT_SCALE, KWS_QUANT_ZERO_POINT, KWS_LOG_FLOOR};
    kws_frontend_init(&fe, q);

    const double t0 = now_s();
    uint32_t frames = 0;
    for (size_t i = 0; i < n; i += 160) { /* feed in 10 ms chunks, like I2S DMA */
        size_t chunk = (n - i) < 160 ? (n - i) : 160;
        if (kws_frontend_push(&fe, pcm + i, chunk)) frames++;
    }
    const double elapsed = now_s() - t0;
    const double audio_s = (double)n / KWS_SAMPLE_RATE;

    /* NOTE: these are HOST numbers.  They bound the algorithm's cost and let us
     * regression-test it in CI, but the figure that must be quoted for the SIH
     * "under 10% CPU while idle listening" requirement has to be measured on the
     * physical board: run the same benchmark with edge/esp32 and see
     * docs/BENCHMARKS.md for how the number is derived.  No scaling factor
     * between an x86 core and the ESP32-S3 is printed here on purpose - that
     * extrapolation would be a guess dressed up as a measurement. */
    printf("front-end benchmark (HOST CPU - not the device figure)\n");
    printf("  audio processed : %.2f s\n", audio_s);
    printf("  frames emitted  : %u\n", frames);
    printf("  wall time       : %.3f ms\n", elapsed * 1000.0);
    printf("  per frame       : %.1f us  (FFT + 40-band melbank + log + quantise)\n",
           frames ? elapsed * 1e6 / frames : 0.0);
    printf("  real-time factor: %.5f  (fraction of one host core while listening)\n", elapsed / audio_s);
    free(pcm);
    return 0;
}

static int cmd_g711(const char *in_path, const char *out_path)
{
    size_t n = 0;
    int16_t *pcm = read_pcm(in_path, &n);
    if (!pcm) return 2;

    size_t n8 = 0;
    int16_t *down = (int16_t *)malloc((n / 2 + 16) * sizeof(int16_t));
    kws_downsample_16k_to_8k(pcm, n, down, &n8);

    uint8_t *ulaw = (uint8_t *)malloc(n8);
    kws_ulaw_encode(down, n8, ulaw);

    FILE *f = fopen(out_path, "wb");
    if (!f) {
        free(pcm);
        free(down);
        free(ulaw);
        return 3;
    }
    fwrite(ulaw, 1, n8, f);
    fclose(f);

    printf("g711 ulaw uplink\n");
    printf("  16 kHz PCM in : %zu samples, %zu bytes (%.2f s)\n", n, n * 2, (double)n / KWS_SAMPLE_RATE);
    printf("  8 kHz ulaw out: %zu samples, %zu bytes\n", n8, n8);
    printf("  bitrate       : 64 kbit/s vs 256 kbit/s PCM16\n");
    printf("  payload cut   : %.1f %% smaller\n", 100.0 * (1.0 - (double)n8 / (double)(n * 2)));
    free(pcm);
    free(down);
    free(ulaw);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "usage:\n"
                "  %s --features <in.pcm> <out.bin> [n_samples]\n"
                "  %s --bench    <in.pcm> [seconds]\n"
                "  %s --g711     <in.pcm> <out.ulaw>\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "--features") == 0) {
        size_t limit = (argc >= 5) ? (size_t)strtoul(argv[4], NULL, 10) : 0;
        return cmd_features(argv[2], argv[3], limit);
    }
    if (strcmp(argv[1], "--bench") == 0) {
        double seconds = (argc >= 4) ? strtod(argv[3], NULL) : 60.0;
        return cmd_bench(argv[2], seconds);
    }
    if (strcmp(argv[1], "--g711") == 0) {
        return cmd_g711(argv[2], argv[3]);
    }
    fprintf(stderr, "unknown command %s\n", argv[1]);
    return 1;
}
