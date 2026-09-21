/*
 * Uplink implementation: Wi-Fi station + HTTP client for the ASR server.
 *
 * Chunking note: the device sends 160 mu-law bytes per 20 ms frame (exactly one
 * frame of 8 kHz G.711), so the server receives a continuous stream at the
 * codec's natural rate.  Nothing is buffered beyond one DMA block, which keeps
 * RAM flat and the wake-to-transcript path short.
 */
#include "kws_uplink.h"

#include <string.h>

#include "driver/i2s_std.h"
#include "esp_crt_bundle.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

#include "kws_g711.h"

static const char *TAG = "kws_uplink";

/* Wi-Fi credentials are injected at build time so they never land in Git:
 *   idf.py -DKWS_WIFI_SSID=... -DKWS_WIFI_PASS=... build
 * or via `menuconfig` (KWS_WIFI_SSID / KWS_WIFI_PASS). */
#ifndef CONFIG_KWS_WIFI_SSID
#define CONFIG_KWS_WIFI_SSID "set-me"
#endif
#ifndef CONFIG_KWS_WIFI_PASS
#define CONFIG_KWS_WIFI_PASS "set-me"
#endif

extern i2s_chan_handle_t kws_mic_handle(void);  /* provided by kws_app.c */

#define WIFI_CONNECTED_BIT BIT0

static EventGroupHandle_t s_wifi_events;
static kws_uplink_config_t s_cfg = {.wake_path = "/v1/wake", .asr_path = "/v1/asr/ulaw"};

/* ------------------------------------------------------------------------ */
/* Wi-Fi                                                                    */
/* ------------------------------------------------------------------------ */
static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        s_cfg.wifi_ready = false;
        ESP_LOGW(TAG, "Wi-Fi dropped, reconnecting");
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        s_cfg.wifi_ready = true;
        xEventGroupSetBits(s_wifi_events, WIFI_CONNECTED_BIT);
        ESP_LOGI(TAG, "Wi-Fi connected");
    }
}

void kws_uplink_init(const char *base_url)
{
    strncpy(s_cfg.base_url, base_url, sizeof(s_cfg.base_url) - 1);
    s_wifi_events = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    wifi_config_t sta = {};
    strncpy((char *)sta.sta.ssid, CONFIG_KWS_WIFI_SSID, sizeof(sta.sta.ssid) - 1);
    strncpy((char *)sta.sta.password, CONFIG_KWS_WIFI_PASS, sizeof(sta.sta.password) - 1);
    sta.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "station started, server=%s", s_cfg.base_url);
}

/* ------------------------------------------------------------------------ */
/* HTTP helpers                                                              */
/* ------------------------------------------------------------------------ */
static esp_http_client_handle_t make_client(const char *path, esp_http_client_method_t method)
{
    char url[192];
    snprintf(url, sizeof(url), "%s%s", s_cfg.base_url, path);
    esp_http_client_config_t config = {};
    config.url = url;
    config.method = method;
    config.timeout_ms = 4000;
    config.crt_bundle_attach = esp_crt_bundle_attach;  /* allows https:// too */
    config.keep_alive_enable = true;
    return esp_http_client_init(&config);
}

bool kws_uplink_notify_wake(const char *base_url, const char *device_id, const char *keyword, float score,
                            float threshold, int latency_ms, int confirm_windows)
{
    (void)base_url;
    if (!s_cfg.wifi_ready) {
        ESP_LOGW(TAG, "wake not reported: Wi-Fi down");
        return false;
    }
    char body[320];
    snprintf(body, sizeof(body),
             "{\"device_id\":\"%s\",\"keyword\":\"%s\",\"score\":%.4f,\"threshold\":%.4f,"
             "\"latency_ms\":%d,\"policy\":\"confirm_%d\",\"firmware\":\"esp32s3/1.0\"}",
             device_id, keyword, score, threshold, latency_ms, confirm_windows);

    esp_http_client_handle_t client = make_client(s_cfg.wake_path, HTTP_METHOD_POST);
    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, body, (int)strlen(body));
    const esp_err_t err = esp_http_client_perform(client);
    const int status = esp_http_client_get_status_code(client);
    esp_http_client_cleanup(client);

    if (err != ESP_OK || status / 100 != 2) {
        ESP_LOGW(TAG, "wake POST failed (err %d, http %d)", err, status);
        return false;
    }
    return true;
}

int kws_uplink_stream_ulaw(const char *base_url, const char *device_id, int duration_ms)
{
    (void)base_url;
    if (!s_cfg.wifi_ready) {
        ESP_LOGW(TAG, "cannot stream: Wi-Fi down");
        return -1;
    }

    char path[128];
    snprintf(path, sizeof(path), "%s?device_id=%s&sample_rate=8000", s_cfg.asr_path, device_id);
    esp_http_client_handle_t client = make_client(path, HTTP_METHOD_POST);
    esp_http_client_set_header(client, "Content-Type", "application/octet-stream");

    /* chunked transfer: the device does not know the body length in advance */
    esp_http_client_open(client, 0);

    i2s_chan_handle_t mic = kws_mic_handle();
    int16_t pcm16[320];        /* 20 ms at 16 kHz */
    int16_t pcm8[320];         /* 20 ms at  8 kHz after decimation */
    uint8_t ulaw[320];
    int total = 0;
    const int frames = duration_ms / 20;

    for (int f = 0; f < frames; ++f) {
        size_t got = 0;
        if (i2s_channel_read(mic, pcm16, sizeof(pcm16), &got, pdMS_TO_TICKS(100)) != ESP_OK) {
            break;
        }
        size_t n_in = got / sizeof(int16_t);
        size_t n_out = 0;
        kws_downsample_16k_to_8k(pcm16, n_in, pcm8, &n_out);
        kws_ulaw_encode(pcm8, n_out, ulaw);

        if (esp_http_client_write(client, (const char *)ulaw, (int)n_out) < 0) {
            ESP_LOGW(TAG, "upload aborted at frame %d", f);
            break;
        }
        total += (int)n_out;
    }

    esp_http_client_fetch_headers(client);
    int status = esp_http_client_get_status_code(client);

    /* the server replies with the transcript; read it for logging */
    char response[512] = {0};
    int read = esp_http_client_read(client, response, sizeof(response) - 1);
    if (read > 0) {
        response[read] = 0;
        ESP_LOGI(TAG, "server: %s", response);
    }
    esp_http_client_close(client);
    esp_http_client_cleanup(client);

    if (status / 100 != 2) {
        ESP_LOGW(TAG, "uplink failed with HTTP %d", status);
        return -1;
    }
    ESP_LOGI(TAG, "uplink complete: %d mu-law bytes (%d PCM16 bytes avoided)", total, total * 4);
    return total;
}
