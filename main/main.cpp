#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_camera.h"
#include "esp_log.h"

#include "detector.h"

// XIAO ESP32-S3
#define CAM_PIN_PWDN    -1 //power down is not used
#define CAM_PIN_RESET   -1 //software reset will be performed
#define CAM_PIN_XCLK    10
#define CAM_PIN_SIOD    40
#define CAM_PIN_SIOC    39

#define CAM_PIN_D7      48
#define CAM_PIN_D6      11
#define CAM_PIN_D5      12
#define CAM_PIN_D4      14
#define CAM_PIN_D3      16
#define CAM_PIN_D2      18
#define CAM_PIN_D1      17
#define CAM_PIN_D0      15
#define CAM_PIN_VSYNC   38
#define CAM_PIN_HREF    47
#define CAM_PIN_PCLK    13

static const char TAG[] = "main";

static camera_config_t camera_config = {
    .pin_pwdn  = CAM_PIN_PWDN,
    .pin_reset = CAM_PIN_RESET,
    .pin_xclk = CAM_PIN_XCLK,
    .pin_sccb_sda = CAM_PIN_SIOD,
    .pin_sccb_scl = CAM_PIN_SIOC,

    .pin_d7 = CAM_PIN_D7,
    .pin_d6 = CAM_PIN_D6,
    .pin_d5 = CAM_PIN_D5,
    .pin_d4 = CAM_PIN_D4,
    .pin_d3 = CAM_PIN_D3,
    .pin_d2 = CAM_PIN_D2,
    .pin_d1 = CAM_PIN_D1,
    .pin_d0 = CAM_PIN_D0,
    .pin_vsync = CAM_PIN_VSYNC,
    .pin_href = CAM_PIN_HREF,
    .pin_pclk = CAM_PIN_PCLK,

    .xclk_freq_hz = 20000000,//EXPERIMENTAL: Set to 16MHz on ESP32-S2 or ESP32-S3 to enable EDMA mode
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_RGB565,//YUV422,GRAYSCALE,RGB565,JPEG
    .frame_size = FRAMESIZE_SXGA,//QQVGA-UXGA, For ESP32, do not use sizes above QVGA when not JPEG. The performance of the ESP32-S series has improved a lot, but JPEG mode always gives better frame rates.

    .jpeg_quality = 12, //0-63, for OV series camera sensors, lower number means higher quality
    .fb_count = 1, //When jpeg mode is used, if fb_count more than one, the driver will work in continuous mode.
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY//CAMERA_GRAB_LATEST. Sets when buffers should be filled
};

static esp_err_t init_camera(void)
{
    //initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);

    if (err != ESP_OK) 
    {
        ESP_LOGE(TAG, "Failed initialization of a camera!");
        return err;
    }

    return ESP_OK;
}

extern "C" void app_main(void)
{
    if (ESP_OK != init_camera()) 
    {
        return;
    }

    detector_init();

    for (;;)
    {
        ESP_LOGI(TAG, "");
        ESP_LOGI(TAG, "Taking picture...");

        camera_fb_t *pic = esp_camera_fb_get();
        if (!pic) 
        {
            ESP_LOGE(TAG, "Failed capturing a picture!");
            continue;
        }

        const unsigned char * rgb888_buf = (const unsigned char*) malloc(pic->width * pic->height * 3 * sizeof(char));

        const unsigned char * pixel888 = &rgb888_buf[0];

        for (int idx = 0; idx < pic->width * pic->height; idx += 2) 
        {
            char16_t pixel565 = (pic->buf[idx] << 8) | pic->buf[idx + 1];

            pixel888 = (const unsigned char*) (((pixel565 >> 11) * 527 + 23) >> 6);  // R
            pixel888 += 1;

            pixel888 = (const unsigned char*) ((((pixel565 >> 5) & 63) * 259 + 33) >> 6);  // G
            pixel888 += 1;

            pixel888 = (const unsigned char*) (((pixel565 & 31) * 527 + 23) >> 6);  // B
            pixel888 += 1;
        }

        ESP_LOGI(TAG, "Picture taken! It's size was: %zu bytes", pic->len);

        ESP_LOGI(TAG, "Detecting dogs on the picture...");
        detector_detect(rgb888_buf, pic->width, pic->height);

        free((unsigned char*) rgb888_buf);
        rgb888_buf = NULL;
        
        esp_camera_fb_return(pic);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
