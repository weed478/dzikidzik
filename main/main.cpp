#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

#include "mycamera.h"
#include "detector.h"

static const char TAG[] = "main";

#define WIDTH 224
#define HEIGHT 224

static unsigned char rgb888_buf[HEIGHT][WIDTH][3];

extern "C" void app_main(void)
{
    mycamera_init();
    detector_init();

    for (;;)
    {
        ESP_LOGI(TAG, "Taking picture...");
        ESP_ERROR_CHECK(mycamera_grab((unsigned char*) rgb888_buf, HEIGHT, WIDTH));

        ESP_LOGI(TAG, "Detecting dogs on the picture...");
        detector_detect((const unsigned char*) rgb888_buf, HEIGHT, WIDTH);

        vTaskDelay(1);
    }
}
