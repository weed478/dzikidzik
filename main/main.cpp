#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

#include "detector.h"

static const char TAG[] = "main";

extern "C" void app_main(void)
{
    detector_init();

    for (;;)
    {
        detector_detect(nullptr, 0, 0);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
