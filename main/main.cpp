#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

static const char TAG[] = "main";

static constexpr int kTensorArenaSize = 6000000;
EXT_RAM_BSS_ATTR static uint8_t tensor_arena[kTensorArenaSize];

extern "C" void app_main(void)
{
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    const tflite::Model *model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG, "Model provided is schema version %ld not equal to supported "
                      "version %d.",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Pull in only the operation implementations we need.
    tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddQuantize();
    resolver.AddConv2D();
    resolver.AddRelu();
    resolver.AddAdd();
    resolver.AddDepthwiseConv2D();
    resolver.AddMean();
    resolver.AddReshape();
    resolver.AddSoftmax();

    // Build an interpreter to run the model with.
    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return;
    }

    ESP_LOGI(TAG, "Used arena: %d bytes", interpreter.arena_used_bytes());

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    for (;;)
    {
        int64_t start_us = esp_timer_get_time();
        TfLiteStatus invoke_status = interpreter.Invoke();
        int64_t latency = esp_timer_get_time() - start_us;
        if (invoke_status != kTfLiteOk)
        {
            ESP_LOGE(TAG, "Invoke failed");
            return;
        }
        int64_t fps = 1000000 / latency;
        ESP_LOGI(TAG, "Latency: %lld ms, FPS: %lld", latency / 1000, fps);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
