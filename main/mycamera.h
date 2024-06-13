#ifndef __DZIKIDZIK_CAMERA_H__
#define __DZIKIDZIK_CAMERA_H__

#include "esp_err.h"

void mycamera_init(void);

esp_err_t mycamera_grab(unsigned char *image, int height, int width);

#endif
