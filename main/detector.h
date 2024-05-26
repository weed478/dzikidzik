#ifndef __DZIKIDZIK_DETECTOR_H__
#define __DZIKIDZIK_DETECTOR_H__

void detector_init();

int detector_detect(const unsigned char *image, int width, int height);

#endif
