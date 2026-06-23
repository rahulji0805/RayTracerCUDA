// kernel.cuh
#pragma once
#include <cuda_runtime.h>
#include "camera.cuh"

// Renders one frame into d_image (device buffer, width*height*3 bytes,
// already allocated by the caller). Blocking call (synchronizes internally).
void render_gpu(unsigned char* d_image, int width, int height,
                 Camera cam, int samples);
