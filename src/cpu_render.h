// cpu_render.h
#pragma once
#include "camera.cuh"

// CPU reference implementation - same math as the GPU kernel, parallelized
// across rows with OpenMP. Used purely as the CPU baseline for benchmarking;
// the GPU path (kernel.cu) is the actual capstone deliverable.
void render_cpu(unsigned char* image, int width, int height, Camera cam, int samples);
