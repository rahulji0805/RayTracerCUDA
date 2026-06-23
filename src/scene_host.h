// Host-accessible copy of the exact same scene data, for the CPU
// reference renderer used in the CPU-vs-GPU benchmark.
#pragma once

#define NUM_SPHERES_HOST 168
extern const float h_spheres[NUM_SPHERES_HOST][8];
