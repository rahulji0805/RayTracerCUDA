// main.cu
// render.exe - renders a single frame of the 168-sphere stress-test scene
// on the GPU using the CUDA kernel in kernel.cu, and writes it out as a PNG.
//
// Usage: render.exe [width] [height] [output.png] [samples]
//   defaults: 1280 720 outputs/render_hd.png 4   (4 -> 4x4 = 16 rays/pixel AA)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>

#include "vec3.cuh"
#include "camera.cuh"
#include "kernel.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char** argv) {
    int width    = (argc > 1) ? atoi(argv[1]) : 1280;
    int height   = (argc > 2) ? atoi(argv[2]) : 720;
    std::string out_path = (argc > 3) ? argv[3] : "outputs/render_hd.png";
    int samples  = (argc > 4) ? atoi(argv[4]) : 4; // 4x4 = 16 samples/pixel, matches Python SAMPLES

    cudaDeviceProp prop;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "No CUDA device found.\n");
        return 1;
    }
    cudaGetDeviceProperties(&prop, 0);

    printf("===================================\n");
    printf(" CUDA C++ Ray Tracer\n");
    printf("===================================\n");
    printf(" GPU            : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf(" Resolution     : %d x %d (%d pixels)\n", width, height, width * height);
    printf(" Anti-aliasing  : %dx%d supersampling (%d rays/pixel)\n", samples, samples, samples * samples);
    printf(" Spheres        : 168\n");

    // Same look-at camera as the original Python static render.
    Camera cam = build_camera(width, height, 45.0f,
                               Vec3(0.0f, 1.8f, 4.5f),
                               Vec3(0.0f, 0.4f, -1.5f));

    size_t image_bytes = (size_t)width * height * 3;
    unsigned char* d_image = nullptr;
    check_cuda(cudaMalloc(&d_image, image_bytes), "cudaMalloc image");

    unsigned char* h_image = (unsigned char*)malloc(image_bytes);

    // Warm-up launch (absorbs JIT/driver init cost so the timed run is clean).
    render_gpu(d_image, width, height, cam, samples);

    auto t0 = std::chrono::high_resolution_clock::now();
    render_gpu(d_image, width, height, cam, samples);
    auto t1 = std::chrono::high_resolution_clock::now();

    check_cuda(cudaMemcpy(h_image, d_image, image_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double pixels_per_sec = (width * (double)height) / (ms / 1000.0);

    printf(" Render time    : %.2f ms\n", ms);
    printf(" Throughput     : %.0f pixels/sec\n", pixels_per_sec);

    stbi_write_png(out_path.c_str(), width, height, 3, h_image, width * 3);
    printf(" Output         : %s\n", out_path.c_str());
    printf("===================================\n");

    free(h_image);
    cudaFree(d_image);
    return 0;
}
