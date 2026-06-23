// benchmark.cu
// benchmark.exe - times CPU vs GPU (full alloc+transfer+kernel) vs GPU
// (kernel-only, no host<->device transfer) across a range of resolutions,
// on the same 168-sphere stress-test scene. Mirrors the original Python
// benchmark.py methodology so the numbers are directly comparable.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <fstream>

#include "vec3.cuh"
#include "camera.cuh"
#include "kernel.cuh"
#include "cpu_render.h"

#define N_REPEATS 5

struct Resolution { int w, h; };

static double time_cpu(int width, int height, Camera cam, int samples, unsigned char* h_image) {
    render_cpu(h_image, width, height, cam, samples); // warm-up
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_REPEATS; ++i) render_cpu(h_image, width, height, cam, samples);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / N_REPEATS;
}

static double time_gpu_full(int width, int height, Camera cam, int samples) {
    size_t bytes = (size_t)width * height * 3;
    unsigned char* h_image = (unsigned char*)malloc(bytes);

    auto run_once = [&]() {
        unsigned char* d_image;
        cudaMalloc(&d_image, bytes);
        render_gpu(d_image, width, height, cam, samples);
        cudaMemcpy(h_image, d_image, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_image);
    };

    run_once(); // warm-up
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_REPEATS; ++i) run_once();
    auto t1 = std::chrono::high_resolution_clock::now();

    free(h_image);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / N_REPEATS;
}

static double time_gpu_kernel_only(int width, int height, Camera cam, int samples) {
    size_t bytes = (size_t)width * height * 3;
    unsigned char* d_image;
    cudaMalloc(&d_image, bytes); // allocated once, outside the timed region

    render_gpu(d_image, width, height, cam, samples); // warm-up
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_REPEATS; ++i) render_gpu(d_image, width, height, cam, samples);
    auto t1 = std::chrono::high_resolution_clock::now();

    cudaFree(d_image);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / N_REPEATS;
}

int main() {
    std::vector<Resolution> resolutions = {
        {320, 180}, {640, 360}, {800, 450}, {1280, 720}, {1920, 1080}, {3840, 2160}
    };
    int samples = 4; // 4x4 = 16 rays/pixel, matches the render

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cudaDeviceProp prop;
    if (device_count > 0) cudaGetDeviceProperties(&prop, 0);

    printf("%-14s%-30s%-12s%-15s\n", "Resolution", "Backend", "Time (ms)", "Pixels/sec");
    printf("-----------------------------------------------------------------\n");

    std::ofstream csv("outputs/benchmark_cpp.csv");
    csv << "resolution,backend,time_ms,pixels_per_sec\n";

    for (auto& res : resolutions) {
        Camera cam = build_camera(res.w, res.h, 45.0f,
                                   Vec3(0.0f, 1.8f, 4.5f), Vec3(0.0f, 0.4f, -1.5f));
        double n_pixels = (double)res.w * res.h;
        unsigned char* h_image = (unsigned char*)malloc((size_t)res.w * res.h * 3);

        char res_str[32];
        snprintf(res_str, sizeof(res_str), "%dx%d", res.w, res.h);

        double cpu_ms = time_cpu(res.w, res.h, cam, samples, h_image);
        printf("%-14s%-30s%-12.2f%-15.0f\n", res_str, "CPU (OpenMP)", cpu_ms, n_pixels / (cpu_ms / 1000.0));
        csv << res_str << ",CPU," << cpu_ms << "," << (n_pixels / (cpu_ms / 1000.0)) << "\n";

        if (device_count > 0) {
            double gpu_full_ms = time_gpu_full(res.w, res.h, cam, samples);
            double speedup_full = cpu_ms / gpu_full_ms;
            printf("%-14s%-30s%-12.2f%-15.0f(speedup: %.1fx)\n", "", "GPU full (alloc+xfer+kernel)",
                   gpu_full_ms, n_pixels / (gpu_full_ms / 1000.0), speedup_full);
            csv << res_str << ",GPU-full," << gpu_full_ms << "," << (n_pixels / (gpu_full_ms / 1000.0)) << "\n";

            double gpu_kernel_ms = time_gpu_kernel_only(res.w, res.h, cam, samples);
            double speedup_kernel = cpu_ms / gpu_kernel_ms;
            printf("%-14s%-30s%-12.2f%-15.0f(speedup: %.1fx)\n", "", "GPU kernel-only (no xfer)",
                   gpu_kernel_ms, n_pixels / (gpu_kernel_ms / 1000.0), speedup_kernel);
            csv << res_str << ",GPU-kernel-only," << gpu_kernel_ms << "," << (n_pixels / (gpu_kernel_ms / 1000.0)) << "\n";
        }

        free(h_image);
    }

    if (device_count == 0) {
        printf("\nNote: No CUDA GPU detected - only CPU numbers shown.\n");
    }

    csv.close();
    printf("\nSaved outputs/benchmark_cpp.csv\n");
    return 0;
}
