// kernel.cu
// 1 CUDA thread = 1 pixel. Same rendering pipeline as the original
// numba.cuda version: ray-sphere intersection against every sphere
// (168 of them - this is intentionally O(n) per ray, it's what
// saturates the SMs), Lambert diffuse shading, hard shadow rays,
// metal mirror reflection (iterative bounce loop, MAX_DEPTH bounces),
// a procedural checkerboard floor, and NxN supersampled antialiasing.
#include <cuda_runtime.h>
#include <stdio.h>
#include "vec3.cuh"
#include "camera.cuh"
#include "scene_data.cuh"

#define MAX_DEPTH 5
#define EPS 1e-4f

// Pre-normalized directional "sun" light - same as Python LIGHT_DIR = normalize([0.5, 0.85, 0.4]).
__device__ __constant__ float d_light_dir[3] = {0.469841f, 0.798730f, 0.375873f};

struct Ray {
    Vec3 origin, dir;
};

struct HitRecord {
    float t;
    Vec3 p, normal;
    int sphere_idx;
};

// Test ray against every sphere, return closest hit (matches the
// Python `_closest_hit`: brute-force O(NUM_SPHERES) per ray).
__device__ bool closest_hit(const Ray& r, float t_min, float t_max, HitRecord& rec) {
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < NUM_SPHERES; ++i) {
        Vec3 center(d_spheres[i][0], d_spheres[i][1], d_spheres[i][2]);
        float radius = d_spheres[i][3];

        Vec3 oc = r.origin - center;
        float a = dot(r.dir, r.dir);
        float half_b = dot(oc, r.dir);
        float c = dot(oc, oc) - radius * radius;
        float disc = half_b * half_b - a * c;
        if (disc < 0.0f) continue;

        float sqrt_d = sqrtf(disc);
        float root = (-half_b - sqrt_d) / a;
        if (root < t_min || root > closest_so_far) {
            root = (-half_b + sqrt_d) / a;
            if (root < t_min || root > closest_so_far) continue;
        }

        hit_anything = true;
        closest_so_far = root;
        rec.t = root;
        rec.p = r.origin + r.dir * root;
        rec.normal = normalize(rec.p - center);
        rec.sphere_idx = i;
    }
    return hit_anything;
}

// Procedural checkerboard pattern for the ground sphere (material 2).
__device__ Vec3 checker_color(const Vec3& p, const Vec3& base_color) {
    int ix = (int)floorf(p.x);
    int iz = (int)floorf(p.z);
    bool dark = ((ix + iz) & 1) == 0;
    return dark ? base_color * 0.35f : base_color;
}

// Sky gradient with a soft sun glow around the light direction.
__device__ Vec3 sky_color(const Vec3& dir) {
    Vec3 light(d_light_dir[0], d_light_dir[1], d_light_dir[2]);
    float t = 0.5f * (dir.y + 1.0f);
    Vec3 base = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
    float sun = fmaxf(dot(dir, light), 0.0f);
    float glow = powf(sun, 256.0f) * 1.5f + powf(sun, 8.0f) * 0.15f;
    return base + Vec3(1.0f, 0.9f, 0.7f) * glow;
}

// Hard shadow test: is the point in shadow w.r.t. the directional light?
__device__ bool in_shadow(const Vec3& p, const Vec3& normal) {
    Vec3 light(d_light_dir[0], d_light_dir[1], d_light_dir[2]);
    Ray shadow_ray;
    shadow_ray.origin = p + normal * EPS;
    shadow_ray.dir = light;
    HitRecord rec;
    return closest_hit(shadow_ray, EPS, 1e8f, rec);
}

// Iterative path: diffuse/checker surfaces terminate with Lambert shading,
// metal surfaces reflect and continue the loop (up to MAX_DEPTH bounces).
// This mirrors the Python version's bounce loop rather than recursion,
// which keeps register pressure low and avoids CUDA stack-depth issues.
__device__ Vec3 ray_color(Ray r) {
    Vec3 attenuation(1.0f, 1.0f, 1.0f);
    Vec3 result(0.0f, 0.0f, 0.0f);

    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        HitRecord rec;
        if (!closest_hit(r, EPS, 1e8f, rec)) {
            result += attenuation * sky_color(normalize(r.dir));
            return result;
        }

        int mat = (int)(d_spheres[rec.sphere_idx][7] + 0.5f);
        Vec3 base_color(d_spheres[rec.sphere_idx][4],
                         d_spheres[rec.sphere_idx][5],
                         d_spheres[rec.sphere_idx][6]);

        if (mat == 1) {
            // Metal: reflect and continue the loop with tinted attenuation.
            Vec3 reflected = normalize(reflect(normalize(r.dir), rec.normal));
            attenuation = attenuation * base_color;
            r.origin = rec.p + rec.normal * EPS;
            r.dir = reflected;
            continue;
        }

        // Diffuse or checkerboard: Lambert shading + hard shadow, terminate.
        Vec3 surface_color = (mat == 2) ? checker_color(rec.p, base_color) : base_color;
        Vec3 light(d_light_dir[0], d_light_dir[1], d_light_dir[2]);
        float ndotl = fmaxf(dot(rec.normal, light), 0.0f);
        float shadow = in_shadow(rec.p, rec.normal) ? 0.0f : 1.0f;
        float ambient = 0.12f;
        Vec3 shaded = surface_color * (ambient + (1.0f - ambient) * ndotl * shadow);
        result += attenuation * shaded;
        return result;
    }
    return result; // ran out of bounces - treat as absorbed (black-ish)
}

__global__ void render_kernel(unsigned char* image, int width, int height,
                               Camera cam, int samples) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    Vec3 color_sum(0.0f, 0.0f, 0.0f);
    int total_samples = samples * samples;

    for (int sy = 0; sy < samples; ++sy) {
        for (int sx = 0; sx < samples; ++sx) {
            float jx = (sx + 0.5f) / samples;
            float jy = (sy + 0.5f) / samples;
            float s = (px + jx) / (float)width;
            float t = 1.0f - (py + jy) / (float)height; // flip Y: image row 0 = top

            Ray r;
            r.origin = cam.origin;
            r.dir = cam.lower_left_corner + cam.horizontal * s + cam.vertical * t - cam.origin;
            color_sum += ray_color(r);
        }
    }

    Vec3 col = color_sum * (1.0f / total_samples);
    col = clamp01(col);

    // simple gamma correction (gamma 2.0), matches typical ray tracer output
    col = Vec3(sqrtf(col.x), sqrtf(col.y), sqrtf(col.z));

    int idx = (py * width + px) * 3;
    image[idx + 0] = (unsigned char)(255.99f * col.x);
    image[idx + 1] = (unsigned char)(255.99f * col.y);
    image[idx + 2] = (unsigned char)(255.99f * col.z);
}

// Host-side launch wrapper. d_image must already be device memory of
// size width*height*3 bytes. Blocking: synchronizes before returning.
void render_gpu(unsigned char* d_image, int width, int height,
                 Camera cam, int samples) {
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);
    render_kernel<<<blocks, threads>>>(d_image, width, height, cam, samples);
    cudaDeviceSynchronize();
}
