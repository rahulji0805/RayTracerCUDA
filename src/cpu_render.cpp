// cpu_render.cpp
// Single-threaded-per-pixel-row CPU reference implementation. Exact same
// ray-sphere intersection / Lambert / shadow / metal-reflection / checker
// math as kernel.cu, just running on the CPU with OpenMP across rows
// instead of one CUDA thread per pixel. This is the CPU baseline used
// for the CPU-vs-GPU speedup numbers in the benchmark.
#include "cpu_render.h"
#include "vec3.cuh"
#include "scene_host.h"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_DEPTH 5
#define EPS 1e-4f

static const Vec3 LIGHT_DIR_CPU(0.469841f, 0.798730f, 0.375873f);

struct RayC { Vec3 origin, dir; };
struct HitC { float t; Vec3 p, normal; int idx; };

static bool closest_hit_cpu(const RayC& r, float t_min, float t_max, HitC& rec) {
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < NUM_SPHERES_HOST; ++i) {
        Vec3 center(h_spheres[i][0], h_spheres[i][1], h_spheres[i][2]);
        float radius = h_spheres[i][3];
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
        rec.idx = i;
    }
    return hit_anything;
}

static Vec3 checker_color_cpu(const Vec3& p, const Vec3& base_color) {
    int ix = (int)floorf(p.x);
    int iz = (int)floorf(p.z);
    bool dark = ((ix + iz) & 1) == 0;
    return dark ? base_color * 0.35f : base_color;
}

static Vec3 sky_color_cpu(const Vec3& dir) {
    float t = 0.5f * (dir.y + 1.0f);
    Vec3 base = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
    float sun = std::max(dot(dir, LIGHT_DIR_CPU), 0.0f);
    float glow = powf(sun, 256.0f) * 1.5f + powf(sun, 8.0f) * 0.15f;
    return base + Vec3(1.0f, 0.9f, 0.7f) * glow;
}

static bool in_shadow_cpu(const Vec3& p, const Vec3& normal) {
    RayC sr;
    sr.origin = p + normal * EPS;
    sr.dir = LIGHT_DIR_CPU;
    HitC rec;
    return closest_hit_cpu(sr, EPS, 1e8f, rec);
}

static Vec3 ray_color_cpu(RayC r) {
    Vec3 attenuation(1.0f, 1.0f, 1.0f);
    Vec3 result(0.0f, 0.0f, 0.0f);

    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        HitC rec;
        if (!closest_hit_cpu(r, EPS, 1e8f, rec)) {
            result += attenuation * sky_color_cpu(normalize(r.dir));
            return result;
        }

        int mat = (int)(h_spheres[rec.idx][7] + 0.5f);
        Vec3 base_color(h_spheres[rec.idx][4], h_spheres[rec.idx][5], h_spheres[rec.idx][6]);

        if (mat == 1) {
            Vec3 reflected = normalize(reflect(normalize(r.dir), rec.normal));
            attenuation = attenuation * base_color;
            r.origin = rec.p + rec.normal * EPS;
            r.dir = reflected;
            continue;
        }

        Vec3 surface_color = (mat == 2) ? checker_color_cpu(rec.p, base_color) : base_color;
        float ndotl = std::max(dot(rec.normal, LIGHT_DIR_CPU), 0.0f);
        float shadow = in_shadow_cpu(rec.p, rec.normal) ? 0.0f : 1.0f;
        float ambient = 0.12f;
        Vec3 shaded = surface_color * (ambient + (1.0f - ambient) * ndotl * shadow);
        result += attenuation * shaded;
        return result;
    }
    return result;
}

void render_cpu(unsigned char* image, int width, int height, Camera cam, int samples) {
    int total_samples = samples * samples;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4)
    #endif
    for (int py = 0; py < height; ++py) {
        for (int px = 0; px < width; ++px) {
            Vec3 color_sum(0.0f, 0.0f, 0.0f);
            for (int sy = 0; sy < samples; ++sy) {
                for (int sx = 0; sx < samples; ++sx) {
                    float jx = (sx + 0.5f) / samples;
                    float jy = (sy + 0.5f) / samples;
                    float s = (px + jx) / (float)width;
                    float t = 1.0f - (py + jy) / (float)height;

                    RayC r;
                    r.origin = cam.origin;
                    r.dir = cam.lower_left_corner + cam.horizontal * s + cam.vertical * t - cam.origin;
                    color_sum += ray_color_cpu(r);
                }
            }
            Vec3 col = color_sum * (1.0f / total_samples);
            col = clamp01(col);
            col = Vec3(sqrtf(col.x), sqrtf(col.y), sqrtf(col.z));

            int idx = (py * width + px) * 3;
            image[idx + 0] = (unsigned char)(255.99f * col.x);
            image[idx + 1] = (unsigned char)(255.99f * col.y);
            image[idx + 2] = (unsigned char)(255.99f * col.z);
        }
    }
}
