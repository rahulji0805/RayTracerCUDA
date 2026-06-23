// camera.cuh
#pragma once
#include "vec3.cuh"

struct Camera {
    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
};

// Same look-at camera math as the original Python build_camera().
inline Camera build_camera(int width, int height, float vfov_deg,
                            Vec3 look_from, Vec3 look_at, Vec3 vup = Vec3(0, 1, 0)) {
    float aspect_ratio = (float)width / (float)height;
    float theta = vfov_deg * 3.14159265358979323846f / 180.0f;
    float h = tanf(theta / 2.0f);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect_ratio * viewport_height;

    Vec3 w = normalize(look_from - look_at);
    Vec3 u = normalize(cross(vup, w));
    Vec3 v = cross(w, u);

    Camera cam;
    cam.origin = look_from;
    cam.horizontal = u * viewport_width;
    cam.vertical = v * viewport_height;
    cam.lower_left_corner = look_from - cam.horizontal * 0.5f - cam.vertical * 0.5f - w;
    return cam;
}
