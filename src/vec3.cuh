// vec3.cuh
// Minimal 3D vector math, usable on both host and device (__host__ __device__).
#pragma once
#include <cuda_runtime.h>
#include <math.h>

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    __host__ __device__ Vec3 operator*(const Vec3& o) const { return Vec3(x * o.x, y * o.y, z * o.z); }
    __host__ __device__ Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }

    __host__ __device__ Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    __host__ __device__ Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};

__host__ __device__ inline Vec3 operator*(float s, const Vec3& v) { return v * s; }

__host__ __device__ inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float length(const Vec3& v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline Vec3 normalize(const Vec3& v) {
    float len = length(v);
    if (len < 1e-8f) return v;
    float inv = 1.0f / len;
    return Vec3(v.x * inv, v.y * inv, v.z * inv);
}

__host__ __device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__host__ __device__ inline Vec3 clamp01(const Vec3& v) {
    return Vec3(fminf(fmaxf(v.x, 0.0f), 1.0f),
                fminf(fmaxf(v.y, 0.0f), 1.0f),
                fminf(fmaxf(v.z, 0.0f), 1.0f));
}
