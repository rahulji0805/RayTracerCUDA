// live_demo.cu
// Win32 + CUDA interactive ray tracer
// Controls: WASD = move, LMB+drag = look, ESC = quit
// Build: nvcc -O2 -arch=sm_89 src/live_demo.cu src/kernel.cu src/scene_host.cpp -Xlinker user32.lib,gdi32.lib -o live_demo.exe

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>

#include "vec3.cuh"
#include "camera.cuh"
#include "kernel.cuh"

// -----------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------
static constexpr int WIN_W   = 1280;
static constexpr int WIN_H   = 720;
static constexpr int SAMPLES = 1;    // 1 sample = real-time speed

// -----------------------------------------------------------------------
// Camera state
// -----------------------------------------------------------------------
struct CamState {
    Vec3  pos   = Vec3(0.0f, 1.5f, 6.0f);
    float yaw   = -90.0f;   // degrees
    float pitch =  -8.0f;
    float speed =  4.0f;
    float sens  =  0.15f;
};

static CamState g_cam;

static Vec3 cam_forward(const CamState& c) {
    float yr = c.yaw   * 3.14159265f / 180.0f;
    float pr = c.pitch * 3.14159265f / 180.0f;
    return normalize(Vec3(cosf(pr)*cosf(yr), sinf(pr), cosf(pr)*sinf(yr)));
}

// -----------------------------------------------------------------------
// CUDA buffers
// -----------------------------------------------------------------------
static unsigned char* d_image = nullptr;
static unsigned char* h_image = nullptr;  // pinned

// -----------------------------------------------------------------------
// Input state
// -----------------------------------------------------------------------
static bool g_keys[256]     = {};
static bool g_mouse_cap     = false;
static int  g_mouse_last_x  = 0;
static int  g_mouse_last_y  = 0;

// -----------------------------------------------------------------------
// DIB section (Win32 blitting)
// -----------------------------------------------------------------------
static HBITMAP g_hBitmap = nullptr;
static void*   g_pBits   = nullptr;
static HDC     g_hMemDC  = nullptr;

static void create_dib(HWND hwnd) {
    HDC hdc = GetDC(hwnd);
    BITMAPINFO bmi     = {};
    bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth       =  WIN_W;
    bmi.bmiHeader.biHeight      = -WIN_H;  // top-down
    bmi.bmiHeader.biPlanes      = 1;
    bmi.bmiHeader.biBitCount    = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    g_hBitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &g_pBits, nullptr, 0);
    g_hMemDC  = CreateCompatibleDC(hdc);
    SelectObject(g_hMemDC, g_hBitmap);
    ReleaseDC(hwnd, hdc);
}

// -----------------------------------------------------------------------
// Render one frame
// -----------------------------------------------------------------------
static void render_frame(float dt) {
    // Update position
    Vec3  fwd   = cam_forward(g_cam);
    Vec3  world_up(0, 1, 0);
    Vec3  right = normalize(cross(fwd, world_up));

    float spd = g_cam.speed * dt;
    if (g_keys['W']) g_cam.pos += fwd   * spd;
    if (g_keys['S']) g_cam.pos = g_cam.pos - fwd   * spd;
    if (g_keys['A']) g_cam.pos = g_cam.pos - right * spd;
    if (g_keys['D']) g_cam.pos += right * spd;
    if (g_keys['Q'] || g_keys[VK_SPACE])   g_cam.pos += world_up * spd;
    if (g_keys['E'] || g_keys[VK_CONTROL]) g_cam.pos = g_cam.pos - world_up * spd;

    // Clamp pitch
    if (g_cam.pitch >  89.0f) g_cam.pitch =  89.0f;
    if (g_cam.pitch < -89.0f) g_cam.pitch = -89.0f;

    // Build camera — matches build_camera(width, height, vfov, from, at, up)
    Vec3 look_at = g_cam.pos + cam_forward(g_cam);
    Camera cam = build_camera(WIN_W, WIN_H, 60.0f, g_cam.pos, look_at, world_up);

    // Launch kernel — matches render_gpu(d_image, width, height, cam, samples)
    render_gpu(d_image, WIN_W, WIN_H, cam, SAMPLES);
    cudaDeviceSynchronize();

    // Copy GPU → pinned host
    // kernel writes RGB (3 bytes/pixel)
    cudaMemcpy(h_image, d_image, WIN_W * WIN_H * 3, cudaMemcpyDeviceToHost);

    // Convert RGB → BGRA for Win32 DIB
    unsigned char* src = h_image;
    unsigned char* dst = (unsigned char*)g_pBits;
    for (int i = 0; i < WIN_W * WIN_H; i++) {
        dst[0] = src[2];  // B
        dst[1] = src[1];  // G
        dst[2] = src[0];  // R
        dst[3] = 0;
        src += 3;
        dst += 4;
    }
}

// -----------------------------------------------------------------------
// Window procedure
// -----------------------------------------------------------------------
static HWND g_hwnd = nullptr;

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    case WM_KEYDOWN:
        if (wp < 256) g_keys[wp] = true;
        if (wp == VK_ESCAPE)   DestroyWindow(hwnd);
        if (wp == VK_UP)   g_cam.speed *= 1.5f;
        if (wp == VK_DOWN) g_cam.speed /= 1.5f;
        return 0;

    case WM_KEYUP:
        if (wp < 256) g_keys[wp] = false;
        return 0;

    case WM_LBUTTONDOWN:
        g_mouse_cap    = true;
        g_mouse_last_x = LOWORD(lp);
        g_mouse_last_y = HIWORD(lp);
        SetCapture(hwnd);
        ShowCursor(FALSE);
        return 0;

    case WM_LBUTTONUP:
        g_mouse_cap = false;
        ReleaseCapture();
        ShowCursor(TRUE);
        return 0;

    case WM_MOUSEMOVE:
        if (g_mouse_cap) {
            int dx = (int)LOWORD(lp) - g_mouse_last_x;
            int dy = (int)HIWORD(lp) - g_mouse_last_y;
            g_cam.yaw   += dx * g_cam.sens;
            g_cam.pitch -= dy * g_cam.sens;
            g_mouse_last_x = LOWORD(lp);
            g_mouse_last_y = HIWORD(lp);
        }
        return 0;

    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        BitBlt(hdc, 0, 0, WIN_W, WIN_H, g_hMemDC, 0, 0, SRCCOPY);
        EndPaint(hwnd, &ps);
        return 0;
    }
    }
    return DefWindowProc(hwnd, msg, wp, lp);
}

// -----------------------------------------------------------------------
// WinMain
// -----------------------------------------------------------------------
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int) {
    // kernel outputs RGB 3 bytes/pixel
    size_t img_bytes = (size_t)WIN_W * WIN_H * 3;
    cudaMalloc(&d_image, img_bytes);
    cudaMallocHost(&h_image, img_bytes);

    // Register window class
    WNDCLASSEX wc    = {};
    wc.cbSize        = sizeof(wc);
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hInst;
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = "CUDARayTracerLive";
    RegisterClassEx(&wc);

    RECT rc = {0, 0, WIN_W, WIN_H};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX, FALSE);
    g_hwnd = CreateWindowEx(
        0, wc.lpszClassName,
        "CUDA Ray Tracer  |  WASD=move  LMB+drag=look  Up/Dn=speed  ESC=quit",
        WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top,
        nullptr, nullptr, hInst, nullptr);

    create_dib(g_hwnd);
    ShowWindow(g_hwnd, SW_SHOW);
    UpdateWindow(g_hwnd);

    // Main loop
    auto   t_prev     = std::chrono::high_resolution_clock::now();
    float  fps_accum  = 0.0f;
    int    fps_count  = 0;

    MSG msg = {};
    while (true) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) goto done;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        auto  t_now = std::chrono::high_resolution_clock::now();
        float dt    = std::chrono::duration<float>(t_now - t_prev).count();
        t_prev = t_now;
        if (dt > 0.1f) dt = 0.1f;

        render_frame(dt);

        HDC hdc = GetDC(g_hwnd);
        BitBlt(hdc, 0, 0, WIN_W, WIN_H, g_hMemDC, 0, 0, SRCCOPY);
        ReleaseDC(g_hwnd, hdc);

        fps_accum += dt;
        fps_count++;
        if (fps_count >= 30) {
            float fps = fps_count / fps_accum;
            char  title[256];
            snprintf(title, sizeof(title),
                "CUDA Ray Tracer  |  %.0f FPS  %.1f ms  |  WASD=move  LMB+drag=look  Up/Dn=speed  ESC=quit",
                fps, 1000.0f / fps);
            SetWindowTextA(g_hwnd, title);
            fps_accum = 0.0f;
            fps_count = 0;
        }
    }

done:
    cudaFreeHost(h_image);
    cudaFree(d_image);
    DeleteDC(g_hMemDC);
    DeleteObject(g_hBitmap);
    return 0;
}
