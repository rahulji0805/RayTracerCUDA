# CUDA C++ Ray Tracer

**Target GPU:** NVIDIA RTX 4060 Laptop GPU (Ada Lovelace, `sm_89`)

A CUDA C++ ray tracer with an interactive real-time demo. One CUDA thread renders one pixel. 168 spheres, diffuse and metal materials, hard shadows, procedural checkerboard floor, NxN supersampled anti-aliasing. Includes a Win32 interactive window (`live_demo.exe`) running at **313 FPS** on the target hardware.

## What's actually CUDA here

- `src/kernel.cu` — `__global__ render_kernel`: one thread per pixel, ray-sphere intersection against all 168 spheres, Lambert shading, hard shadow rays, iterative metal-reflection bounce loop (`MAX_DEPTH = 5`), procedural checkerboard, NxN supersampled AA.
- `src/main.cu` — host code: `cudaMalloc` for the framebuffer, kernel launch, `cudaMemcpy` back to host, PNG output via `stb_image_write`.
- `src/benchmark.cu` — CPU (OpenMP) vs GPU timing harness. Two GPU paths: full (alloc + transfer + kernel) and kernel-only (buffers allocated once outside the loop, isolates pure compute).
- `src/live_demo.cu` — Win32 interactive window. CUDA renders each frame into a device buffer, `cudaMemcpy` to pinned host memory, then `BitBlt` to screen. WASD + mouse camera, real-time FPS counter in title bar.
- Scene data lives in GPU **constant memory** (`__constant__ float d_spheres[168][8]`) — the correct memory space for read-only data accessed identically by every thread.

## Project structure

```
RayTracerCUDA_Cpp/
├── src/
│   ├── vec3.cuh             # __host__ __device__ vector math
│   ├── camera.cuh           # look-at camera, build_camera()
│   ├── scene_data.cuh       # 168-sphere scene in __constant__ memory
│   ├── scene_host.h/.cpp    # host-accessible scene data (CPU benchmark)
│   ├── kernel.cu/.cuh       # CUDA kernel + render_gpu() launch wrapper
│   ├── cpu_render.h/.cpp    # CPU reference renderer (OpenMP), benchmark baseline
│   ├── main.cu              # render.exe entry point
│   ├── benchmark.cu         # benchmark.exe entry point
│   ├── live_demo.cu         # live_demo.exe — real-time Win32 interactive window
│   └── stb_image_write.h    # single-header PNG writer (github.com/nothings/stb)
├── build.bat                # nvcc build script (Windows, targets sm_89)
├── gen_scene.py             # (dev tool) regenerates scene_data.cuh/scene_host.cpp
└── outputs/                 # renders + benchmark CSV land here
```

## Requirements

1. **NVIDIA CUDA Toolkit** — tested on v13.3 (`nvcc --version` to confirm)
2. **Visual Studio Build Tools** with the "Desktop development with C++" workload — `nvcc` needs MSVC's `cl.exe` as host compiler on Windows.
   Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Build

Run from a **Developer Command Prompt for VS**:

```bat
build.bat
```

Produces three executables: `render.exe`, `benchmark.exe`, `live_demo.exe` — all targeting `sm_89` with `-O2 --use_fast_math`.

If `nvcc` reports it can't find `cl.exe`, open a "Developer Command Prompt for VS" (not a regular CMD) and retry.

## Run

### Static render
```bat
render.exe 1280 720 outputs\render_hd.png 4
```
Arguments: `[width] [height] [output.png] [samples]` — `samples` is the NxN AA grid (4 = 4×4 = 16 rays/pixel).

### Benchmark
```bat
benchmark.exe
```
Sweeps 320×180 through 3840×2160, prints CPU vs GPU-full vs GPU-kernel-only table, writes `outputs/benchmark_cpp.csv`.

### Interactive live demo
```bat
live_demo.exe
```

| Control | Action |
|---|---|
| `WASD` | Move camera |
| Left mouse button + drag | Look around |
| `Q` / `Space` | Move up |
| `E` / `Ctrl` | Move down |
| `↑` / `↓` | Increase / decrease speed |
| `ESC` | Quit |

## Performance

### Live demo — RTX 4060 Laptop GPU (plugged in, ~90W-115W)

**313 FPS @ 1280×720** — 3.2 ms per frame, 1 sample/pixel (no AA), full Win32 display pipeline included.

### Benchmark — RTX 4060 Laptop GPU

"GPU full" includes `cudaMalloc` + `cudaMemcpy` + kernel. "GPU kernel-only" times just the kernel with buffers allocated once outside the loop.

#### Plugged in (~90W)

| Resolution | CPU (ms) | GPU full (ms) | Speedup | GPU kernel-only (ms) | Speedup |
|---|---|---|---|---|---|
| 320×180   | 96.14    | 2.65   | 36.3x  | 1.91   | 50.2x  |
| 640×360   | 365.74   | 6.11   | 59.9x  | 5.41   | 67.6x  |
| 800×450   | 553.01   | 9.16   | 60.4x  | 8.10   | 68.2x  |
| 1280×720  | 1441.04  | 47.14  | 30.6x  | 18.85  | 76.4x  |
| 1920×1080 | 3331.25  | 39.39  | 84.6x  | 30.74  | 108.4x |
| 3840×2160 | 15637.74 | 124.60 | 125.5x | 119.63 | 130.7x |

#### On battery (~60W, NVIDIA TGP-throttled)

| Resolution | CPU (ms) | GPU full (ms) | Speedup | GPU kernel-only (ms) | Speedup |
|---|---|---|---|---|---|
| 320×180   | 103.26   | 7.64   | 13.5x | 7.06   | 14.6x  |
| 640×360   | 381.95   | 7.00   | 54.6x | 5.70   | 67.0x  |
| 800×450   | 588.52   | 9.74   | 60.4x | 8.37   | 70.3x  |
| 1280×720  | 1565.57  | 100.48 | 15.6x | 55.53  | 28.2x  |
| 1920×1080 | 4398.55  | 135.50 | 32.5x | 62.31  | 70.6x  |
| 3840×2160 | 16943.92 | 259.53 | 65.3x | 146.63 | 115.6x |

**Key result:** up to **130x** kernel-only speedup at 4K (plugged in). The battery vs plugged-in gap is direct evidence of mobile GPU TGP throttling — same kernel, same scene, same thread count, but sustained clocks drop significantly at 60W vs 90W. The dip at 1280×720 in the plugged-in run is a GPU clock-state transition artifact, not an algorithmic issue.

## Implementation notes

**Memory model:** Scene geometry (168 spheres, 8 floats each) sits in `__constant__` memory — broadcast-cached per SM, zero overhead for uniform reads across a warp.

**Warp divergence:** Each warp contains threads that hit diffuse spheres, metal spheres, the floor, and misses (sky). The divergent branches (Lambert vs reflection vs checkerboard vs background) are unavoidable given per-pixel ray outcomes and account for the gap between theoretical peak throughput and measured numbers.

**Transfer overhead:** At small resolutions the `cudaMalloc`/`cudaMemcpy` round-trip dominates (see 320×180: 2.65ms full vs 1.91ms kernel-only). At 4K the kernel runtime dwarfs the transfer cost, so full and kernel-only speedups converge.

**Live demo pipeline:** CUDA kernel → `cudaMemcpy` to pinned host buffer → RGB→BGRA swap → `BitBlt` to Win32 window. Pinned (page-locked) host memory avoids an extra OS copy on the DMA transfer path.

## Future work

- BVH acceleration structure to replace the brute-force O(n) sphere loop — reduces per-ray intersection cost and warp divergence at scale
- CUDA–OpenGL interop for zero-copy display (shared kernel/display buffer, eliminates the `cudaMemcpy` + `BitBlt` path entirely)
- Path tracing with Monte Carlo AA (random per-sample jitter instead of fixed NxN grid) for soft shadows and indirect lighting
- Multi-bounce dielectrics (glass/refraction)

## License

MIT

---

**Author:**  
Rahul Bhukal  
Department of Electronics and Communication Engineering  
Deenbandhu Chhotu Ram University of Science and Technology