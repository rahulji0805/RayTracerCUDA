# Installation & Setup

## 1. Install NVIDIA CUDA Toolkit

Download from: https://developer.nvidia.com/cuda-downloads

Select: Windows → x86_64 → your Windows version → exe (local)

After install, verify:
```bat
nvcc --version
```
Should show `release 12.x` or higher.

---

## 2. Install Visual Studio Build Tools

Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

During install, select workload: **Desktop development with C++**

This installs `cl.exe` (MSVC compiler), which `nvcc` needs as its host compiler on Windows.

After install, always build from **Developer Command Prompt for VS** — not a regular CMD or PowerShell.

To open it: Start Menu → search "Developer Command Prompt for VS 2022" → open.

Verify:
```bat
cl
```
Should show: `Microsoft (R) C/C++ Optimizing Compiler Version 19.x`

---

## 3. Verify GPU

```bat
nvidia-smi
```
Should show your GPU name and driver version. CUDA Toolkit version must match or be lower than the driver's supported CUDA version (shown in top-right of `nvidia-smi` output).

---

## 4. Clone and build

```bat
git clone https://github.com/rahulji0805/RayTracerCUDA.git
cd RayTracerCUDA
build.bat
```

If your GPU is not an RTX 4060 (sm_89), edit `build.bat` and change `-arch=sm_89` to match your GPU:

| GPU | arch flag |
|---|---|
| RTX 30 series (Ampere) | `sm_86` |
| RTX 40 series (Ada) | `sm_89` |
| RTX 20 series (Turing) | `sm_75` |
| GTX 10 series (Pascal) | `sm_61` |

Or use `-arch=native` to auto-detect.

---

## 5. Run

```bat
render.exe 1280 720 outputs\render_hd.png 4
benchmark.exe
live_demo.exe
```

See README.md for full usage details.