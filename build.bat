@echo off
setlocal

:: -----------------------------------------------------------------------
:: CUDA Ray Tracer - Build Script
:: Builds: render.exe, benchmark.exe, live_demo.exe
:: Requires: MSVC (cl.exe) + CUDA Toolkit (nvcc)
:: Run from: Developer Command Prompt for VS
:: -----------------------------------------------------------------------

set NVCC=nvcc
set SRC=src
set CUDA_FLAGS=-O2 -arch=sm_89 --use_fast_math

:: -----------------------------------------------------------------------
echo [1/3] Building render.exe ...
%NVCC% %CUDA_FLAGS% ^
    %SRC%\main.cu ^
    %SRC%\kernel.cu ^
    %SRC%\cpu_render.cpp ^
    %SRC%\scene_host.cpp ^
    -o render.exe
if errorlevel 1 ( echo [ERROR] render.exe failed & exit /b 1 )

:: -----------------------------------------------------------------------
echo [2/3] Building benchmark.exe ...
%NVCC% %CUDA_FLAGS% ^
    %SRC%\benchmark.cu ^
    %SRC%\kernel.cu ^
    %SRC%\cpu_render.cpp ^
    %SRC%\scene_host.cpp ^
    -o benchmark.exe
if errorlevel 1 ( echo [ERROR] benchmark.exe failed & exit /b 1 )

:: -----------------------------------------------------------------------
echo [3/3] Building live_demo.exe ...
%NVCC% %CUDA_FLAGS% ^
    %SRC%\live_demo.cu ^
    %SRC%\kernel.cu ^
    %SRC%\scene_host.cpp ^
    -Xlinker user32.lib,gdi32.lib ^
    -o live_demo.exe
if errorlevel 1 ( echo [ERROR] live_demo.exe failed & exit /b 1 )

:: -----------------------------------------------------------------------
echo.
echo ============================================================
echo  Build successful: render.exe, benchmark.exe, live_demo.exe
echo.
echo  Usage:
echo    render.exe 1280 720 outputs\render_hd.png 4
echo    benchmark.exe
echo    live_demo.exe
echo.
echo  Live Demo Controls:
echo    WASD         = move camera
echo    LMB + drag   = look around (mouse)
echo    Q / Space    = move up
echo    E / Ctrl     = move down
echo    Arrow Up/Dn  = increase/decrease speed
echo    ESC          = quit
echo ============================================================