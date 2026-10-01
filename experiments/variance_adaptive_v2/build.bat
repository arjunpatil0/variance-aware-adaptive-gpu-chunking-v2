@echo off
REM build.bat — Compile variance_adaptive_v2 on Windows using nvcc + MSVC.
REM
REM Requirements:
REM   NVIDIA CUDA Toolkit (nvcc on PATH)
REM   Microsoft C++ Build Tools (vcvars64.bat must be callable)

REM Activate MSVC environment so nvcc can find cl.exe
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Could not activate MSVC environment via vcvars64.bat.
    echo          nvcc may still work if cl.exe is already on PATH.
)
REM   Microsoft C++ Build Tools (cl.exe, invoked by nvcc)
REM
REM The -allow-unsupported-compiler flag is included because the CUDA Toolkit
REM may not officially support the installed MSVC version; this matches the
REM approach used in the original project's build instructions.
REM
REM All translation units flow through nvcc:
REM   .cu files  -> compiled as CUDA device + host code
REM   .cpp files -> included from main.cu; compiled as host code by MSVC
REM
REM Output: variance_adaptive_v2.exe (in the variance_adaptive_v2/ directory)

echo Building variance_adaptive_v2...

nvcc -allow-unsupported-compiler src/main.cu -o variance_adaptive_v2.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build successful: variance_adaptive_v2.exe
    echo.
    echo Usage:
    echo   variance_adaptive_v2.exe --calibrate
    echo   variance_adaptive_v2.exe --experiment C --multiplier 50 --reps 20 --warmup 3
    echo   variance_adaptive_v2.exe --experiment A --multiplier 50 --reps 20 --warmup 3
    echo   variance_adaptive_v2.exe --experiment B --multiplier 50 --reps 20 --warmup 3
    echo   variance_adaptive_v2.exe --experiment D --multiplier 50 --reps 20 --warmup 3
) else (
    echo.
    echo Build FAILED. Check nvcc output above.
    exit /b 1
)
