@echo off
rem =====================================================================
rem  AMD ROCm / HIP build/test/dev environment locator for Windows.
rem =====================================================================
set "PYTHONIOENCODING=utf-8"

if defined VSCMD_VER goto :have_msvc

set "_USER_VCVARS_VER=1"
if not defined VCVARS_VER (
    set "_USER_VCVARS_VER="
    set "VCVARS_VER=14.44"
)

set "_VCVARS="
for %%P in (
    "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) do (
    if not defined _VCVARS if exist %%~P set "_VCVARS=%%~P"
)

set "_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not defined _VCVARS if exist "%_VSWHERE%" (
    for /f "usebackq delims=" %%I in (`"%_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if not defined _VCVARS if exist "%%I\VC\Auxiliary\Build\vcvars64.bat" set "_VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat"
    )
)

if not defined _VCVARS (
    set "_USER_VCVARS_VER="
    echo [ERROR] Could not locate Visual Studio vcvars64.bat for ROCm build.
    exit /b 1
)

call "%_VCVARS%" -vcvars_ver=%VCVARS_VER% > nul 2>&1
if errorlevel 1 (
    if defined _USER_VCVARS_VER (
        set "_USER_VCVARS_VER="
        echo [ERROR] vcvars64.bat failed for VCVARS_VER=%VCVARS_VER%.
        exit /b 1
    )
    echo [ROCm] MSVC toolset %VCVARS_VER% not found; retrying with default VS toolset...
    call "%_VCVARS%" > nul 2>&1
    if errorlevel 1 exit /b 1
)
set "_USER_VCVARS_VER="

:have_msvc
if not defined ROCM_PATH if defined HIP_PATH set "ROCM_PATH=%HIP_PATH%"
if not defined ROCM_PATH if exist "%ProgramFiles%\AMD\ROCm" (
    for /f "delims=" %%D in ('dir /b /ad /o-n "%ProgramFiles%\AMD\ROCm" 2^>nul') do (
        if not defined ROCM_PATH set "ROCM_PATH=%ProgramFiles%\AMD\ROCm\%%D"
    )
)
if not defined ROCM_PATH (
    echo [ERROR] No ROCm / HIP SDK installation found. Set ROCM_PATH or HIP_PATH.
    exit /b 1
)
if "%ROCM_PATH:~-1%"=="\" set "ROCM_PATH=%ROCM_PATH:~0,-1%"
if not exist "%ROCM_PATH%\bin\hipcc.exe" if not exist "%ROCM_PATH%\lib\llvm\bin\clang++.exe" if not exist "%ROCM_PATH%\bin\clang++.exe" (
    echo [ERROR] ROCm compiler not found under %ROCM_PATH%.
    exit /b 1
)
set "HIP_PATH=%ROCM_PATH%"
set "PATH=%ROCM_PATH%\bin;%PATH%"
if exist "%ROCM_PATH%\lib\llvm\bin" set "PATH=%PATH%;%ROCM_PATH%\lib\llvm\bin"
exit /b 0
