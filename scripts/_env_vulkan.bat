@echo off
rem =====================================================================
rem  Vulkan build/test/dev environment locator for Windows.
rem =====================================================================
set "PYTHONIOENCODING=utf-8"

if defined VSCMD_VER goto :have_msvc

set "_VCVARS="
for %%P in (
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
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
    echo [ERROR] Could not locate Visual Studio vcvars64.bat for Vulkan build.
    exit /b 1
)

call "%_VCVARS%" > nul
if errorlevel 1 exit /b 1

:have_msvc
if not exist "%SystemRoot%\System32\vulkan-1.dll" (
    echo [Vulkan] vulkan-1.dll not found. Without it the app falls back to CPU.
)
exit /b 0
