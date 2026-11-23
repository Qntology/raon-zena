@echo off
setlocal
chcp 65001 > nul
call "%~dp0_env_rocm.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0.."
echo [BUILD] Building raon-zena (ROCm / Release)...
cargo build --release -p raon-zena --features rocm %*
exit /b %errorlevel%
