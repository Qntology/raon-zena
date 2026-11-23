@echo off
setlocal
chcp 65001 > nul
call "%~dp0_env_rocm.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0.."
echo [CHECK] cargo check (ROCm / %ROCM_PATH%)...
cargo check --workspace --all-targets --features rocm %*
exit /b %errorlevel%
