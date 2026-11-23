@echo off
setlocal
chcp 65001 > nul
call "%~dp0_env_rocm.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0.."
echo [TEST] cargo test --workspace (ROCm / %ROCM_PATH%)...
cargo test --workspace --features rocm %*
exit /b %errorlevel%
