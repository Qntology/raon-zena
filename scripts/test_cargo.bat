@echo off
setlocal
chcp 65001 > nul
call "%~dp0_env_cuda.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0.."
echo [TEST] cargo test --workspace (CUDA + FlashAttention)...
cargo test --workspace --features cuda,flash-attn %*
exit /b %errorlevel%
