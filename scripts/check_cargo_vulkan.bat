@echo off
setlocal
chcp 65001 > nul
call "%~dp0_env_vulkan.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0.."
echo [CHECK] cargo check (Vulkan)...
cargo check --workspace --all-targets --features vulkan %*
exit /b %errorlevel%
