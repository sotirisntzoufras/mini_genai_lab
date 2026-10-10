@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
exit /b
:missing
echo Run install_windows_cpu.bat first, then start this file again.
pause
exit /b 1
