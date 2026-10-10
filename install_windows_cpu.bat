@echo off
setlocal
cd /d "%~dp0"
py -3.11 --version >nul 2>&1
if errorlevel 1 goto no_python
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Installation completed. Double-click start_windows.bat to open the app.
pause
exit /b 0
:no_python
echo Python 3.11 with the Python launcher was not found. Install it and try again.
pause
exit /b 1
:failed
echo Installation failed. Read the error above and keep this window open for help.
pause
exit /b 1
