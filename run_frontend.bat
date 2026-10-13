@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ===================================================================
echo  Starting Localhost Server for Gen Z Sustainable Fashion Analysis 
echo ===================================================================
echo.
echo Opening localhost in your browser...
echo.

if exist "C:\Users\Heli P Shah\ANACONDA\python.exe" (
    "C:\Users\Heli P Shah\ANACONDA\python.exe" server.py
) else (
    python server.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Fallback: Starting standard Python HTTP server on port 8000...
    start http://localhost:8000/index.html
    python -m http.server 8000
)

pause
