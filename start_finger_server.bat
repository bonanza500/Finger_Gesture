@echo off
REM Launcher for Finger Gesture Detection Server

echo ============================================================
echo   Finger Gesture Detection Server
echo   10 Gestures (1-10)
echo ============================================================
echo.

REM Change to script directory
cd /d "%~dp0"

echo Starting server on localhost:5001...
echo.
echo If you see errors about missing packages:
echo   pip install mediapipe opencv-python flask flask-cors scikit-image joblib
echo.
echo Press Ctrl+C to stop the server
echo.
echo ============================================================
echo.

python finger_server.py

echo.
echo ============================================================
echo   Server stopped
echo ============================================================
echo.
pause
