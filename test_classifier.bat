@echo off
setlocal

cd /d "%~dp0"

echo [INFO] Testing classifier...
python "test_classifier.py" %*
if errorlevel 1 (
  echo [ERROR] Testing failed.
  exit /b 1
)

echo [INFO] Testing finished successfully.
endlocal
