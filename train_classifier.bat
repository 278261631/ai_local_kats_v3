@echo off
setlocal

cd /d "%~dp0"

echo [INFO] Training classifier...
python "train_and_test_classifier.py" %*
if errorlevel 1 (
  echo [ERROR] Training failed.
  exit /b 1
)

echo [INFO] Training finished successfully.
endlocal
