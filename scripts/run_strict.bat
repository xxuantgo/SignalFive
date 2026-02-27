@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."
if not exist logs mkdir logs

for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd_HHmmss')"') do set "TS=%%I"
set "LOG_FILE=logs\strict_oos_%TS%.log"

echo [SignalFive] Start strict OOS pipeline
echo [SignalFive] Log file: %LOG_FILE%

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
python src/signalfive/pipelines/run_strict_oos_stitch.py ^
  --n-trials 180 ^
  --n-startup-trials 36 ^
  --seed 7 ^
  --outer-test-start 2021-01-04 ^
  --outer-test-end 2025-10-30 ^
  --outer-test-months 12 ^
  --outer-step-months 12 ^
  --inner-train-months 24 ^
  --inner-test-months 12 ^
  --inner-step-months 12 ^
  --inner-min-folds 2 ^
  --inner-fallback-valid-days 120 ^
  --min-train-days 220 ^
  --no-bayes-first-n-outer 2 ^
  --top-n-low 3 ^
  --top-n-high 7 ^
  --top-n-step 1 ^
  --alpha-low 0.93 ^
  --alpha-high 0.95 ^
  --window-low 120 ^
  --window-high 180 ^
  --window-step 5 ^
  --lambda-low 3e-3 ^
  --lambda-high 1.5e-2 ^
  --beta-low 0.05 ^
  --beta-high 0.25 ^
  --beta-step 0.02 ^
  --cvar-methods cornish_fisher,empirical ^
  --obj-std-penalty 1.5 ^
  --obj-worst-penalty 1.8 ^
  --obj-sharpe-floor 0.2 ^
  --obj-turnover-penalty 0.3 ^
  --export-backtest-plots > "%LOG_FILE%" 2>&1

set "RET=%ERRORLEVEL%"
if not "%RET%"=="0" (
  echo [SignalFive] Failed, exit code: %RET%
  echo [SignalFive] Check log: %LOG_FILE%
  exit /b %RET%
)

echo [SignalFive] Completed successfully
echo [SignalFive] Check log: %LOG_FILE%
