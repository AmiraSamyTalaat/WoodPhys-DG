@echo off
cd /d %~dp0
python scripts\stage4r\run_all_stage4r.py --force
if errorlevel 1 exit /b 1
