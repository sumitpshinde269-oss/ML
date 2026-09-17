@echo off
setlocal
cd /d "%~dp0"
call "%~dp0ultralytics\.venv\Scripts\python.exe" "%~dp0main.py" %*
