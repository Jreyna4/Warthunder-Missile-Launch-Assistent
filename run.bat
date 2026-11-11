@echo off
cd /d "%~dp0"
python -c "import numpy, openpyxl" 2>NUL || python -m pip install -r requirements.txt
python -m wtmla %*
if errorlevel 1 pause
