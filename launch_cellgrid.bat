@echo off
setlocal
echo Starting CellGrid Figure Generator...
rem Keep the workstation's Conda environment and import Torch before Streamlit.
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" cellgrid-win-gpu
if errorlevel 1 goto failed
cd /d "%~dp0"
if errorlevel 1 goto failed
python -c "import torch; from streamlit.web import cli; cli.main()" run figure_app.py --server.port=0 --server.headless=false
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo CellGrid could not start. See the error above.
pause
exit /b 1
