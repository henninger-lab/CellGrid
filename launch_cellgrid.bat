@echo off
echo Starting CellGrid Figure Generator...
call C:\Users\zeiss\miniconda3\Scripts\activate.bat cellgrid-win-gpu
cd /d C:\CellGrid
streamlit run figure_app.py --server.port=0 --server.headless=false
pause