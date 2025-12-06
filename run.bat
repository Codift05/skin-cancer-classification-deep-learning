@echo off
REM Skin Cancer Classification - Project Setup & Run Script
REM Windows batch script

echo.
echo ==========================================
echo  Skin Cancer Classification Project
echo ==========================================
echo.

:menu
echo.
echo Select option:
echo 1. Install dependencies
echo 2. Run training notebook (Jupyter)
echo 3. Run web app (Streamlit)
echo 4. Install & setup (first time)
echo 5. Exit
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto install_deps
if "%choice%"=="2" goto run_notebook
if "%choice%"=="3" goto run_app
if "%choice%"=="4" goto first_setup
if "%choice%"=="5" goto exit_script
goto menu

:install_deps
echo.
echo Installing dependencies...
pip install -r requirements.txt
pause
goto menu

:run_notebook
echo.
echo Starting Jupyter Lab with training notebook...
cd notebook
jupyter lab training.ipynb
cd ..
goto menu

:run_app
echo.
echo Starting Streamlit web app...
streamlit run app/app.py
goto menu

:first_setup
echo.
echo ==========================================
echo  First Time Setup
echo ==========================================
echo.
echo This will:
echo - Create virtual environment
echo - Install all dependencies
echo - Check dataset structure
echo.
pause

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ==========================================
echo  Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Place dataset in data/ folder (benign/ and malignant/ subdirectories)
echo 2. Run: python setup.py --train  (to train model)
echo 3. Run: streamlit run app/app.py  (to start web app)
echo.
pause
goto menu

:exit_script
echo.
echo Goodbye!
pause
exit /b
