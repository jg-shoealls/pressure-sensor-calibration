@echo off
setlocal
pushd "%~dp0"

rem ===================================================================
rem Pressure Monitor - PyInstaller build script
rem
rem Usage:
rem   build.bat fast   : recommended distribution, keeps ML, starts faster
rem   build.bat full   : single exe, keeps ML, slower first start
rem   build.bat lite   : smaller package, ML disabled, fastest/smallest
rem
rem Required environment:
rem   py -3.11 -m venv venv
rem   venv\Scripts\activate
rem   pip install pyserial numpy matplotlib torch pyinstaller
rem ===================================================================

set "MODE=%~1"
if "%MODE%"=="" set "MODE=fast"
if /I not "%MODE%"=="fast" if /I not "%MODE%"=="full" if /I not "%MODE%"=="lite" (
    echo ERROR: Unknown build mode "%MODE%".
    echo Use one of: fast, full, lite
    pause
    popd
    exit /b 1
)

echo [1/7] Activating virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv was not found.
    echo Create it with:
    echo   py -3.11 -m venv venv
    pause
    popd
    exit /b 1
)

call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate venv.
    pause
    popd
    exit /b 1
)

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: The Python executable inside venv is broken.
    echo Fix:
    echo   1. Delete the venv folder.
    echo   2. Run: py -3.11 -m venv venv
    echo   3. Run: venv\Scripts\activate
    echo   4. Run: pip install pyserial numpy matplotlib torch pyinstaller
    pause
    popd
    exit /b 1
)

echo [2/7] Checking required packages...
if /I "%MODE%"=="lite" (
    python -c "import serial, numpy, matplotlib" >nul 2>&1
) else (
    python -c "import serial, numpy, matplotlib, torch" >nul 2>&1
)
if errorlevel 1 (
    echo ERROR: Required packages are missing.
    echo Install them with:
    if /I "%MODE%"=="lite" (
        echo   pip install pyserial numpy matplotlib pyinstaller
    ) else (
        echo   pip install pyserial numpy matplotlib torch pyinstaller
    )
    pause
    popd
    exit /b 1
)

echo [3/7] Checking PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller.
        pause
        popd
        exit /b 1
    )
)

echo [4/7] Cleaning old build output...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist PressureMonitor.spec del PressureMonitor.spec

set "APPNAME=PressureMonitor"
set "ONEFILE_FLAG="
set "TORCH_FLAGS=--hidden-import=torch --hidden-import=torch.nn --exclude-module torch.utils.tensorboard --exclude-module torch.onnx --exclude-module torch._inductor --exclude-module triton"

if /I "%MODE%"=="full" (
    set "ONEFILE_FLAG=--onefile"
)

if /I "%MODE%"=="lite" (
    set "APPNAME=PressureMonitor_Lite"
    set "TORCH_FLAGS=--exclude-module torch --exclude-module torch.nn --exclude-module torch.utils --exclude-module torch.optim --exclude-module torch.distributed --exclude-module torch.testing --exclude-module torch.onnx --exclude-module torch._dynamo --exclude-module torch._inductor --exclude-module triton"
    set "PRESSURE_UI_DISABLE_TORCH=1"
)

echo [5/7] Building %MODE% package. This can take several minutes...
python -m PyInstaller --clean --noconfirm %ONEFILE_FLAG% --windowed --name %APPNAME% ^
    --hidden-import=matplotlib.backends.backend_tkagg ^
    --hidden-import=ml_anomaly ^
    --collect-data matplotlib ^
    %TORCH_FLAGS% ^
    pressure_monitor.py
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed.
    echo Close any running %APPNAME%.exe window and run this build again.
    pause
    popd
    exit /b 1
)

echo.
if /I "%MODE%"=="full" (
    set "DIST_ROOT=dist"
) else (
    set "DIST_ROOT=dist\%APPNAME%"
)

if exist "%DIST_ROOT%\%APPNAME%.exe" (
    echo [6/7] Preparing distribution files...
    if not exist "%DIST_ROOT%\records" mkdir "%DIST_ROOT%\records"
    if not exist "%DIST_ROOT%\clips" mkdir "%DIST_ROOT%\clips"
    if /I not "%MODE%"=="lite" (
        if exist "anomaly_model.pt" copy /Y "anomaly_model.pt" "%DIST_ROOT%\anomaly_model.pt" >nul
        if exist "anomaly_stats.npz" copy /Y "anomaly_stats.npz" "%DIST_ROOT%\anomaly_stats.npz" >nul
    )
    if exist "calibration.json" copy /Y "calibration.json" "%DIST_ROOT%\calibration.json" >nul

    echo [7/7] Build complete.
    echo =====================================================
    echo Mode:
    echo   %MODE%
    echo.
    echo Run:
    echo   %DIST_ROOT%\%APPNAME%.exe
    echo.
    if /I "%MODE%"=="fast" (
        echo Distribute the whole folder:
        echo   %DIST_ROOT%
        echo This mode keeps ML and starts faster than onefile.
    )
    if /I "%MODE%"=="full" (
        echo Distribute the dist folder or the exe with model files.
        echo This mode keeps ML but startup can be slower.
    )
    if /I "%MODE%"=="lite" (
        echo Distribute the whole folder:
        echo   %DIST_ROOT%
        echo This mode disables PyTorch ML features to reduce size.
    )
    echo =====================================================
) else (
    echo Build failed. Check the error output above.
)

popd
pause
