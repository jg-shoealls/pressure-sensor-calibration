@echo off
REM ===================================================================
REM 압력센서 모니터 - exe 빌드 스크립트
REM
REM 사용법: 이 파일을 pressure_monitor.py 와 같은 폴더(pressure_ui)에 넣고
REM         더블클릭 실행 (또는 터미널에서 build.bat 입력)
REM
REM 전제조건: venv가 이미 만들어져 있어야 함 (python -m venv venv)
REM ===================================================================

echo [1/4] 가상환경 활성화...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo 오류: venv를 찾을 수 없습니다. 먼저 "python -m venv venv"로 가상환경을 만드세요.
    pause
    exit /b 1
)

echo [2/4] pyinstaller 설치 확인...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo pyinstaller 설치 중...
    pip install pyinstaller
)

echo [3/4] 기존 빌드 결과물 정리...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist PressureMonitor.spec del PressureMonitor.spec

echo [4/4] exe 빌드 시작 (몇 분 정도 걸릴 수 있습니다)...
pyinstaller --onefile --windowed --name PressureMonitor ^
    --hidden-import=matplotlib.backends.backend_tkagg ^
    --collect-data matplotlib ^
    pressure_monitor.py

echo.
if exist dist\PressureMonitor.exe (
    echo =====================================================
    echo 빌드 완료!  dist\PressureMonitor.exe 를 확인하세요.
    echo =====================================================
) else (
    echo 빌드 실패. 위의 오류 메시지를 확인하세요.
)

pause
