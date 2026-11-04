@echo off
rem Build the standalone Windows app into dist\PurgeSimulator\
rem Run from anywhere; needs Python 3.10+ on PATH.
cd /d "%~dp0\.."
python -m pip install -r requirements.txt pyinstaller || exit /b 1
python -m PyInstaller packaging\purge_simulator.spec --noconfirm || exit /b 1
if not exist dist\PurgeSimulator\PurgeSimData mkdir dist\PurgeSimulator\PurgeSimData
echo.
echo Built: dist\PurgeSimulator\PurgeSimulator.exe
echo Portable: copy the whole dist\PurgeSimulator folder to any drive; scenarios and
echo settings stay in its PurgeSimData folder.
