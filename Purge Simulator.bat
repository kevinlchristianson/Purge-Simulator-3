@echo off
rem Start Purge Simulator from wherever this folder is (an external drive works).
rem Scenarios, reports and settings are kept in the PurgeSimData folder next to this file,
rem so they move with the drive. Needs Python 3.10+ on this computer.
cd /d "%~dp0"
where python >NUL 2>NUL || (echo Python was not found on this computer. Install Python 3.10+ or use the PurgeSimulator.exe build. & pause & exit /b 1)
python app.py --portable
if errorlevel 1 pause
