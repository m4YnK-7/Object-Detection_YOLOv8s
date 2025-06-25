@echo off

:: Set the current script directory
set "SCRIPT_DIR=%~dp0"

:: Call the first batch file inside env_setup
echo Running create_env.bat...
call "%SCRIPT_DIR%create_env.bat"

:: Call the second batch file (assumed to be in the same directory as this script)
echo Running install_packages.bat...
call "%SCRIPT_DIR%install_packages.bat"

echo All tasks completed.
pause
