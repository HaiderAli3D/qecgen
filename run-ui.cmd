@echo off
rem ---------------------------------------------------------------------------
rem  One-click launcher for the qecgen web UI.
rem
rem  This deliberately runs `python -m qecgen.cli` with the working directory and
rem  PYTHONPATH pinned to the folder this file sits in, rather than calling the
rem  installed `qecgen` console script. If an editable install resolves to a
rem  different checkout -- easy to end up with, and the case this was written for --
rem  the console script serves that tree's code while looking identical in the
rem  terminal, right down to the config table. Pinning the path makes the launcher
rem  serve the checkout it sits in, by construction rather than by luck, and being
rem  self-locating (%~dp0) means it keeps working if the repo moves.
rem ---------------------------------------------------------------------------
setlocal
title qecgen UI

cd /d "%~dp0"
set "PYTHONPATH=%~dp0"
set "QECGEN_URL=http://127.0.0.1:8765"

rem Already serving? Open the page instead of dying on the port. The probe hits
rem the API rather than the socket, so a stranger on 8765 does not read as qecgen.
curl -s -f -m 2 -o nul "%QECGEN_URL%/api/capabilities" >nul 2>&1
if not errorlevel 1 (
    echo qecgen UI is already running.
    echo Opening %QECGEN_URL%
    start "" "%QECGEN_URL%"
    rem `timeout` refuses to run when stdin is redirected and prints an ERROR
    rem line; ping never touches stdin, so this stays quiet from a script too.
    ping -n 3 127.0.0.1 >nul 2>&1
    exit /b 0
)

rem qecgen needs 3.13+, and 3.12 is also installed here, so ask for 3.13 by name.
set "PY=py -3.13"
py -3.13 -c "" >nul 2>&1
if errorlevel 1 set "PY=python"

echo Starting the qecgen UI from "%~dp0"
echo The browser opens by itself once the server is accepting connections.
echo Press Ctrl+C, or close this window, to stop the server.
echo.

%PY% -u -m qecgen.cli ui --data-root data --open
set "RC=%ERRORLEVEL%"

rem A double-clicked window vanishes on exit and takes the traceback with it.
if not "%RC%"=="0" (
    echo.
    echo qecgen exited with code %RC%.
    pause
)
exit /b %RC%
