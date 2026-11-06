@echo off
rem Runs the probe's timing section and its Python TensorFlow twin one after the other, writing timing_cs.txt and
rem timing_py.txt into the data dir. Start it pinned so both processes inherit the affinity (Git Bash does not pin;
rem mask 0x14 = two P-cores on the reference host):
rem   cmd //c "start /b /wait /affinity 14 cmd /c <absolute path of this file> <data dir> <python.exe of the TF venv>"
rem The probe must be built first (dotnet build -c Release in this folder).
setlocal
set HERE=%~dp0
set DATA=%~1
set PY=%~2
set TF_CPP_MIN_LOG_LEVEL=2
"%HERE%bin\Release\net10.0\NumSharp.DotNetRunScript.exe" "%DATA%" timing-only 1000000 > "%DATA%\timing_cs.txt" 2>&1
"%PY%" -I "%HERE%py_timing.py" 1000000 > "%DATA%\timing_py.txt" 2>&1
endlocal
