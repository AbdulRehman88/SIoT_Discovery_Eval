@echo off
set PY=python
set ROOT=D:\Dr_Abdul_Rehman\MyPycharm\August 02 2025\SIoT_Discovery_Eval
set TRACES=%ROOT%\results\traces

for /D %%D in ("%TRACES%\*") do (
  echo Plotting %%D
  call %PY% "%ROOT%\src\evaluation\plot_trace.py" --trace_dir "%%D"
)
echo Done. Check each folder for trace_vis.png and trace_vis.pdf
pause
