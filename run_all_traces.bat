@echo off
set PY=python
set ROOT=D:\Dr_Abdul_Rehman\MyPycharm\August 02 2025\SIoT_Discovery_Eval

call %PY% "%ROOT%\src\evaluation\siot_trace.py" --graph "%ROOT%\data\processed\EIES.gpickle" --dataset EIES --strategy greedy --alpha 0.6 --top_k 6 --max_hops 6 --seed 13 --out_dir "%ROOT%\results\traces\EIES_greedy_a06"
call %PY% "%ROOT%\src\evaluation\plot_trace.py" --trace_dir "%ROOT%\results\traces\EIES_greedy_a06"

call %PY% "%ROOT%\src\evaluation\siot_trace.py" --graph "%ROOT%\data\processed\FB_Forum.gpickle" --dataset FB_Forum --strategy greedy --alpha 0.6 --top_k 6 --max_hops 6 --seed 14 --out_dir "%ROOT%\results\traces\FB_Forum_greedy_a06"
call %PY% "%ROOT%\src\evaluation\plot_trace.py" --trace_dir "%ROOT%\results\traces\FB_Forum_greedy_a06"

call %PY% "%ROOT%\src\evaluation\siot_trace.py" --graph "%ROOT%\data\processed\Caenorhabditis.gpickle" --dataset Caenorhabditis --strategy greedy --alpha 0.6 --top_k 6 --max_hops 6 --seed 15 --out_dir "%ROOT%\results\traces\Caenorhabditis_greedy_a06"
call %PY% "%ROOT%\src\evaluation\plot_trace.py" --trace_dir "%ROOT%\results\traces\Caenorhabditis_greedy_a06"

call %PY% "%ROOT%\src\evaluation\siot_trace.py" --graph "%ROOT%\data\processed\BitcoinAlpha.gpickle" --dataset BitcoinAlpha --strategy greedy --alpha 0.2 --top_k 6 --max_hops 6 --seed 16 --out_dir "%ROOT%\results\traces\BitcoinAlpha_greedy_a02"
call %PY% "%ROOT%\src\evaluation\plot_trace.py" --trace_dir "%ROOT%\results\traces\BitcoinAlpha_greedy_a02"

call %PY% "%ROOT%\src\evaluation\siot_trace.py" --graph "%ROOT%\data\processed\Epinions.gpickle" --dataset Epinions --strategy greedy --alpha 0.7 --top_k 6 --max_hops 6 --seed 21 --out_dir "%ROOT%\results\traces\Epinions_greedy_a07"
call %PY% "%ROOT%\src\evaluation\plot_trace.py" --trace_dir "%ROOT%\results\traces\Epinions_greedy_a07"

call %PY% "%ROOT%\src\evaluation\siot_trace.py" --graph "%ROOT%\data\processed\Epinions.gpickle" --dataset Epinions --strategy fallback --alpha 0.7 --top_k 6 --max_hops 6 --seed 21 --out_dir "%ROOT%\results\traces\Epinions_fallback_a07"
call %PY% "%ROOT%\src\evaluation\plot_trace.py" --trace_dir "%ROOT%\results\traces\Epinions_fallback_a07"

echo Done. Figures saved inside %ROOT%\results\traces\...\trace_vis.png and .pdf
pause
