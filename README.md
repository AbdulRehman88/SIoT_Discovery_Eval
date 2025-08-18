\# SIoT\_Discovery\_Eval



Reproducible evaluation code for Social Internet of Things (SIoT) \*\*service discovery\*\* experiments.  

It preprocesses several real-world graphs, runs discovery strategies (e.g., greedy with/without fallback), and generates publication-ready plots.



\## Project Structure (key parts)

\- `src/` — all code

\- `config.yaml` — main configuration

\- `data/` — raw inputs (ignored by Git; add your datasets here)

\- `data/processed/` — cached NetworkX graphs (`.gpickle`, ignored)

\- `results/` — figures, traces, CSVs (ignored)

\- `requirements.txt` — Python dependencies

\- `.gitignore` — ignores data/results/venv/artifacts



\## Datasets (place under `data/`)

\- \*\*EIES\*\* (electronic mail interactions)

\- \*\*Epinions\*\* trust network

\- \*\*FB\_Forum\*\* message interactions

\- \*\*Bitcoin Alpha (BTC-Alpha)\*\* trust/ratings

\- \*\*Caenorhabditis (C. elegans) neural\*\* network



Keep the same subfolder names you used locally (e.g., `data/EIES`, `data/Epinions`, etc.).



\## Environment Setup

\- Python 3.10+ recommended

```bash

python -m venv .venv

\# Windows

.venv\\\\Scripts\\\\activate

pip install -r requirements.txt



