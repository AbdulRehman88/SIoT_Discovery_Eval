# SIoT_Discovery_Eval

This repository provides the complete implementation of the Class-based Greedy Discovery algorithm proposed in our work *“Trust- and Class-Aware Service Discovery with Dual Control in the Social Internet of Things.”*

It includes all source code, configuration files, and evaluation scripts required to reproduce our experimental results. The code processes several publicly available real-world graph datasets, executes service discovery experiments (greedy discovery with and without fallback), and generates figures and logs.

---

## Project Structure
- `src/` — algorithm implementations, preprocessing, and evaluation scripts  
- `config.yaml` — main configuration file controlling experiment parameters  
- `data/` — raw input datasets (not included due to licensing)  
- `data/processed/` — cached NetworkX graphs (`.gpickle`, generated automatically)  
- `results/` — plots, traces, and summary CSVs (ignored in Git)  
- `requirements.txt` — Python dependencies  
- `.gitignore` — specifies ignored files and folders  

---

## Datasets
This project relies on publicly available datasets:

- EIES (electronic mail interactions)  
- Epinions trust network  
- FB Forum message interactions  
- Bitcoin Alpha (BTC-Alpha) trust/ratings network  
- Caenorhabditis (C. elegans) neural network  

The datasets are not redistributed here due to licensing.  
Download them from the Stanford SNAP Repository (https://snap.stanford.edu/data/) and the Network Repository (http://networkrepository.com/).  
After downloading, place them under `data/` using the following subfolder structure:

- `data/EIES/`  
- `data/Epinions/`  
- `data/FB_Forum/`  
- `data/BTCAlpha/`  
- `data/Caenorhabditis/`  

Preprocessing scripts (e.g., `src/process_all_datasets.py`) will generate `.gpickle` graphs in `data/processed/`.

---

## Environment Setup
Python 3.10 or later is recommended.  

Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
