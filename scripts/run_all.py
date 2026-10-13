#!/usr/bin/env python3
"""Reproduce evaluated models, numerical outputs, figures and LaTeX from included feature tables."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for name in ['run_models.py','make_figures.py','run_temporal_analysis.py','make_temporal_figures.py','write_report.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,check=True)
print('All numerical outputs, figures and LaTeX regenerated. Compile Report/main.tex in Overleaf.')
