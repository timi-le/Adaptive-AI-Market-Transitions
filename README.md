# Adaptive Participation and Causal Evidence Research
Timilehin Olapade — ATRX Intelligence, Haldane Technologies Inc.

This release contains two completed, evidence-bounded manuscripts and a short non-archival workshop version. The numerical claims concern recorded-response audits and explicitly synthetic experiments.

## Manuscripts
- `papers/market/main.pdf`: Adaptive Participation and Market Feedback: A Deployment Audit and Reproducible Simulation Study.
- `papers/cmtf/main.pdf`: Agentic Causal Macro Intelligence: Evidence Contracts and Regime-Routing Failure under Misspecification.
- `papers/agenthon/main.pdf`: Auditing Financial Decision Systems before Simulating Market Feedback (short version of the market study).

All three include editable LaTeX. The first two include supplementary methodological detail. Their abstracts are available in plain text. Prior proposal wording about established alpha decay, crash boundaries, and full model-population inference has been superseded by the measured results here.

## Run
Use Python 3.12. Create a virtual environment, then run:
```sh
python -m pip install -r requirements.txt
python reproduce.py
python -m unittest discover -s tests
```
`reproduce.py` generates 1,312 market episodes and 800 CMTF diagnostic datasets (4,000 method rows), checks reported market contrasts, and verifies accounting residuals. It does not call a model service or broker. Open `notebooks/Reproduce_Research.ipynb` for a guided run. Install JupyterLab separately if needed to open a browser-based notebook editor.

## Build LaTeX / Overleaf
Upload a paper folder as a ZIP, select `main.tex`, and compile with pdfLaTeX. Locally use `latexmk -pdf main.tex` inside the paper folder. Figures and references are included.

## Interpretation
1. Four exported model labels show descriptive response diversity.
2. Assumed response profiles produce different participation/shock interactions in the simulator.
3. Synthetic regime routing helps under nominal assumptions but can fail under observation misspecification; oracle states do not remove omitted-confounder bias.

See DATA_AVAILABILITY.md for access restrictions. Private logs and credentials are excluded. Public versions use neutral formatting and are not labeled as conference submissions. The author retains responsibility for all claims and submission decisions.
