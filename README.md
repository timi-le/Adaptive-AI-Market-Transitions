# Adaptive Participation and Market Feedback

**A Deployment Audit and Reproducible Simulation Study**  
Timilehin Olapade · ATRX Intelligence / Haldane Technologies Inc.

This repository contains the market-feedback research study and its short workshop derivative. The separate CMTF evidence-routing paper is not part of this repository's current tree.

## Read the work
- [Market manuscript](papers/market/main.pdf) · [LaTeX source](papers/market/main.tex)
- [Short workshop manuscript](papers/agenthon/main.pdf) · [LaTeX source](papers/agenthon/main.tex)
- [Evidence and disclosure boundaries](DATA_AVAILABILITY.md)

## What the study establishes
- An operational audit reports 28,165 events and 229 entry-to-execution-response pairs; missing portfolio responses prevent full two-role replay.
- Directional agreement across four exported model labels ranges from 52.2% to 80.7% on 5,998 common inputs. This is an export audit, not a fresh model benchmark.
- 1,312 synthetic episodes examine participation, behavioral profiles and shocks. The sign of the participation–shock interaction depends on the assumed profile.

These results do not establish live alpha, calibrated crash thresholds, endogenous liquidity withdrawal, or actual inference from six alternative models. The simulator is not an ABIDES matching engine or an Agenthon competition submission.

## Reproduce
Use Python 3.12 in a fresh virtual environment:

```bash
python -m pip install -r requirements.txt
python reproduce.py
python -m unittest discover -s tests
```

The market-only runner regenerates the 1,312 episodes, compares reported contrasts with archived values, and updates figures. Eight market contract tests are supplied. [Jupyter notebook](notebooks/Reproduce.ipynb) runs the same commands. No broker connection, paid inference API, GPU or private logs are required for the synthetic study.

## Repository map
| Directory | Purpose |
|---|---|
| `papers/market` | Main manuscript, bibliography and figures |
| `papers/agenthon` | Short non-archival workshop derivative |
| `market_lab` | Research simulator and log utilities |
| `configs` | Declared experimental settings |
| `scripts` | Experiments and figure generation |
| `results` | Synthetic outputs and aggregate audits |
| `audit_scripts` | Restricted-input audit procedures |
| `tests` | Contract checks |
| `notebooks` | Guided reproduction |

## Scope and versioning
Research artifacts are distinct from the proprietary production system. Raw operational evidence is excluded; its audit cannot be independently regenerated from this repository alone. See the manuscript AI Use Statement for the division between author-originated work and assistance.

The historical combined revision remains in Git history. Separation of the current tree is organizational; it does not erase previously public files. The companion CMTF repository is pending creation; no remote URL is claimed here.

No venue-specific submission status is asserted by this public preprint. Do not use this named repository as a link in an anonymous submission. No software license is granted merely by public availability.
