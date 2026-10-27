# Isolating the contribution of deep neural networks in hybrid scan-based MPPT controllers under partial shading

Code, data and results accompanying the manuscript *Isolating the Contribution of Deep Neural Networks in Hybrid Scan-Based MPPT Controllers Under Partial Shading* (W. Hameed, K. Dimililer), submitted to *The Journal of Supercomputing*.

All numbers, tables and figures of the manuscript are generated from this repository. The study is simulation-only (no hardware-in-the-loop).

Archived release: [10.5281/zenodo.23706406](https://doi.org/10.5281/zenodo.23706406) (v1.0.0).

## Contents

| Path | Content |
|---|---|
| `mppt/` | Controllers (P&O, IncCond, Poly2(G,T), plain ANN-MPPT, PSO, Scan+P&O, DNN scan-only, hybrid DNN+P&O, tree-based variant), plant and boost-converter models, scenarios, schematics |
| `experiments/` | Experiment scripts (main comparison, ablation, statistics/noise/quantization, converter, HPC timing, figures, tables) |
| `tools/` | Report, documentation and traceability builders |
| `tests/` | Consistency and physics tests (pytest) |
| `PVGIS_Ramadi_2005-2020.csv` | Hourly irradiance and temperature, PVGIS-SARAH3, 33.42° N, 43.30° E, 2005–2020 (European Commission JRC) |
| `results/` | All result tables (CSV), `PAPER_NUMBERS.json`, `RESULTS.md`, figures (PDF and PNG) with the plotted data in `results/figures/data/` |
| `docs/` | `FIGURE_MAP.md` (figure to script to data), `METRICS.md`, `PARAMETERS.md`, `FIGURE_CAPTIONS.md` |
| `manuscript/` | The manuscript (`Paper_main.docx`) and its supplementary material (`Supplementary_material.docx`) in the Springer Nature Word template, with the text sources in Markdown (figures appear there as placeholders; the figure files are in `results/figures/`) |
| `logs/` | Logs of the runs that produced `results/` |
| `REBUILD_REPORT_v4.md`, `CHANGELOG_ROUND4.md` | Rebuild report and changelog of the numbers of the final run |
| `archive/` | Earlier rebuild reports, notebook audit and the original notebook (kept for history; one consistency test reads the notebook) |

## Reproduce

```bash
pip install -r requirements.txt        # Python 3.11
python run_all.py --quick              # about 5 minutes, writes to results_quick/
python run_all.py                      # full run, several hours on 4 CPUs
```

The steps run one after the other (the timing experiments need an otherwise idle machine); each step writes its log to `logs/`. The speed-ups and latencies reported in the paper depend on the machine; the other results are deterministic for the fixed seeds.

## Where to find what

- Every figure: `docs/FIGURE_MAP.md`.
- Every number quoted in the paper: `results/PAPER_NUMBERS.json` and `results/RESULTS.md`.

## Data and license

The irradiance and temperature data are from PVGIS-SARAH3 (Huld et al., 2012); see the PVGIS terms of use for the data. The code is released under the MIT license (`LICENSE`).
