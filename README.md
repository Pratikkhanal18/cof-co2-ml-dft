# cof-co2-ml-dft

Code for the article **"Machine-learning-accelerated screening of experimentally reported covalent organic frameworks for selective CO₂ capture, grounded in molecular simulation and dispersion-corrected DFT"** by Pratik Khanal and Amrit Khanal.

The repository contains all scripts for the workflow:

- CO₂/N₂ Henry selectivities from Widom insertion (RASPA) for the 874 CURATED-COFs frameworks;
- pore-geometry (Zeo++) and chemical descriptors;
- PCA and k-means diversity selection;
- counterpoise-corrected PBE-D3(BJ) cluster DFT (PySCF);
- random-forest, MLP and SchNet models;
- figure and manuscript generation.

**Data:** all results, charged structures and raw simulation outputs are on Zenodo: [10.5281/zenodo.23463092](https://doi.org/10.5281/zenodo.23463092).

## Repository layout

| Path | Contents |
|---|---|
| `cofio.py` | Minimal P1 CIF reader and periodic neighbour list |
| `descriptors.py` | Composition and bonding-pattern descriptors |
| `merge_features.py` | Merges descriptors with Zeo++ and Widom results; quality-control filters (`screening_set`) |
| `pca_cluster.py` | Standardised PCA, k-means scan, selection of 100 diverse frameworks |
| `parse_widom.py` | Parses RASPA Widom output (K_H, ⟨ΔU⟩, Qst⁰, selectivity) |
| `ml_baselines.py` | Random forest and MLP: 5-fold CV and the screening scenario |
| `schnet_np.py`, `gradcheck.py` | SchNet in NumPy with hand-written backpropagation; finite-difference gradient check |
| `build_graphs.py`, `train_schnet.py` | Periodic graphs and SchNet training (`cv <fold>` or `screen <seed>`) |
| `ml_analysis.py` | Screening efficiency, uncertainty and Shapley values; Fig. 4 |
| `analysis_figs.py`, `dft_validation_summary.py` | Figs 1–3, linkage classes, DFT and experimental-validation tables |
| `raspa/` | Force field (`ff/`), Widom/GCMC job generators and batch scripts |
| `eqeq/`, `zeo/` | Batch scripts for EQeq charges and Zeo++ geometry |
| `dft/` | D3(BJ) implementation (`d3bj.py` plus reference data), cluster cutting, DFT pipelines and cluster geometries |
| `manuscript/` | Builds the manuscript `.docx` from the result files (`build_manuscript.py` → `render.js`) |

## Requirements

- Python ≥ 3.10 with the packages in `requirements.txt` (`pip install -r requirements.txt`)
- [RASPA2](https://github.com/iRASPA/RASPA2): the `simulate` binary on `PATH`, with `RASPA_DIR` set
- [Zeo++](https://github.com/lsmo-epfl/zeopp-lsmo): the `network` binary on `PATH`
- [EQeq](https://github.com/numat/EQeq): compiled as `eqeq` on `PATH` (`g++ -O3 main.cpp -o eqeq`)
- Node.js with the `docx` package (only for building the manuscript)

## Reproducing the workflow

Run every command from the repository root. Alternatively, export `COF_ROOT` to point to it.

```bash
# 0. structures
git clone https://github.com/danieleongari/CURATED-COFs data/CURATED-COFs

# 1. charges, pore geometry and descriptors
bash eqeq/run_eqeq.sh                 # writes eqeq/out/<id>.json
bash zeo/run_zeo.sh                   # writes zeo/<id>.res/.sa/.vol
python descriptors.py                 # descriptors_basic.csv

# 2. Widom insertion (CO2, N2, 298 K), then parsing
mkdir -p raspa/widom && tail -n +2 data/CURATED-COFs/cof-frameworks.csv | cut -d, -f1 > raspa/widom/order.txt
bash raspa/run_all.sh
python parse_widom.py                 # widom_results.csv
bash raspa/run_refine.sh <id>         # 6000-cycle refinement of uncertain/top frameworks

# 3. PCA, ML and figures
python pca_cluster.py
python ml_baselines.py
python build_graphs.py 4.0
for k in 0 1 2 3 4; do python train_schnet.py cv $k; done
for s in 0 1 2; do EPOCHS=100 python train_schnet.py screen $s; done
python analysis_figs.py && python ml_analysis.py

# 4. DFT on linkage-motif clusters (PySCF), validation tables
python dft/dft_fast.py imine          # likewise for azine (FULLOPT=1), beta_ketoenamine, triazine, boronate_ester, benzene
python dft_validation_summary.py
```

To skip the simulations, download the processed results from Zenodo and unpack them into `results/`.

## Citation

Please cite the article and the dataset (see `CITATION.cff`):

> Khanal, P. & Khanal, A. Dataset for "Machine-learning-accelerated screening of experimentally reported covalent organic frameworks for selective CO2 capture". Zenodo (2026). https://doi.org/10.5281/zenodo.23463092

Please also cite the CURATED-COFs database (Ongari et al., *ACS Cent. Sci.* 2019, 5, 1663; 2020, 6, 1890), RASPA, Zeo++, EQeq and PySCF.

## Licence

MIT (see `LICENSE`). The D3 reference data in `dft/d3_data.pkl` derive from s-dftd3 (LGPL-3.0-or-later).
