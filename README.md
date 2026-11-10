# Bangladesh SPEI drought forecasting: code for Paper 1

Code for the paper

> Ahnaf, M., & Sarkar, S. K. *Evaluation of machine-learning and deep-learning SPEI drought forecasts against persistence across timescales and lead times in Bangladesh.* (Manuscript under preparation.)

- Mahir Ahnaf (ORCID [0009-0007-0992-9560](https://orcid.org/0009-0007-0992-9560)), corresponding author
- Showmitra Kumar Sarkar (ORCID [0000-0001-5928-7577](https://orcid.org/0000-0001-5928-7577))
- Department of Urban and Regional Planning, Khulna University of Engineering & Technology (KUET), Khulna-9203, Bangladesh

Licence: MIT (see `LICENSE`). Version 1.0.

Data (inputs, archived runs, trained models and the tables of the paper): Zenodo, https://doi.org/10.5281/zenodo.24115493 (CC BY 4.0).

## What the code does

- Builds SPEI-1, SPEI-3, SPEI-6 and SPEI-12 for the eight divisions of Bangladesh from ERA5 monthly data, 1980-2024 (FAO-56 Penman-Monteith PET).
- Tests ERA5 precipitation and SPEI against 35 BMD stations, with the station zero-value rule (Text S2 of the paper).
- Screens the predictors (correlation, lead-lag, ACF/PACF, VIF).
- Tunes and trains five models (XGBoost, CNN, Bi-LSTM, CNN-BiLSTM, CEEMDAN-BiLSTM) with five seeds (42-46): 135 runs at leads 1-3 months, test period 2019-2024, plus 5 strict-pipeline runs.
- Applies the model-selection rule fixed in advance, the circular block bootstrap and Diebold-Mariano tests, and SHAP.
- Rebuilds every table and figure of the paper and its Supplementary Information (SI).

## Folder layout

1. Download this repository: the green **Code** button, then **Download ZIP** (or `git clone`).
2. Download `Data.zip` from Zenodo (https://doi.org/10.5281/zenodo.24115493) and unzip it next to the repository folder:

```
any_folder/
    <repository>/  this code (any folder name)
    P1_Data/       inputs, archived runs, trained models and the CSVs behind the paper tables
    P1_work/       created on first use; everything the scripts rebuild goes here
```

- `paths.py` is the only file that holds folder paths, and all of them are relative.
- For a different layout, set the environment variables `P1_DATA` and `P1_WORK`. The forecasting code also reads `P1_OUT_ROOT` (see below).
- `python paths.py` lists every path and whether it exists.

## Quick check: rebuild the tables and figures from the archived runs

This needs no training and no GPU, and takes a few minutes. Run it from inside the repository folder:

```
python 3_predictor_screening/vif_test.py
python 6_evaluation/p1_uncertainty.py --out ../P1_work/tables/uncertainty
python 6_evaluation/p1_paper_tables.py
python 8_figures/make_all.py
python 8_figures/check_figs.py
```

Expected results:

- `vif_test.py` writes the VIF table that `p1_paper_tables.py` uses for Table S7.
- `p1_uncertainty.py` rebuilds the bootstrap intervals and Diebold-Mariano tests from the 135 prediction files. The three CSVs in `../P1_work/tables/uncertainty` are byte-identical to those in `P1_Data/stage2_output/_tables`.
- `p1_paper_tables.py` rebuilds every table CSV into `P1_work/tables`, compares each with `P1_Data/tables`, and ends with "Every rebuilt table equals the archived copy."
- `make_all.py` writes every figure built from code to `P1_work/figures` (PDF, 600-dpi TIFF and PNG), with a check log (`figures_check_log.md`) that compares the plotted values with the tables.
- `check_figs.py` checks page size, resolution, compression and font embedding (`figures_file_check.md`).

Some files are external inputs and are not distributed (see `P1_Data/external/README.md`). Without them:

- the station-derived tables (Tables 1 and 4 and Tables S3-S6) and Figs. S1-S2 are skipped;
- Fig. 9 and Fig. S8 are skipped.

With the BMD station file and a division boundary file placed in `P1_Data/external`, run first:

```
python 2_era5_validation/p1_station_zero_check.py
```

This applies the zero-value rule, prints every corrected station number next to the value in the paper, and writes the Tier-1 tables used by the figures and table builder. It ends with "All values match the paper." With `--no-rule` it reproduces the numbers before the correction.

Figs. 1, 2 and S3 were made outside this code; they are in the paper and its SI.

## Full run order

Each folder depends only on the folders before it. Times are for one CPU (Windows, Python 3.11).

### 1_data_and_index: ERA5, PET and SPEI

| Script | Purpose | Needs | Writes (in `P1_work/1_data_and_index`) |
|---|---|---|---|
| `01_divisional_average.py` | Assigns the 274 ERA5 land cells (0.25°) to the eight divisions and averages the nine variables per division and month (540 months). | raw ERA5 file and division boundary (external) | `divisional_averaged_1980_2024.csv`, `cell_to_division_mapping.csv` |
| `02_compute_pet.py` | FAO-56 Penman-Monteith PET with G = 0.14(T<sub>i</sub> − T<sub>i−1</sub>), pressure from the mean elevation of each division, and the water balance D = P − PET. | output of 01, DEM and division boundary (external) | `divisional_pet_waterbalance_1980_2024.csv`, `division_elevation_summary.csv` |
| `03_compute_spei.py` | SPEI-1/3/6/12. Each calendar month is fitted over 1980-2024 by L-moments with the generalized logistic distribution, the form of the log-logistic used by the reference SPEI package. | the PET file (the rebuilt one, or the archived copy in `P1_Data/inputs`) | `divisional_spei_1980_2024.csv`, fitted parameters, summary statistics |
| `station_zero_rule.py` | The BMD zero-value rule (Text S2), used by the validation and figure code: station-years with a zero annual total, and zero monthly totals in May-September, are set to missing. | | |

The three outputs are archived in `P1_Data/inputs`, so later steps run without 01 and 02. Run on the archived PET file, `03_compute_spei.py` reproduces the archived SPEI to within 1e-14.

### 2_era5_validation: ERA5 against the BMD stations

| Script | Purpose |
|---|---|
| `Validation_1.py` | ERA5 against BMD: precipitation scores, PET sensitivity, SPEI agreement, drought detection and the station seasonal-shift screening (Tables 1, S3-S6; Figs. S1-S2). The zero-value rule is on (`ZERO_RULE = True`). |
| `p1_station_zero_check.py` | Runs `Validation_1.py` and checks every corrected number against the paper (see above). |

Needs the BMD station compilation (external). Writes to `P1_work/2_era5_validation`.

### 3_predictor_screening

| Script | Purpose | Writes (in `P1_work/3_predictor_screening`) |
|---|---|---|
| `correlation.py` | Pearson correlations of the ERA5 drivers, PET and D with SPEI (Fig. 3a). | `correlation/` |
| `leadlag_test.py` | Lead-lag precedence of de-seasonalized drivers against future SPEI (Fig. 3b-d). | `leadlag/` |
| `acf_pacf_test.py` | ACF and PACF of SPEI, used to set the autoregressive lag depth (7, 13 and 13 months for SPEI-3, -6 and -12; Fig. S4). | `acf_pacf/` |
| `vif_test.py` | Variance inflation factors on de-seasonalized anomalies, with the physical and greedy removal (Table S7). | `vif/` |

All four read only `P1_Data/inputs`.

### 4_tuning: hyperparameter tuning (thesis stage)

| File | Purpose |
|---|---|
| `06_build_feature_matrix.py`, `07_data_prep.py` | Feature matrices and train/validation/test arrays for tuning (run these first). |
| `xgboost_optuna.py` | Optuna tuning of XGBoost. |
| `cnn_forecast.py`, `lstm_forecast.py`, `cnn_lstm_forecast.py`, `ceemdan_lstm_forecast.py` | Optuna tuning of the CNN, Bi-LSTM, CNN-BiLSTM and CEEMDAN-BiLSTM. |
| `prove_improvement.py` | Seed-stability check of the tuned Bi-LSTM against its baseline. |
| `best_params_*.csv`, `lag_depth_sensitivity_XGBoost.csv` | The tuned values found. They are written into `5_forecasting/common.py` and listed in Tables S8 and S9. |

Tuning takes several hours per model. Outputs go to `P1_work/4_tuning`. Rerunning the searches can give different trials; the paper uses the archived values.

### 5_forecasting: the 135 runs

Run the `S` scripts in order. Each prints what to expect and can be stopped and restarted; finished runs are skipped.

| Script | Purpose | Time |
|---|---|---|
| `S1_build_arrays.py` | Feature matrices and model arrays for leads 1-3 (calls `run1_build_matrices.py` and `run2_data_prep.py`). Expect "All checks passed." twice. | about 2 min |
| `S2_spei_refit.py` | Refits SPEI on 1980-2018 only and compares it with the main SPEI (Table S18A; calls `p1_spei_refit.py`). | under 1 min |
| `S3_timing_test.py` | Five short runs plus one repeat, to check timing and determinism. | a few minutes |
| `S4_setB_lead1_seeds.py` | Set B: lead 1, five models × three timescales × five seeds (75 runs). | about 1.5 h |
| `S5_sensitivity_F1F2.py` | Strict pipeline, with SPEI fitted and climate anomalies computed on 1980-2018 only, for Bi-LSTM at SPEI-6, five seeds (Table S18B; calls `run11_sensitivity.py`). Writes `<OUT_ROOT>/_strict`. | about 5 min |
| `S6_setC_leads2_3.py` | Set C: leads 2-3 for the selected model of each timescale and XGBoost, five seeds (60 runs). The models are chosen by `p1_selection.py`. | about 25 min |

Shared modules:

- `p1_config.py`: folder paths (from `paths.py`).
- `common.py`: settings, tuned hyperparameters and metric definitions.
- `dl_models.py`: the four deep-learning models.
- `xgb_model.py`: XGBoost.
- `ceemdan_features.py`: the leakage-free CEEMDAN front end.
- `run10_seed_runs.py`: the run driver.
- `p1_selection.py`: the selection rule fixed in advance.

By default the forecasting code reads and writes `P1_Data/stage2_output`. Because finished runs are skipped, running the S scripts there changes nothing. To retrain from scratch, point `P1_OUT_ROOT` to an empty folder, for example `set P1_OUT_ROOT=D:\retrain` (Windows) or `export P1_OUT_ROOT=$HOME/retrain` (Linux/macOS).

Deep-learning runs can differ in the last decimals on other hardware or library versions. The archived runs are the reference for every number in the paper.

### 6_evaluation

| Script | Purpose |
|---|---|
| `p1_uncertainty.py` | Circular block bootstrap (blocks of 12 months, 2,000 resamples, paired across models) and Diebold-Mariano tests with the Harvey-Leybourne-Newbold correction. By default it writes `stage2_bootstrap_ci.csv`, `stage2_bootstrap_diff.csv` and `stage2_dm_tests.csv` to `<OUT_ROOT>/_tables`; use `--out DIR` to write them elsewhere. |
| `p1_paper_tables.py` | Rebuilds every CSV behind the tables of the paper and the SI from the run files, and compares each with `P1_Data/tables`. |

### 7_shap

| Script | Purpose |
|---|---|
| `S7_shap.py` | SHAP (GradientExplainer, 150 training windows as background) for the selected models at lead 1, all five seeds (calls `run12_shap.py`). Run after S4. Writes `<OUT_ROOT>/_shap`. |

### 8_figures

- `make_all.py` builds every figure made from code, and `check_figs.py` checks the files.
- `journal_style.py` sets one style for all figures; `fig_data.py` loads the tables and prediction files; `p1_fig_config.py` holds the paths.

| Figure | Script |
|---|---|
| Fig. 3 | `fig03_screening.py` |
| Fig. 4 | `fig04_lead1_metrics.py` |
| Fig. 5 | `fig05_fidelity.py` |
| Fig. 6 | `fig06_detection.py` |
| Fig. 7 | `fig07_multilead.py` |
| Fig. 8 | `fig08_spei12_seeds.py` |
| Fig. 9 and Fig. S8 | `fig09_maps.py` (needs the division boundary) |
| Fig. 10 | `fig10_shap.py` (with `shap_bars.py` and `figS09_shap_shares.py`) |
| Figs. S1-S2 | `figS01_validation.py` (with `tier1_logic.py`; needs the Tier-1 outputs) |
| Fig. S4 | `figS04_acf.py` |
| Fig. S5 | `figS05_architectures.py` |
| Figs. S6-S7 | `figS06_series.py` |
| Fig. S9 | `figS09_shap_shares.py` |
| Fig. S10 | `figS10_refit.py` |

`python 8_figures/make_all.py --only Fig04,Fig07` builds selected figures.

## Software

The runs in the paper were made on Windows 10 with CPU only. The versions recorded in the run logs are:

- Python 3.11.1
- numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0
- TensorFlow 2.21.0 with Keras 3.15.0
- XGBoost 3.2.0
- EMD-signal (PyEMD) 1.10.0

Also used (versions not recorded in the logs): shap (GradientExplainer), optuna and optuna-integration (tuning), statsmodels (screening), scipy, matplotlib, geopandas and shapely (maps), xarray with netCDF4 (ERA5 file) and rasterio (DEM).

The quick check above was rerun on Linux with Python 3.13.18, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1, matplotlib 3.11.2, geopandas 1.2.0, shapely 2.2.0, Pillow 12.3.0 and pypdf 6.17.0, and gave the archived tables exactly.

Install with `pip install -r requirements.txt`. Use Python 3.11 or later.

## How to cite

Cite the paper and this code; GitHub's **Cite this repository** button gives the details from `CITATION.cff`. The archived release of the code has its own DOI on Zenodo.

## Data sources and licences

| Data | Source | Licence | In the Zenodo data record? |
|---|---|---|---|
| ERA5 monthly averaged data on single levels, 1980-2024 | Copernicus Climate Change Service, Climate Data Store; Hersbach et al. (2020, 2023), https://doi.org/10.24381/cds.f17050d7 | CC BY 4.0 (Copernicus licence) | Divisional series derived from it are in `P1_Data/inputs`; the raw file is not included |
| BMD 35-station compilation | Diptho, Ghosh and Basak (2026), Mendeley Data, Version 1, https://doi.org/10.17632/2YHKDCW64T.1; original records held by the Bangladesh Meteorological Department (BMD) | CC BY 4.0 on Mendeley Data | No (download from Mendeley Data; the original records are available from BMD on request) |
| HydroSHEDS v1 DEM (SRTM-derived, 15 arc-seconds) | Lehner et al. (2008), https://www.hydrosheds.org | HydroSHEDS licence | No (see `P1_Data/external/README.md`) |
| Administrative division boundaries (eight polygons) | | | No (see `P1_Data/external/README.md`) |

The divisional series and every result derived from ERA5 contain modified Copernicus Climate Change Service information (1980-2024). Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

## References

- Diptho, R. A., Ghosh, P., & Basak, S. (2026). *Bangladesh historical rainfall and meteorological dataset for future rainfall prediction* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/2YHKDCW64T.1
- Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society, 146*(730), 1999-2049. https://doi.org/10.1002/qj.3803
- Hersbach, H., Bell, B., Berrisford, P., et al. (2023). *ERA5 monthly averaged data on single levels from 1940 to present* [Data set]. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). https://doi.org/10.24381/cds.f17050d7
- Lehner, B., Verdin, K., & Jarvis, A. (2008). New global hydrography derived from spaceborne elevation data. *Eos, Transactions American Geophysical Union, 89*(10), 93-94. https://doi.org/10.1029/2008EO100001
