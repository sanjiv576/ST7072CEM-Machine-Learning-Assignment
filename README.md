# Reframing Cryospheric Forecasting: Cost-Sensitive

XGBoost and Unsupervised Geoclimatic Clustering
for 2030 Glacier Mass Balance Status in the Nepal
Himalayas

This project develops a machine-learning workflow for predicting whether
glaciers in the Nepal Himalayas are likely to experience **mass loss** or
**mass gain**. It combines glacier observations from the World Glacier
Monitoring Service (WGMS) with climate variables from the ERA5 reanalysis
dataset, evaluates several classifiers under a spatial hold-out split, and
provides an interactive Streamlit simulation prototype.

## Project goals

- Integrate WGMS glacier observations with ERA5 climate data at a
  glacier-year level.
- Prepare, clean, and transform physical and meteorological predictors.
- Address severe class imbalance using cost-sensitive learning and SMOTEENN.
- Compare Logistic Regression, SVM, Random Forest, LightGBM, and XGBoost.
- Evaluate spatial generalisation without allowing glacier groups to appear in
  both training and test partitions.
- Export a tuned XGBoost model and supporting preprocessing artifacts.
- Demonstrate a 2030 climate scenario through a Streamlit dashboard.

## Workflow

The notebooks are intended to be run in order:

1. [`1_data_collection_conversion_and_integration.ipynb`](notebooks/1_data_collection_conversion_and_integration.ipynb)
   imports and integrates WGMS and ERA5 data.
2. [`2_data_preparation.ipynb`](notebooks/2_data_preparation.ipynb) cleans
   the modelling matrix and prepares the target and predictors.
3. [`3_eda.ipynb`](notebooks/3_eda.ipynb) explores distributions,
   relationships, spatial patterns, and correlations.
4. [`4_feature_engineering_and_dimensionality_reduction.ipynb`](notebooks/4_feature_engineering_and_dimensionality_reduction.ipynb)
   performs feature selection, scaling, clustering, and PCA.
5. [`5_modeling_and_tuning.ipynb`](notebooks/5_modeling_and_tuning.ipynb)
   benchmarks classifiers and tunes the XGBoost champion model.
6. [`6_model_evaluation.ipynb`](notebooks/6_model_evaluation.ipynb)
   evaluates spatial performance, diagnostics, ROC curves, feature
   importance, and computational complexity.
7. [`7_application_prototype.ipynb`](notebooks/7_application_prototype.ipynb)
   creates the Streamlit application in [`notebooks/app.py`](notebooks/app.py).

## Data

The `final_datasets/` directory contains source-derived, intermediate, and
final modelling data:

- `final_datasets/wgms/` contains Nepal WGMS workbooks and feature metadata.
- `final_datasets/era5/` contains ERA5 NetCDF/CSV data and derived climate
  features.
- `final_datasets/combined_datasets/` contains integrated WGMS-ERA5 tables.
- `final_datasets/clean_datasets/` contains the prepared Parquet/CSV matrix
  and NumPy arrays used by the modelling notebooks.
- `final_datasets/final_used_features_metadata_hardcoded.csv` documents the
  final feature set.

The common integration grain is `glacier_id + year`. Missing values are not
silently treated as zero, and identifiers should not be used as numerical
predictors. See [`docs/1_data_collections_and_conversions.txt`](docs/1_data_collections_and_conversions.txt)
for the data conversion and missingness decisions.

## Model and evaluation

The modelling workflow compares five classifiers and uses class weighting to
make minority-class errors more important. The evaluation notebook uses
`GroupShuffleSplit` with glacier groups, producing a spatially held-out test
set rather than a potentially optimistic random row split.

The recorded spatial-test ROC-AUC results are:

| Classifier             | ROC-AUC |
| ---------------------- | ------: |
| Cost-Sensitive XGBoost |  0.9280 |
| Logistic Regression    |  0.9973 |
| Support Vector Machine |  0.9925 |
| LightGBM               |  0.9740 |
| Random Forest          |  0.9500 |

These values are stored notebook outputs and should be regenerated when the
data, dependencies, or split configuration changes. Evaluation figures are
written to [`plots/`](plots/), including confusion matrix, model comparison,
feature importance, and ROC visualisations.

## Running the project

Create and activate a Python environment, then install the pinned
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Open the notebooks with Jupyter or VS Code and run them in sequence. Notebook
paths are relative to the `notebooks/` directory, so run notebook cells from
their normal location.

To launch the application:

```powershell
cd notebooks
streamlit run app.py
```

The dashboard loads the saved preprocessing and model artifacts from
[`pkl/`](pkl/), applies user-selected temperature and snowfall changes, and
visualises predicted glacier status on a map and elevation plot.

## Repository structure

```text
assignment_code/
├── docs/                 Project decisions and data-processing notes
├── final_datasets/       Source, integrated, clean, and final datasets
├── notebooks/            End-to-end analysis and application notebooks
├── pkl/                  Saved preprocessing and model artifacts
├── plots/                Generated evaluation and EDA figures
└── requirements.txt      Python dependencies
```

## Reproducibility notes

- Most modelling steps use `random_state=42`.
- Spatial evaluation groups observations by glacier location/group.
- The saved `.pkl` files depend on the package versions in
  [`requirements.txt`](requirements.txt).
- The application is a scenario simulator, not a calibrated physical forecast
  of a specific future year. Its outputs depend on the selected perturbations
  and the training data distribution.
