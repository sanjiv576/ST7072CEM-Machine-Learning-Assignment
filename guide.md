# Predicting Glacier Mass Loss in the Nepal Himalayas

## 1. Purpose and scientific limitation

This guide describes a defensible machine-learning workflow for predicting glacier change using the data collected in this project.

The current files include:

- RGI glacier inventory attributes for South Asia
- A Nepal-filtered glacier inventory
- A 2019 Everest airborne LiDAR survey and a converted LiDAR CSV tile

These data are useful for describing glacier geometry and terrain, but they do **not** contain a time series of glacier mass balance. The LiDAR survey is a spatial snapshot from 2019. It cannot, by itself, show whether a glacier gained or lost mass because mass loss requires observations from at least two dates.

Therefore:

> Do not train a model with the current inventory/LiDAR data and call its output “predicted mass loss” unless an observed mass-loss or elevation-change target has been added.

The recommended project is:

> Predict annual glacier elevation change or annual surface mass balance for selected Nepal Himalayan glaciers using repeated elevation/area observations, climate data, glacier attributes, and terrain features. Use the 2019 LiDAR data as a high-resolution spatial reference or validation dataset.

## 2. What the current data contain

### 2.1 RGI glacier attributes

The glacier attribute file contains variables such as:

```text
rgi_id
cenlon, cenlat
area_km2
zmin_m, zmax_m, zmed_m, zmean_m
slope_deg, aspect_deg
lmax_m
term_type
conn_lvl
surge_type
src_date
dem_source
```

The notebook also combines several RGI tables into
`combined_data/rgi_combined_long_format.csv`. That file is a row-wise outer
concatenation of different tables, so many columns are structurally missing in
each row. It should not be passed directly to a model.

Use the glacier attribute table as the main table and join other tables with
`rgi_id`.

### 2.2 LiDAR data

The LiDAR data provide very detailed surface elevation for the survey date.
The CSV contains raw integer coordinates. Coordinates must be read using the
scale, offset, and CRS in the corresponding LAZ header. Do not interpret raw
`X`, `Y`, and `Z` values as metre coordinates.

The LiDAR data can support:

- A high-resolution DEM or DSM
- Slope, aspect, curvature, and roughness
- Glacier-surface elevation
- Glacier and moraine geometry
- Validation of coarser satellite DEMs

The LiDAR data alone cannot support:

- Annual mass-loss estimation
- Future melt prediction
- Climate trend learning
- GLOF occurrence prediction

## 3. Choose one target variable

Choose the target before selecting a model. The target must be observed, not
derived from the same predictors in a way that causes leakage.

### Recommended target A: annual elevation change

Given DEMs at dates \(t_1\) and \(t_2\):

```text
dh = DEM(t2) - DEM(t1)
annual_dh = dh / (t2 - t1)
```

Negative `annual_dh` indicates surface lowering. A useful target table has one
row per glacier and year, or one row per glacier grid cell and year:

```text
rgi_id, year, annual_dh_m, target_uncertainty_m
```

### Recommended target B: annual glacier area change

From glacier outlines:

```text
annual_area_change_km2 = area_t2_km2 - area_t1_km2
```

This is easier to construct than mass balance, but it measures horizontal
retreat/area change rather than total ice mass change.

### Target C: annual surface mass balance

If observations are available, use:

```text
annual_mass_balance_mwe
```

where `mwe` means metres water equivalent. Negative values indicate mass loss.
Observed mass-balance measurements are preferable to treating a simple DEM
difference as mass balance.

## 4. Additional data required

Create a multi-temporal target dataset by adding data from several dates.
Possible sources include:

- Satellite DEMs or ICESat-2 elevation observations
- Sentinel-2 or Landsat glacier outlines
- Historical glacier inventories
- Glacier mass-balance observations
- Climate reanalysis or station data
- Glacier velocity from optical or SAR imagery
- Debris-cover and snow-cover indicators

For each glacier-year, useful predictor variables include:

```text
area_km2
zmin_m, zmax_m, zmed_m, zmean_m
slope_deg, aspect_deg
lmax_m
mean_summer_temperature
mean_annual_temperature
summer_precipitation
winter_snowfall
shortwave_radiation
glacier_velocity
debris_fraction
previous_year_area_km2
previous_year_elevation_change_m
```

A suitable modelling table looks like:

| rgi_id | year | area_km2 | zmean_m | slope_deg | summer_temperature | summer_precipitation | velocity_m_year | annual_dh_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RGI...001 | 2018 | 2.31 | 5220 | 18.5 | 2.1 | 610 | 12.4 | -0.42 |
| RGI...001 | 2019 | 2.27 | 5214 | 18.5 | 2.4 | 580 | 11.8 | -0.37 |

Save this table as:

```text
combined_data/glacier_training_data.csv
```

## 5. Prepare the glacier feature table

Use the RGI glacier-attribute file rather than the long-format concatenation.

```python
from pathlib import Path
import pandas as pd

project_dir = Path(".")
attribute_file = (
    project_dir
    / "rgi_data"
    / "RGI2000-v7.0-G-15_south_asia_east-attributes.csv"
)

glacier = pd.read_csv(attribute_file)
glacier["src_date"] = pd.to_datetime(glacier["src_date"], errors="coerce")

feature_columns = [
    "rgi_id",
    "cenlon",
    "cenlat",
    "area_km2",
    "zmin_m",
    "zmax_m",
    "zmed_m",
    "zmean_m",
    "slope_deg",
    "aspect_deg",
    "lmax_m",
    "term_type",
    "conn_lvl",
    "surge_type",
]

features = glacier[feature_columns].copy()
```

Do not use identifiers, row numbers, or filtering-boundary columns as
predictors:

```text
rgi_id
glac_name
combined_row_id
source_row_number
filter_min_latitude
filter_max_latitude
filter_min_longitude
filter_max_longitude
```

`rgi_id` is required for joining and grouping, but should not be passed to the
regression model as a numeric feature.

## 6. Join observations and create lagged predictors

Assume `glacier_training_data.csv` contains one row per glacier and year, with
the target column `annual_dh_m`.

```python
from pathlib import Path
import pandas as pd

project_dir = Path(".")

features = pd.read_csv(
    project_dir
    / "rgi_data"
    / "RGI2000-v7.0-G-15_south_asia_east-attributes.csv"
)

observations = pd.read_csv(
    project_dir / "combined_data" / "glacier_training_data.csv"
)

observations["year"] = pd.to_numeric(observations["year"], errors="raise")

data = observations.merge(
    features,
    on="rgi_id",
    how="inner",
    validate="many_to_one",
)

data = data.sort_values(["rgi_id", "year"]).reset_index(drop=True)

data["previous_area_km2"] = (
    data.groupby("rgi_id")["area_km2"].shift(1)
)

data["previous_annual_dh_m"] = (
    data.groupby("rgi_id")["annual_dh_m"].shift(1)
)

# A lagged target may be used as a predictor only when it is known before
# the prediction year. Never use the current-year target as an input.
data = data.dropna(
    subset=["annual_dh_m", "previous_area_km2"]
).copy()
```

If the target is annual mass balance, replace `annual_dh_m` with
`annual_mass_balance_mwe`.

## 7. Convert aspect into continuous features

Aspect is circular: 0 degrees and 360 degrees represent the same direction.
Using raw aspect as a linear variable creates an artificial discontinuity.

```python
import numpy as np

aspect_radians = np.deg2rad(data["aspect_deg"])
data["aspect_sin"] = np.sin(aspect_radians)
data["aspect_cos"] = np.cos(aspect_radians)
```

Use `aspect_sin` and `aspect_cos` instead of `aspect_deg`.

## 8. Baseline machine-learning model

Start with a regularized linear model. It is interpretable and gives a useful
baseline before trying more complex algorithms.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

target_column = "annual_dh_m"

numeric_features = [
    "cenlon",
    "cenlat",
    "area_km2",
    "zmin_m",
    "zmax_m",
    "zmed_m",
    "zmean_m",
    "slope_deg",
    "lmax_m",
    "aspect_sin",
    "aspect_cos",
    "previous_area_km2",
    "previous_annual_dh_m",
    "mean_summer_temperature",
    "mean_annual_temperature",
    "summer_precipitation",
    "winter_snowfall",
    "glacier_velocity",
]

categorical_features = [
    "term_type",
    "conn_lvl",
    "surge_type",
]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0)),
    ]
)
```

The columns listed in `numeric_features` must exist in the training table.
Remove a variable from the list or add the corresponding data before fitting.

## 9. Leakage-safe validation

Do not randomly split individual LiDAR points. Nearby points are spatially
correlated, and point-level random splitting produces overly optimistic scores.

For glacier-year records, split by glacier so that the same glacier is not
present in both training and testing:

```python
from sklearn.model_selection import GroupShuffleSplit

required_columns = (
    numeric_features
    + categorical_features
    + [target_column, "rgi_id"]
)

missing_columns = sorted(set(required_columns) - set(data.columns))
if missing_columns:
    raise ValueError(f"Missing modelling columns: {missing_columns}")

X = data[numeric_features + categorical_features]
y = data[target_column]
groups = data["rgi_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_index, test_index = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]
y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

model.fit(X_train, y_train)
prediction = model.predict(X_test)

mae = mean_absolute_error(y_test, prediction)
rmse = mean_squared_error(y_test, prediction) ** 0.5
r2 = r2_score(y_test, prediction)

print(f"MAE:  {mae:.3f} m/year")
print(f"RMSE: {rmse:.3f} m/year")
print(f"R2:   {r2:.3f}")
```

For a forecasting study, use a time-based split instead:

```python
train = data[data["year"] <= 2018].copy()
test = data[data["year"] >= 2019].copy()
```

Do not use future climate or future glacier observations when predicting an
earlier year.

## 10. Compare tree-based models

After the baseline, compare a random forest or gradient-boosted model.

```python
from sklearn.ensemble import RandomForestRegressor

random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

random_forest.fit(X_train, y_train)
rf_prediction = random_forest.predict(X_test)

print(
    "Random forest MAE:",
    mean_absolute_error(y_test, rf_prediction),
)
print(
    "Random forest RMSE:",
    mean_squared_error(y_test, rf_prediction) ** 0.5,
)
```

Use the same train/test split for every model. Select the model using the
validation error and scientific interpretability, not only the best training
score.

## 11. Calculate an elevation-change target from DEMs

For two co-registered DEM rasters, calculate:

```text
annual_dh = (dem_t2 - dem_t1) / elapsed_years
```

Example using `rasterio`:

```python
import numpy as np
import rasterio

with rasterio.open("dem_2019.tif") as first_dem:
    dem_2019 = first_dem.read(1).astype("float32")
    profile = first_dem.profile
    nodata_2019 = first_dem.nodata

with rasterio.open("dem_2023.tif") as second_dem:
    dem_2023 = second_dem.read(1).astype("float32")
    nodata_2023 = second_dem.nodata

if dem_2019.shape != dem_2023.shape:
    raise ValueError("The DEMs must be resampled to the same grid first.")

valid = np.isfinite(dem_2019) & np.isfinite(dem_2023)

if nodata_2019 is not None:
    valid &= dem_2019 != nodata_2019
if nodata_2023 is not None:
    valid &= dem_2023 != nodata_2023

elapsed_years = 2023 - 2019
annual_dh = np.full(dem_2019.shape, np.nan, dtype="float32")
annual_dh[valid] = (dem_2023[valid] - dem_2019[valid]) / elapsed_years

profile.update(dtype="float32", count=1, nodata=np.nan)

with rasterio.open("annual_dh_2019_2023.tif", "w", **profile) as output:
    output.write(annual_dh, 1)
```

Before this calculation, the DEMs must have:

- The same CRS
- The same horizontal grid and resolution
- The same vertical datum
- Consistent co-registration
- Comparable snow and surface conditions

DEM registration error, seasonal snow, interpolation, and vegetation/debris
effects must be assessed and reported.

## 12. Convert elevation change to mass change cautiously

Elevation change is not automatically mass change. A simplified conversion is:

\[
\Delta M = \Delta V \rho
\]

where:

- \(\Delta V\) is volume change in cubic metres
- \(\rho\) is an assumed effective snow/firn/ice density
- \(\Delta M\) is mass change in kilograms

For grid cells:

```python
volume_change_m3 = annual_dh * pixel_area_m2 * glacier_mask
mass_change_kg = volume_change_m3 * effective_density_kg_m3
```

For water-equivalent balance:

\[
b_{\mathrm{w.e.}} =
\frac{\Delta V \rho}{A \rho_{\mathrm{water}}}
\]

Use density appropriate to the surface type where possible. Report the density
assumption and uncertainty. Do not present this conversion as a direct
measurement if density and snow/firn information are unavailable.

## 13. Uncertainty and reporting

Report more than one accuracy score. At minimum include:

```text
MAE
RMSE
bias = mean(prediction - observation)
R²
number of glaciers in the test set
prediction interval or uncertainty estimate
```

For DEM-derived targets, include uncertainty from:

- Vertical DEM error
- Co-registration error
- Snow and firn variability
- Effective density
- Glacier-mask uncertainty
- Temporal mismatch between observations

A model with a low average error but large uncertainty on small or high-altitude
glaciers should not be described as universally reliable.

## 14. Recommended project sequence

1. Confirm the CRS, scale, offset, and acquisition metadata for every LAZ file.
2. Identify the LAZ file from which the converted CSV was created.
3. Build a documented LiDAR DEM or DSM.
4. Obtain repeated DEM, outline, or mass-balance observations.
5. Add climate variables for the same glacier-years.
6. Construct one row per glacier-year or glacier-grid-cell-year.
7. Define one target: `annual_dh_m`, `annual_area_change_km2`, or
   `annual_mass_balance_mwe`.
8. Train a Ridge baseline.
9. Compare a tree-based model using the same split.
10. Validate by withheld glaciers or future years.
11. Quantify uncertainty and inspect errors by glacier size, elevation, and
    region.
12. Use the 2019 LiDAR dataset as a reference/validation source rather than
    claiming that it alone measures future mass loss.

## 15. Final interpretation

The current dataset can support terrain and glacier-morphology analysis. It
cannot by itself support a valid supervised model of future glacier mass loss.
The core requirement is a reliable multi-temporal target dataset:

```text
2019 LiDAR terrain baseline
+ repeated DEM or glacier-outline observations
+ climate and meteorological time series
+ glacier attributes and velocity/debris indicators
+ observed elevation change or mass balance
+ leakage-safe spatial/temporal validation
```

Once those data are available, the code in this guide provides a reproducible
starting point for modelling and evaluation.

## 16. Remaining steps

Use this checklist to track progress. Tick a box only when the step has been
completed and the output has been saved or documented. Record the completion
date and any notes in your project log.

### Phase 1: Confirm and organise the current data

- [ ] **1. Confirm the project scope.** Decide whether the primary target is
  `annual_dh_m`, `annual_area_change_km2`, or
  `annual_mass_balance_mwe`. Recommended first target:
  `annual_dh_m`.
- [ ] **2. Inventory all files.** Record the file name, source, date, spatial
  coverage, CRS, resolution, and file format for every LAZ, CSV, raster, and
  vector file.
- [ ] **3. Verify every LAZ header.** Record the scale, offset, CRS,
  acquisition date, point count, and bounding box. Do not rely on the example
  scale and offset in this guide.
- [ ] **4. Identify the source LAZ file for `nat_geo_1.csv`.** Store this
  relationship in a data-provenance note.
- [ ] **5. Inspect the RGI tables.** Confirm that `rgi_id` is unique in the
  glacier attribute table and use that table as the main glacier table.
- [ ] **6. Create a clean project folder structure.** At minimum, create
  `raw_data`, `processed_data`, `targets`, `features`, `models`, `figures`,
  and `reports`. Never overwrite raw data.

### Phase 2: Prepare the LiDAR reference data

- [ ] **7. Install and test a LAZ-aware reader.** Confirm that it reads scaled
  coordinates and preserves the CRS.
- [ ] **8. Remove invalid or withheld points.** Record the filtering rules and
  the number of points removed.
- [ ] **9. Classify the point cloud.** Separate ground, ice, snow, rock,
  debris, and water where the data and method allow. Remember that the
  converted CSV currently has unclassified points.
- [ ] **10. Generate a documented LiDAR DEM or DSM.** Record the grid
  resolution, interpolation method, point filters, vertical datum, and output
  path.
- [ ] **11. Derive terrain features.** Generate elevation, slope, aspect,
  curvature, and roughness rasters or glacier-level summaries.
- [ ] **12. Check the LiDAR output visually.** Plot the point cloud and DEM,
  inspect gaps and artefacts, and save quality-control figures.

### Phase 3: Obtain multi-temporal observations

- [ ] **13. Select the observation period.** Choose a time range with enough
  repeated observations for training and testing, preferably several years.
- [ ] **14. Obtain a second elevation observation.** Use a satellite DEM,
  ICESat-2, or another compatible elevation source for at least one
  comparison date.
- [ ] **15. Obtain additional dates.** More dates are required for a trend or
  forecasting model; two dates only provide one change estimate.
- [ ] **16. Obtain repeated glacier outlines.** Use Sentinel-2, Landsat, or a
  suitable inventory to calculate annual area changes if area loss is also
  studied.
- [ ] **17. Obtain climate data.** Collect temperature, precipitation or
  snowfall, and radiation variables for the same glacier-years.
- [ ] **18. Obtain optional explanatory data.** Add velocity, debris cover,
  snow cover, or field mass-balance observations if available.
- [ ] **19. Document data compatibility.** Check CRS, vertical datum,
  resolution, acquisition season, cloud/snow conditions, and spatial overlap
  across all sources.

### Phase 4: Construct the modelling dataset

- [ ] **20. Co-register and resample DEMs.** Put all DEMs on the same grid,
  CRS, vertical datum, and resolution before differencing.
- [ ] **21. Calculate elevation differences.** Generate
  `annual_dh_m` and a corresponding uncertainty estimate for each glacier or
  glacier grid cell.
- [ ] **22. Create glacier masks.** Ensure that each elevation-change
  calculation uses the correct glacier boundary for the observation date.
- [ ] **23. Convert DEM change cautiously if required.** Calculate volume or
  water-equivalent mass balance only after documenting density assumptions.
- [ ] **24. Aggregate climate data.** Create annual and seasonal climate
  summaries matching each glacier and observation year.
- [ ] **25. Join all sources using `rgi_id` and year.** Validate that the join
  does not duplicate glacier-year records unexpectedly.
- [ ] **26. Create lagged predictors.** Add previous-year area, previous-year
  elevation change, and other variables that would have been known before the
  prediction date.
- [ ] **27. Create the final CSV.** Save one row per glacier-year or
  glacier-grid-cell-year as
  `combined_data/glacier_training_data.csv`.
- [ ] **28. Run a data-quality audit.** Check missing values, duplicate keys,
  impossible ranges, outliers, target distribution, and the number of unique
  glaciers and years.

### Phase 5: Train and evaluate models

- [ ] **29. Define the feature list and target in code.** Exclude identifiers,
  row numbers, future observations, and filtering-boundary columns.
- [ ] **30. Train the Ridge baseline.** Save the fitted pipeline, feature
  list, data version, and random seed.
- [ ] **31. Train a tree-based baseline.** Compare a random forest or gradient
  boosting model using exactly the same target and split.
- [ ] **32. Use leakage-safe validation.** Use withheld glaciers for spatial
  generalisation and earlier years for forecasting. Do not use a point-level
  random split.
- [ ] **33. Calculate evaluation metrics.** Report MAE, RMSE, bias, R², and
  the number of glaciers and years in each split.
- [ ] **34. Evaluate uncertainty.** Add prediction intervals or an uncertainty
  estimate and include DEM, density, mask, and observation uncertainties.
- [ ] **35. Inspect errors by subgroup.** Compare results by glacier area,
  elevation, slope, region, and observation year.
- [ ] **36. Perform sensitivity checks.** Test reasonable changes to DEM
  resolution, density assumptions, feature sets, and model hyperparameters.
- [ ] **37. Select the final model.** Choose based on honest test performance,
  uncertainty, interpretability, and scientific plausibility rather than
  training score alone.

### Phase 6: Produce and preserve the final results

- [ ] **38. Refit only after model selection.** Fit the selected pipeline on
  the approved training data; keep the test set untouched until final
  reporting.
- [ ] **39. Generate predictions.** Save glacier ID, prediction year,
  observed target where available, prediction, and uncertainty.
- [ ] **40. Create figures.** Include maps, observed-versus-predicted plots,
  residual plots, time trends, and feature-effect plots where appropriate.
- [ ] **41. Write the limitations.** State clearly that the 2019 LiDAR
  campaign is a baseline/reference and not a standalone mass-loss time
  series.
- [ ] **42. Make the experiment reproducible.** Save code, package versions,
  input-file manifest, configuration, random seeds, model files, metrics, and
  figures.
- [ ] **43. Review the scientific conclusion.** Ensure that the conclusion
  matches the target actually measured. Do not claim mass loss when the target
  is only area change or surface elevation change.
- [ ] **44. Freeze the final dataset and report.** Archive the exact data
  version used for the final model and record the completion date.

### Progress summary

At each review session, record:

```text
Review date:
Last completed step:
Current step:
Next action:
Files or outputs created:
Blocking issue:
```
