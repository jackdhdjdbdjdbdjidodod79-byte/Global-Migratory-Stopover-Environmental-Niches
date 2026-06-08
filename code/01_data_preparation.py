# -*- coding: utf-8 -*-
"""
01_data_preparation.py

Data preparation script for:
Environmental niche differentiation of migratory bird stopover sites
revealed by remote sensing and explainable machine learning

This script reads the final stopover environmental dataset, selects key
environmental variables, checks missing values, standardizes predictors,
and exports cleaned analysis-ready datasets.

Author: Likai Lin, Yan Gui
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths
# ============================================================

# If this script is placed in the "code" folder, the data folder is one level up.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_CSV = os.path.join(DATA_DIR, "stopover_env_all_variables_10km.csv")

OUTPUT_CLEAN = os.path.join(OUTPUT_DIR, "analysis_dataset_clean.csv")
OUTPUT_STANDARDIZED = os.path.join(OUTPUT_DIR, "analysis_dataset_standardized.csv")
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, "data_preparation_summary.txt")


# ============================================================
# 2. Read data
# ============================================================

print("=" * 70)
print("Reading input data")
print("=" * 70)

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_CSV}\n\n"
        "Please place stopover_env_all_variables_10km.csv in the data folder."
    )

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

print(f"Input data shape: {df.shape}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 3. Define variables
# ============================================================

# ID and grouping variables
ID_COLUMNS = [
    "system_index",
    "stopover_id",
    "individual_id",
    "species",
    "common_name_en",
    "functional_group",
    "lat",
    "lon",
    "year",
    "month",
    "season",
    "duration_hours",
    "n_points",
]

# Core environmental variables used in the main analyses
ENV_VARIABLES = [
    # Thermal environment
    "lst_day_mean_c",
    "lst_night_mean_c",

    # Vegetation and anthropogenic indicators
    "ndvi_mean",
    "log_ntl_2023",
    "log_pop_density_2020",

    # Topography
    "elevation_m",
    "slope_deg",

    # Hydrology and wetland indicators
    "water_occurrence",
    "water_perm_frac",
    "water_high_frac",
    "water_wetland_frac",
    "wc_water_frac",
    "wetland_frac",
    "wetland_mangrove_frac",

    # Land-cover composition
    "cropland_frac",
    "forest_frac",
    "grassland_frac",
    "builtup_frac",
    "bare_frac",
    "open_vegetation_frac",
    "shrub_frac",
    "tree_frac",

    # Protected area
    "pa_frac",
]

# Short display names for figures and summaries
VARIABLE_LABELS = {
    "lst_day_mean_c": "Day LST",
    "lst_night_mean_c": "Night LST",
    "ndvi_mean": "NDVI",
    "log_ntl_2023": "NTL",
    "log_pop_density_2020": "Population",
    "elevation_m": "Elevation",
    "slope_deg": "Slope",
    "water_occurrence": "Water occurrence",
    "water_perm_frac": "Permanent water",
    "water_high_frac": "High-frequency water",
    "water_wetland_frac": "Water-wetland",
    "wc_water_frac": "WorldCover water",
    "wetland_frac": "Wetland",
    "wetland_mangrove_frac": "Wetland/mangrove",
    "cropland_frac": "Cropland",
    "forest_frac": "Forest",
    "grassland_frac": "Grassland",
    "builtup_frac": "Built-up",
    "bare_frac": "Bare land",
    "open_vegetation_frac": "Open vegetation",
    "shrub_frac": "Shrubland",
    "tree_frac": "Tree cover",
    "pa_frac": "Protected area",
}


# ============================================================
# 4. Check required columns
# ============================================================

required_columns = ID_COLUMNS + ENV_VARIABLES
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(
        "The following required columns are missing from the input dataset:\n"
        + "\n".join(missing_columns)
    )

print("\nAll required columns found.")


# ============================================================
# 5. Select and clean data
# ============================================================

df_sub = df[required_columns].copy()

# Replace infinite values with NaN
df_sub.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove records without functional group
df_sub = df_sub.dropna(subset=["functional_group"])

# Convert environmental variables to numeric
for col in ENV_VARIABLES:
    df_sub[col] = pd.to_numeric(df_sub[col], errors="coerce")

# For main modelling analyses, remove records with missing values
df_clean = df_sub.dropna(subset=ENV_VARIABLES).copy()

print("\nAfter removing missing environmental values:")
print(f"Clean data shape: {df_clean.shape}")

print("\nFunctional group counts:")
print(df_clean["functional_group"].value_counts())


# ============================================================
# 6. Standardize environmental predictors
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df_clean[ENV_VARIABLES])

df_scaled = df_clean[ID_COLUMNS].copy()

for i, col in enumerate(ENV_VARIABLES):
    df_scaled[col] = X_scaled[:, i]

# Add readable labels as metadata table
label_table = pd.DataFrame({
    "variable": ENV_VARIABLES,
    "label": [VARIABLE_LABELS[v] for v in ENV_VARIABLES],
    "mean_before_standardization": scaler.mean_,
    "scale_before_standardization": scaler.scale_,
})

LABEL_OUTPUT = os.path.join(OUTPUT_DIR, "variable_labels_and_scaling.csv")
label_table.to_csv(LABEL_OUTPUT, index=False, encoding="utf-8-sig")


# ============================================================
# 7. Missing-value summary
# ============================================================

missing_summary = pd.DataFrame({
    "variable": ENV_VARIABLES,
    "missing_count_original": [df_sub[v].isna().sum() for v in ENV_VARIABLES],
    "missing_percent_original": [df_sub[v].isna().mean() * 100 for v in ENV_VARIABLES],
})

MISSING_OUTPUT = os.path.join(OUTPUT_DIR, "missing_value_summary.csv")
missing_summary.to_csv(MISSING_OUTPUT, index=False, encoding="utf-8-sig")


# ============================================================
# 8. Export analysis-ready datasets
# ============================================================

df_clean.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8-sig")
df_scaled.to_csv(OUTPUT_STANDARDIZED, index=False, encoding="utf-8-sig")

print("\nFiles exported:")
print(OUTPUT_CLEAN)
print(OUTPUT_STANDARDIZED)
print(LABEL_OUTPUT)
print(MISSING_OUTPUT)


# ============================================================
# 9. Write summary report
# ============================================================

with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
    f.write("Data preparation summary\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Input file: {INPUT_CSV}\n")
    f.write(f"Original data shape: {df.shape}\n")
    f.write(f"Clean analysis data shape: {df_clean.shape}\n\n")

    f.write("Functional group counts after cleaning:\n")
    f.write(df_clean["functional_group"].value_counts().to_string())
    f.write("\n\n")

    f.write("Environmental variables used:\n")
    for v in ENV_VARIABLES:
        f.write(f"- {v}: {VARIABLE_LABELS[v]}\n")

    f.write("\nMissing value summary in original selected dataset:\n")
    f.write(missing_summary.to_string(index=False))
    f.write("\n\n")

    f.write("Output files:\n")
    f.write(f"- {OUTPUT_CLEAN}\n")
    f.write(f"- {OUTPUT_STANDARDIZED}\n")
    f.write(f"- {LABEL_OUTPUT}\n")
    f.write(f"- {MISSING_OUTPUT}\n")

print("\nSummary report exported:")
print(OUTPUT_SUMMARY)

print("\nData preparation completed successfully.")
print("=" * 70)
