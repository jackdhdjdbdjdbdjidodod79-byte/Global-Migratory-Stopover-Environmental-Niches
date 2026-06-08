# -*- coding: utf-8 -*-
"""
03_random_forest_shap_analysis.py

Random Forest and SHAP analysis for migratory bird stopover environments.

Input:
    outputs/analysis_dataset_clean.csv

Outputs:
    outputs/rf_shap/
        rf_model_performance.txt
        rf_permutation_importance.csv
        shap_mean_importance.csv
        shap_values_for_plotting.csv
        Fig_RF_importance.png
        Fig_SHAP_summary.png
        Fig_SHAP_dependence_*.png
        Fig_SHAP_interaction_surface.png

Author: Likai Lin, Yan Gui
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

try:
    import shap
except ImportError:
    raise ImportError("Please install SHAP first: pip install shap")


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RF_DIR = os.path.join(OUTPUT_DIR, "rf_shap")

os.makedirs(RF_DIR, exist_ok=True)

INPUT_CLEAN = os.path.join(OUTPUT_DIR, "analysis_dataset_clean.csv")


# ============================================================
# 2. Variables
# ============================================================

FEATURES = [
    "log_ntl_2023",
    "cropland_frac",
    "elevation_m",
    "lst_night_mean_c",
    "log_pop_density_2020",
    "shrub_frac",
    "bare_frac",
    "lst_day_mean_c",
    "ndvi_mean",
    "pa_frac",
    "grassland_frac",
    "wetland_mangrove_frac",
    "forest_frac",
    "tree_frac",
    "wetland_frac",
    "water_wetland_frac",
    "wc_water_frac",
    "water_occurrence",
]

FEATURE_LABELS = {
    "log_ntl_2023": "NTL",
    "cropland_frac": "Cropland",
    "elevation_m": "Elevation",
    "lst_night_mean_c": "Night LST",
    "log_pop_density_2020": "Population",
    "shrub_frac": "Shrubland",
    "bare_frac": "Bare land",
    "lst_day_mean_c": "Day LST",
    "ndvi_mean": "NDVI",
    "pa_frac": "Protected area",
    "grassland_frac": "Grassland",
    "wetland_mangrove_frac": "Wetland/mangrove",
    "forest_frac": "Forest",
    "tree_frac": "Tree cover",
    "wetland_frac": "Wetland",
    "water_wetland_frac": "Water-wetland",
    "wc_water_frac": "WorldCover water",
    "water_occurrence": "Water occurrence",
}

GROUP_ORDER = ["Waterbird", "Raptor", "Shorebird", "Seabird"]


# ============================================================
# 3. Helper functions
# ============================================================

def set_publication_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.dpi"] = 300


def get_multiclass_shap_matrix(shap_values, target_class=None):
    """
    Convert SHAP output into a 2D matrix for plotting.

    For multiclass classification:
    - shap.TreeExplainer may return a list of arrays.
    - New SHAP versions may return a 3D array.
    This function averages absolute SHAP values across classes unless
    a target_class index is provided.
    """
    if isinstance(shap_values, list):
        if target_class is None:
            return np.mean(np.abs(np.stack(shap_values, axis=2)), axis=2)
        return shap_values[target_class]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        if target_class is None:
            return np.mean(np.abs(shap_values), axis=2)
        return shap_values[:, :, target_class]

    return shap_values


def smooth_line(x, y, bins=60):
    """Simple binned smoothing for dependence plots."""
    x = np.asarray(x)
    y = np.asarray(y)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < bins:
        bins = max(10, len(x) // 5)

    q = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(x, q))

    xs, ys = [], []
    for i in range(len(edges) - 1):
        mask = (x >= edges[i]) & (x <= edges[i + 1])
        if mask.sum() >= 5:
            xs.append(np.nanmedian(x[mask]))
            ys.append(np.nanmedian(y[mask]))

    return np.array(xs), np.array(ys)


def savefig(path):
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 4. Read data
# ============================================================

set_publication_style()

if not os.path.exists(INPUT_CLEAN):
    raise FileNotFoundError(
        f"{INPUT_CLEAN} not found. Please run 01_data_preparation.py first."
    )

df = pd.read_csv(INPUT_CLEAN, encoding="utf-8-sig")

df = df.dropna(subset=FEATURES + ["functional_group"]).copy()

# Keep only known groups
df = df[df["functional_group"].isin(GROUP_ORDER)].copy()

print("=" * 70)
print("Random Forest + SHAP analysis")
print("=" * 70)
print(f"Data shape: {df.shape}")
print(df["functional_group"].value_counts())


# ============================================================
# 5. Prepare X and y
# ============================================================

X = df[FEATURES].copy()
y_raw = df["functional_group"].copy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

class_names = list(label_encoder.classes_)

print("\nClass encoding:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# 6. Train Random Forest
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

print(f"\nTest accuracy: {acc:.3f}")
print(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


# ============================================================
# 7. Export model performance
# ============================================================

performance_txt = os.path.join(RF_DIR, "rf_model_performance.txt")

with open(performance_txt, "w", encoding="utf-8") as f:
    f.write("Random Forest model performance\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Sample size: {df.shape[0]}\n")
    f.write(f"Number of predictors: {len(FEATURES)}\n\n")
    f.write(f"Test accuracy: {acc:.4f}\n")
    f.write(f"Five-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
    f.write("Class encoding:\n")
    for i, name in enumerate(class_names):
        f.write(f"{i}: {name}\n")
    f.write("\nClassification report:\n")
    f.write(classification_report(y_test, y_pred, target_names=class_names))
    f.write("\nConfusion matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))

print(f"\nModel performance exported: {performance_txt}")


# ============================================================
# 8. Permutation importance
# ============================================================

print("\nCalculating permutation importance...")

perm = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=30,
    random_state=42,
    n_jobs=-1,
    scoring="accuracy"
)

importance_df = pd.DataFrame({
    "variable": FEATURES,
    "label": [FEATURE_LABELS[v] for v in FEATURES],
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)

importance_csv = os.path.join(RF_DIR, "rf_permutation_importance.csv")
importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")

print(f"Permutation importance exported: {importance_csv}")


# ============================================================
# 9. Plot RF importance
# ============================================================

top_n = 15
plot_df = importance_df.head(top_n).iloc[::-1]

plt.figure(figsize=(7.5, 5.2))
plt.barh(
    plot_df["label"],
    plot_df["importance_mean"],
    xerr=plot_df["importance_std"],
    edgecolor="black",
    linewidth=0.6,
    alpha=0.75
)
plt.xlabel("Permutation importance")
plt.title("(a) Comprehensive environmental drivers\nRF accuracy = %.2f" % acc, loc="left", fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.25)
savefig(os.path.join(RF_DIR, "Fig_RF_importance.png"))


# ============================================================
# 10. SHAP analysis
# ============================================================

print("\nCalculating SHAP values...")

# Use a subset for SHAP if the dataset is very large.
# This keeps the public script computationally efficient.
max_shap_n = 2500

if X.shape[0] > max_shap_n:
    shap_sample = X.sample(max_shap_n, random_state=42)
else:
    shap_sample = X.copy()

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(shap_sample)

# Mean absolute SHAP across classes for multiclass RF
shap_matrix_abs = get_multiclass_shap_matrix(shap_values, target_class=None)

mean_abs_shap = np.mean(np.abs(shap_matrix_abs), axis=0)

shap_importance_df = pd.DataFrame({
    "variable": FEATURES,
    "label": [FEATURE_LABELS[v] for v in FEATURES],
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False)

shap_importance_csv = os.path.join(RF_DIR, "shap_mean_importance.csv")
shap_importance_df.to_csv(shap_importance_csv, index=False, encoding="utf-8-sig")

print(f"SHAP importance exported: {shap_importance_csv}")


# ============================================================
# 11. Export SHAP values for plotting
# ============================================================

shap_export = shap_sample.copy()
for i, var in enumerate(FEATURES):
    shap_export[f"SHAP_{var}"] = shap_matrix_abs[:, i]

shap_values_csv = os.path.join(RF_DIR, "shap_values_for_plotting.csv")
shap_export.to_csv(shap_values_csv, index=False, encoding="utf-8-sig")

print(f"SHAP values exported: {shap_values_csv}")


# ============================================================
# 12. SHAP summary plot
# ============================================================

top_features = shap_importance_df["variable"].head(12).tolist()
top_labels = [FEATURE_LABELS[v] for v in top_features]

top_idx = [FEATURES.index(v) for v in top_features]

X_top = shap_sample[top_features].copy()
shap_top = shap_matrix_abs[:, top_idx]

plt.figure(figsize=(9, 5.8))

for row, (var, label, idx) in enumerate(zip(top_features[::-1], top_labels[::-1], top_idx[::-1])):
    shap_vals = shap_matrix_abs[:, idx]
    feat_vals = shap_sample[var].values

    # Normalize colors
    vmin, vmax = np.nanpercentile(feat_vals, [2, 98])
    colors = np.clip((feat_vals - vmin) / (vmax - vmin + 1e-9), 0, 1)

    jitter = np.random.default_rng(42 + row).normal(0, 0.08, size=len(shap_vals))
    y_pos = np.full(len(shap_vals), row) + jitter

    plt.scatter(
        shap_vals,
        y_pos,
        c=colors,
        cmap="YlGn",
        s=8,
        alpha=0.55,
        linewidths=0
    )

plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
plt.yticks(range(len(top_features)), top_labels[::-1])
plt.xlabel("SHAP value")
plt.title("(b) SHAP summary of leading environmental predictors", loc="left", fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.2)

cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="YlGn"), ax=plt.gca(), pad=0.02)
cbar.set_label("Feature value")
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Low", "High"])

savefig(os.path.join(RF_DIR, "Fig_SHAP_summary.png"))


# ============================================================
# 13. SHAP dependence plots
# ============================================================

DEPENDENCE_FEATURES = [
    "log_ntl_2023",
    "lst_night_mean_c",
    "elevation_m",
    "water_occurrence",
    "lst_day_mean_c",
    "ndvi_mean",
    "cropland_frac",
    "water_wetland_frac",
]

for var in DEPENDENCE_FEATURES:
    if var not in FEATURES:
        continue

    idx = FEATURES.index(var)

    x = shap_sample[var].values
    y_shap = shap_matrix_abs[:, idx]

    xs, ys = smooth_line(x, y_shap, bins=70)

    plt.figure(figsize=(4.2, 3.4))
    plt.scatter(
        x,
        y_shap,
        c=x,
        cmap="YlGn",
        s=10,
        alpha=0.55,
        linewidths=0
    )
    plt.plot(xs, ys, color="black", linewidth=1.6, alpha=0.75)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel(FEATURE_LABELS[var])
    plt.ylabel("SHAP value")
    plt.title(f"Dependence on {FEATURE_LABELS[var]}", loc="left", fontweight="bold")
    plt.grid(linestyle="--", alpha=0.2)

    out = os.path.join(RF_DIR, f"Fig_SHAP_dependence_{var}.png")
    savefig(out)


# ============================================================
# 14. Interaction surface:
# Night LST × Water occurrence
# ============================================================

print("\nGenerating SHAP interaction surface...")

x_var = "lst_night_mean_c"
y_var = "water_occurrence"

x_idx = FEATURES.index(x_var)
y_idx = FEATURES.index(y_var)

x = shap_sample[x_var].values
y_water = shap_sample[y_var].values

combined_shap = shap_matrix_abs[:, x_idx] + shap_matrix_abs[:, y_idx]

# Grid
x_bins = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 60)
y_bins = np.linspace(np.nanpercentile(y_water, 1), np.nanpercentile(y_water, 99), 60)

Z = np.full((len(y_bins) - 1, len(x_bins) - 1), np.nan)

for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        mask = (
            (x >= x_bins[i]) & (x < x_bins[i + 1]) &
            (y_water >= y_bins[j]) & (y_water < y_bins[j + 1])
        )
        if mask.sum() >= 5:
            Z[j, i] = np.nanmean(combined_shap[mask])

# Fill missing values with nearest available column/row median
if np.isnan(Z).any():
    global_median = np.nanmedian(Z)
    Z = np.where(np.isnan(Z), global_median, Z)

X_grid, Y_grid = np.meshgrid(
    (x_bins[:-1] + x_bins[1:]) / 2,
    (y_bins[:-1] + y_bins[1:]) / 2
)

plt.figure(figsize=(5.0, 4.0))
cf = plt.contourf(
    X_grid,
    Y_grid,
    Z,
    levels=18,
    cmap="RdYlGn"
)
cs = plt.contour(
    X_grid,
    Y_grid,
    Z,
    levels=8,
    colors="black",
    linewidths=0.5,
    alpha=0.7
)
plt.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
plt.xlabel("Night LST")
plt.ylabel("Water occurrence")
plt.title("Interaction surface: Night LST × Water occurrence", loc="left", fontweight="bold")
cbar = plt.colorbar(cf)
cbar.set_label("Combined SHAP value")
savefig(os.path.join(RF_DIR, "Fig_SHAP_interaction_surface.png"))


# ============================================================
# 15. Summary
# ============================================================

summary_path = os.path.join(RF_DIR, "rf_shap_summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Random Forest and SHAP analysis summary\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Input data: {INPUT_CLEAN}\n")
    f.write(f"Sample size: {df.shape[0]}\n")
    f.write(f"Number of predictors: {len(FEATURES)}\n\n")

    f.write(f"Test accuracy: {acc:.4f}\n")
    f.write(f"Five-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")

    f.write("Top permutation importance variables:\n")
    f.write(importance_df.head(15).to_string(index=False))
    f.write("\n\n")

    f.write("Top mean absolute SHAP variables:\n")
    f.write(shap_importance_df.head(15).to_string(index=False))
    f.write("\n\n")

    f.write("Generated figures:\n")
    f.write("- Fig_RF_importance.png\n")
    f.write("- Fig_SHAP_summary.png\n")
    f.write("- Fig_SHAP_dependence_*.png\n")
    f.write("- Fig_SHAP_interaction_surface.png\n")

print("\nSummary exported:")
print(summary_path)

print("\nRandom Forest + SHAP analysis completed successfully.")
print("=" * 70)
