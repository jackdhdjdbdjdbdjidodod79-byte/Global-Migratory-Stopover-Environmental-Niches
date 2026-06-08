# -*- coding: utf-8 -*-
"""
02_multivariate_statistics.py

Multivariate and univariate statistical analyses for:
Environmental niche differentiation of migratory bird stopover sites
revealed by remote sensing and explainable machine learning

This script performs:
1. PCA
2. PERMANOVA
3. Kruskal-Wallis tests
4. Dunn post-hoc pairwise comparisons
5. Significance-letter grouping

Input:
    outputs/analysis_dataset_clean.csv
    outputs/analysis_dataset_standardized.csv

Outputs:
    outputs/statistics/
        pca_scores.csv
        pca_loadings.csv
        permanova_result.txt
        kruskal_wallis_results.csv
        dunn_posthoc_*.csv
        significance_letters.csv
        multivariate_statistics_summary.txt

Author: Likai Lin, Yan Gui
"""

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from scipy.stats import kruskal

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STAT_DIR = os.path.join(OUTPUT_DIR, "statistics")

os.makedirs(STAT_DIR, exist_ok=True)

INPUT_CLEAN = os.path.join(OUTPUT_DIR, "analysis_dataset_clean.csv")
INPUT_STANDARDIZED = os.path.join(OUTPUT_DIR, "analysis_dataset_standardized.csv")

SUMMARY_TXT = os.path.join(STAT_DIR, "multivariate_statistics_summary.txt")


# ============================================================
# 2. Environmental variables
# ============================================================

ENV_VARIABLES = [
    "lst_day_mean_c",
    "lst_night_mean_c",
    "ndvi_mean",
    "log_ntl_2023",
    "log_pop_density_2020",
    "elevation_m",
    "slope_deg",
    "water_occurrence",
    "water_perm_frac",
    "water_high_frac",
    "water_wetland_frac",
    "wc_water_frac",
    "wetland_frac",
    "wetland_mangrove_frac",
    "cropland_frac",
    "forest_frac",
    "grassland_frac",
    "builtup_frac",
    "bare_frac",
    "open_vegetation_frac",
    "shrub_frac",
    "tree_frac",
    "pa_frac",
]

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
# 3. Helper functions
# ============================================================

def format_p_value(p):
    """Format p values for reporting."""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def p_to_stars(p):
    """Convert p value to significance stars."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def bonferroni_correct(p_values):
    """Bonferroni correction."""
    p_values = np.array(p_values, dtype=float)
    corrected = p_values * len(p_values)
    corrected[corrected > 1] = 1
    return corrected


def simple_dunn_posthoc(df, variable, group_col="functional_group"):
    """
    Dunn-style pairwise comparison based on rank sums.

    This avoids requiring scikit-posthocs and is suitable for open reproducible scripts.
    Bonferroni correction is applied.
    """
    from scipy.stats import rankdata, norm

    temp = df[[variable, group_col]].dropna().copy()
    temp["rank"] = rankdata(temp[variable].values)

    groups = sorted(temp[group_col].unique())
    n_total = len(temp)

    rank_mean = temp.groupby(group_col)["rank"].mean()
    group_n = temp.groupby(group_col)["rank"].count()

    comparisons = []
    raw_p = []

    for g1, g2 in combinations(groups, 2):
        r1 = rank_mean[g1]
        r2 = rank_mean[g2]
        n1 = group_n[g1]
        n2 = group_n[g2]

        se = np.sqrt((n_total * (n_total + 1) / 12.0) * (1.0 / n1 + 1.0 / n2))
        z = (r1 - r2) / se
        p = 2 * (1 - norm.cdf(abs(z)))

        comparisons.append((g1, g2, z, p))
        raw_p.append(p)

    adjusted_p = bonferroni_correct(raw_p)

    records = []
    for (g1, g2, z, p), p_adj in zip(comparisons, adjusted_p):
        records.append({
            "variable": variable,
            "label": VARIABLE_LABELS.get(variable, variable),
            "group_1": g1,
            "group_2": g2,
            "z_value": z,
            "p_raw": p,
            "p_adjusted": p_adj,
            "significance": p_to_stars(p_adj),
        })

    result_long = pd.DataFrame(records)

    # Matrix format
    matrix = pd.DataFrame(np.ones((len(groups), len(groups))), index=groups, columns=groups)
    star_matrix = pd.DataFrame("", index=groups, columns=groups)

    for _, row in result_long.iterrows():
        g1 = row["group_1"]
        g2 = row["group_2"]
        p_adj = row["p_adjusted"]
        sig = row["significance"]

        matrix.loc[g1, g2] = p_adj
        matrix.loc[g2, g1] = p_adj
        star_matrix.loc[g1, g2] = sig
        star_matrix.loc[g2, g1] = sig

    np.fill_diagonal(matrix.values, 1.0)
    np.fill_diagonal(star_matrix.values, "—")

    return result_long, matrix, star_matrix


def significance_letters_from_pairwise(pairwise_df, df, variable, group_col="functional_group"):
    """
    Generate compact letter display based on adjusted pairwise p-values.

    Groups are ordered by median from high to low.
    Groups sharing at least one letter are not significantly different.
    """
    medians = df.groupby(group_col)[variable].median().sort_values(ascending=False)
    groups = list(medians.index)

    # Significant pairs
    significant_pairs = set()
    for _, row in pairwise_df.iterrows():
        if row["p_adjusted"] < 0.05:
            significant_pairs.add(tuple(sorted([row["group_1"], row["group_2"]])))

    letters = {}
    letter_list = []

    for group in groups:
        assigned = False

        for letter in letter_list:
            conflict = False
            for other_group, other_letters in letters.items():
                if letter in other_letters:
                    pair = tuple(sorted([group, other_group]))
                    if pair in significant_pairs:
                        conflict = True
                        break

            if not conflict:
                letters[group].append(letter)
                assigned = True
                break

        if not assigned:
            new_letter = chr(ord("a") + len(letter_list))
            letter_list.append(new_letter)
            letters[group] = [new_letter]

        if group not in letters:
            letters[group] = []

    records = []
    for group in groups:
        records.append({
            "variable": variable,
            "label": VARIABLE_LABELS.get(variable, variable),
            "functional_group": group,
            "median": medians[group],
            "letter": "".join(letters[group]),
        })

    return pd.DataFrame(records)


def run_permanova_skbio(X, groups, permutations=999, seed=42):
    """
    Run PERMANOVA using scikit-bio if available.
    If scikit-bio is not installed, return None.
    """
    try:
        from skbio.stats.distance import DistanceMatrix
        from skbio.stats.distance import permanova

        dist_array = pairwise_distances(X, metric="euclidean")

        # Numerical safety
        dist_array = (dist_array + dist_array.T) / 2.0
        np.fill_diagonal(dist_array, 0.0)

        ids = [f"S{i}" for i in range(dist_array.shape[0])]
        dm = DistanceMatrix(dist_array, ids=ids)

        grouping = pd.Series(groups.values, index=ids, name="functional_group")

        np.random.seed(seed)
        result = permanova(dm, grouping, permutations=permutations)

        return result

    except Exception as e:
        print("\nPERMANOVA using scikit-bio failed or scikit-bio is not installed.")
        print(f"Reason: {e}")
        return None


# ============================================================
# 4. Read analysis-ready data
# ============================================================

print("=" * 70)
print("Reading analysis-ready datasets")
print("=" * 70)

if not os.path.exists(INPUT_CLEAN):
    raise FileNotFoundError(
        f"{INPUT_CLEAN} not found. Please run 01_data_preparation.py first."
    )

if not os.path.exists(INPUT_STANDARDIZED):
    raise FileNotFoundError(
        f"{INPUT_STANDARDIZED} not found. Please run 01_data_preparation.py first."
    )

df_clean = pd.read_csv(INPUT_CLEAN, encoding="utf-8-sig")
df_std = pd.read_csv(INPUT_STANDARDIZED, encoding="utf-8-sig")

print(f"Clean data shape: {df_clean.shape}")
print(f"Standardized data shape: {df_std.shape}")

X = df_std[ENV_VARIABLES].values
groups = df_std["functional_group"]

print("\nFunctional group counts:")
print(groups.value_counts())


# ============================================================
# 5. PCA
# ============================================================

print("\n" + "=" * 70)
print("PCA")
print("=" * 70)

pca = PCA(n_components=5, random_state=42)
pca_scores = pca.fit_transform(X)

explained = pca.explained_variance_ratio_

pca_score_df = df_std[
    ["system_index", "stopover_id", "individual_id", "species", "functional_group", "lat", "lon"]
].copy()

for i in range(pca_scores.shape[1]):
    pca_score_df[f"PC{i + 1}"] = pca_scores[:, i]

pca_score_output = os.path.join(STAT_DIR, "pca_scores.csv")
pca_score_df.to_csv(pca_score_output, index=False, encoding="utf-8-sig")

loadings = pd.DataFrame(
    pca.components_.T,
    index=ENV_VARIABLES,
    columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])]
)

loadings.insert(0, "label", [VARIABLE_LABELS[v] for v in ENV_VARIABLES])

pca_loading_output = os.path.join(STAT_DIR, "pca_loadings.csv")
loadings.to_csv(pca_loading_output, encoding="utf-8-sig")

pca_explained_output = os.path.join(STAT_DIR, "pca_explained_variance.csv")
pd.DataFrame({
    "PC": [f"PC{i + 1}" for i in range(len(explained))],
    "explained_variance_ratio": explained,
    "explained_variance_percent": explained * 100,
}).to_csv(pca_explained_output, index=False, encoding="utf-8-sig")

print("Explained variance:")
for i, v in enumerate(explained):
    print(f"PC{i + 1}: {v * 100:.2f}%")

print(f"PCA scores exported: {pca_score_output}")
print(f"PCA loadings exported: {pca_loading_output}")


# ============================================================
# 6. PERMANOVA
# ============================================================

print("\n" + "=" * 70)
print("PERMANOVA")
print("=" * 70)

permanova_result = run_permanova_skbio(X, groups, permutations=999)

permanova_output = os.path.join(STAT_DIR, "permanova_result.txt")

with open(permanova_output, "w", encoding="utf-8") as f:
    if permanova_result is not None:
        f.write(str(permanova_result))
        print(permanova_result)
    else:
        f.write(
            "PERMANOVA was not performed because scikit-bio was not available "
            "or the distance matrix could not be processed.\n"
            "Install scikit-bio to reproduce PERMANOVA:\n"
            "pip install scikit-bio\n"
        )

print(f"PERMANOVA result exported: {permanova_output}")


# ============================================================
# 7. Kruskal-Wallis tests
# ============================================================

print("\n" + "=" * 70)
print("Kruskal-Wallis tests")
print("=" * 70)

kruskal_records = []

for var in ENV_VARIABLES:
    grouped_values = [
        df_clean.loc[df_clean["functional_group"] == g, var].dropna().values
        for g in sorted(df_clean["functional_group"].unique())
    ]

    # Only test if all groups have data
    if all(len(v) > 0 for v in grouped_values):
        h_stat, p_value = kruskal(*grouped_values)
    else:
        h_stat, p_value = np.nan, np.nan

    kruskal_records.append({
        "variable": var,
        "label": VARIABLE_LABELS.get(var, var),
        "H_statistic": h_stat,
        "p_value": p_value,
        "p_formatted": format_p_value(p_value) if not np.isnan(p_value) else "NA",
        "significance": p_to_stars(p_value) if not np.isnan(p_value) else "NA",
    })

kruskal_df = pd.DataFrame(kruskal_records)
kruskal_output = os.path.join(STAT_DIR, "kruskal_wallis_results.csv")
kruskal_df.to_csv(kruskal_output, index=False, encoding="utf-8-sig")

print(kruskal_df[["label", "H_statistic", "p_formatted", "significance"]])
print(f"Kruskal-Wallis results exported: {kruskal_output}")


# ============================================================
# 8. Dunn posthoc pairwise tests
# ============================================================

print("\n" + "=" * 70)
print("Dunn post-hoc pairwise comparisons")
print("=" * 70)

all_pairwise_records = []
all_letter_records = []

for var in ENV_VARIABLES:
    print(f"Processing: {VARIABLE_LABELS.get(var, var)}")

    pairwise_long, p_matrix, star_matrix = simple_dunn_posthoc(
        df_clean,
        variable=var,
        group_col="functional_group"
    )

    all_pairwise_records.append(pairwise_long)

    p_matrix_output = os.path.join(STAT_DIR, f"dunn_posthoc_p_adjusted_{var}.csv")
    star_matrix_output = os.path.join(STAT_DIR, f"dunn_posthoc_significance_{var}.csv")
    long_output = os.path.join(STAT_DIR, f"dunn_posthoc_long_{var}.csv")

    p_matrix.to_csv(p_matrix_output, encoding="utf-8-sig")
    star_matrix.to_csv(star_matrix_output, encoding="utf-8-sig")
    pairwise_long.to_csv(long_output, index=False, encoding="utf-8-sig")

    letters_df = significance_letters_from_pairwise(
        pairwise_long,
        df_clean,
        variable=var,
        group_col="functional_group"
    )

    all_letter_records.append(letters_df)

pairwise_all_df = pd.concat(all_pairwise_records, ignore_index=True)
pairwise_all_output = os.path.join(STAT_DIR, "dunn_posthoc_all_variables_long.csv")
pairwise_all_df.to_csv(pairwise_all_output, index=False, encoding="utf-8-sig")

letters_all_df = pd.concat(all_letter_records, ignore_index=True)
letters_output = os.path.join(STAT_DIR, "significance_letters.csv")
letters_all_df.to_csv(letters_output, index=False, encoding="utf-8-sig")

print(f"All pairwise results exported: {pairwise_all_output}")
print(f"Significance letters exported: {letters_output}")


# ============================================================
# 9. Summary report
# ============================================================

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write("Multivariate statistics summary\n")
    f.write("=" * 70 + "\n\n")

    f.write("Input datasets:\n")
    f.write(f"- {INPUT_CLEAN}\n")
    f.write(f"- {INPUT_STANDARDIZED}\n\n")

    f.write(f"Sample size: {df_clean.shape[0]}\n")
    f.write(f"Number of environmental variables: {len(ENV_VARIABLES)}\n\n")

    f.write("Functional group counts:\n")
    f.write(df_clean["functional_group"].value_counts().to_string())
    f.write("\n\n")

    f.write("PCA explained variance:\n")
    for i, v in enumerate(explained):
        f.write(f"PC{i + 1}: {v * 100:.2f}%\n")
    f.write("\n")

    f.write("PERMANOVA:\n")
    if permanova_result is not None:
        f.write(str(permanova_result))
    else:
        f.write("PERMANOVA was not available. Please install scikit-bio.\n")
    f.write("\n\n")

    f.write("Kruskal-Wallis results:\n")
    f.write(kruskal_df[["label", "H_statistic", "p_formatted", "significance"]].to_string(index=False))
    f.write("\n\n")

    f.write("Output files:\n")
    f.write(f"- {pca_score_output}\n")
    f.write(f"- {pca_loading_output}\n")
    f.write(f"- {pca_explained_output}\n")
    f.write(f"- {permanova_output}\n")
    f.write(f"- {kruskal_output}\n")
    f.write(f"- {pairwise_all_output}\n")
    f.write(f"- {letters_output}\n")

print("\nSummary report exported:")
print(SUMMARY_TXT)

print("\nMultivariate statistics completed successfully.")
print("=" * 70)
