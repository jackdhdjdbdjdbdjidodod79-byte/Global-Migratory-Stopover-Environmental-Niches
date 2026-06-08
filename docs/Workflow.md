# Workflow

## Step 1

Prepare stopover and environmental datasets

Run:

python code/01_data_preparation.py

Output:

outputs/analysis_dataset_clean.csv

---

## Step 2

Environmental niche differentiation analysis

Run:

python code/02_multivariate_statistics.py

Outputs:

- PCA results
- PERMANOVA
- Kruskal-Wallis tests
- Dunn post-hoc tests

---

## Step 3

Random Forest and SHAP analysis

Run:

python code/03_random_forest_shap_analysis.py

Outputs:

- RF importance
- SHAP summary
- SHAP dependence plots
- Interaction surfaces

---

## Step 4

Generate publication figures

Run:

python code/04_visualization_figures.py

Outputs:

All manuscript figures
