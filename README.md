# Machine Learning - Used Car Price Prediction (HW1)

**Assignment Solution (Grade 95, M.Sc. Data Science, HIT). First home assignment in the Machine Learning course: a complete price-prediction pipeline on a 400-car dataset with heavy, realistic data-quality problems. Based on the [Kaggle 1056lab](https://www.kaggle.com/c/1056lab-used-cars-price-prediction/overview) challenge, using data provided for the course.**

## Overview

An end-to-end regression workflow where most of the battle is the data: negative horsepower values, dominant placeholder values masking ~70% of the target column, duplicated features, and missing values in nearly every column. The pipeline covers leakage-aware train/validation/test splitting, multi-stage cleaning, hierarchical imputation, model comparison under three validation schemes, hyperparameter tuning, and SHAP-based explainability.

## Key Features

- **Data Cleaning (multi-stage):** Invalid negatives and dominant placeholder values replaced with NaN, 3-SD outlier removal with train-derived bounds applied to all splits, category unification, and binning of continuous features.
- **Leakage Detection:** Correlation analysis exposing two duplicated columns (`engine` ≡ `price`, `dashboard color` ≡ `color`) — both dropped before modeling.
- **Hierarchical Imputation:** A custom `fill_with` engine that fills each column from group-level statistics (mode/median) over up to 5 correlation-driven hierarchy levels, falling back gracefully — fitted on train only and applied to validation/test.
- **Model Comparison under 3 Validation Schemes:** Linear Regression vs. Random Forest, each evaluated with Hold-out, 10-Fold Cross-Validation, and Leave-One-Out (MAE, RMSE, R²) — including train-vs-validation gap analysis to diagnose overfitting.
- **Hyperparameter Tuning:** Manual grid search over 144 Random Forest configurations (depth, splits, leaves, features, criterion); the tuned RF still overfit, so the more stable Linear Regression was selected.
- **Explainability (SHAP):** LinearExplainer summary plot on the test set — production year and mileage emerge as the dominant price drivers, with directions matching domain intuition.

## Results

Final model: Linear Regression, evaluated with 10-Fold CV — **R² = 0.41 / MAE ≈ 1,519 on the held-out test set**, with the smallest train-test gap of all configurations. The takeaway: on a small, noisy dataset, the simpler model generalizes better than a tuned Random Forest.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · SHAP · Matplotlib · Seaborn

## Repository Structure

- **`Car_Price_Prediction_ML.ipynb`**: Full implementation — cleaning, hierarchical imputation, model comparison, tuning, and SHAP analysis (Hebrew narrative; code, tables, and charts are language-independent).
- **`Assignment_Instructions_1.pdf`**: Original assignment instructions.
- **`car_prices_dataset.csv`**: The raw dataset (400 records, 14 features).

***
*Course: Machine Learning, M.Sc. Data Science, HIT.*
