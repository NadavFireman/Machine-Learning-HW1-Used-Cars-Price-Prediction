# Machine Learning - Used Car Price Prediction (HW1)

**Assignment Solution (Grade 95, M.Sc. Data Science, HIT). End-to-end price-prediction pipeline on a messy 400-car dataset, based on the [Kaggle 1056lab](https://www.kaggle.com/c/1056lab-used-cars-price-prediction/overview) challenge.**

## Key Features

- **Multi-Stage Cleaning:** Invalid values and dominant placeholders → NaN, 3-SD outlier removal with train-derived bounds, category unification.
- **Leakage Detection:** Duplicated columns exposed via correlation (`engine` ≡ `price`) and dropped before modeling.
- **Hierarchical Imputation:** Custom `fill_with` engine — group-level mode/median over up to 5 hierarchy levels, fitted on train only.
- **Model Comparison:** Linear Regression vs. Random Forest under Hold-out, 10-Fold CV, and Leave-One-Out; manual grid search over 144 RF configurations — the tuned RF still overfit, so LR was selected.
- **Explainability (SHAP):** Test-set summary plot — production year and mileage dominate price, matching domain intuition.

## Results

Linear Regression, 10-Fold CV: **R² = 0.41, MAE ≈ 1,519 on the held-out test set** — the smallest train-test gap of all configurations.

## Repository Structure

- **`Car_Price_Prediction_ML.ipynb`**: Full implementation (Hebrew narrative; code and charts are language-independent).
- **`Assignment_Instructions_1.pdf`**: Original assignment instructions.
- **`car_prices_dataset.csv`**: The raw dataset (400 records, 14 features).
