# Machine Learning - Used Car Price Prediction (HW1)

**Assignment Solution (Grade 89, M.Sc. Data Science, HIT). First home assignment in the Machine Learning course, focusing on a complete pipeline for vehicle price estimation. Based on the [Kaggle 1056lab](https://www.kaggle.com/c/1056lab-used-cars-price-prediction/overview) challenge, using data provided for the course.**

## Overview
This project implements an end-to-end Machine Learning workflow to predict used car prices. It covers the essential stages of a data science project: from initial Exploratory Data Analysis (EDA) and data cleaning to model selection and rigorous evaluation using cross-validation.

## Key Features
- **Exploratory Data Analysis (EDA):** Statistical analysis of vehicle features (mileage, engine displacement, age) to uncover correlations and handle missing data.
- **Data Preprocessing:** Cleaning invalid entries, feature scaling, and categorical variable encoding to prepare the dataset for modeling.
- **Model Comparison:** - **Linear Regression:** Selected as the primary model for its interpretability and stability.
  - **Random Forest Regressor:** Evaluated to test the impact of non-linear decision boundaries on prediction accuracy.
- **Evaluation Metrics:** Performance validated using **10-Fold Cross Validation** to ensure results are generalizable, achieving an **R² score of 0.41** on the test set.

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn

## Repository Structure
- `Car_Price_Prediction_ML.ipynb`: Full implementation of the EDA, preprocessing, and machine learning models.
- `assignment_1.pdf`: Original assignment instructions and requirements.
- `car_prices_dataset.csv`: The dataset used for training and evaluation.
