# Customer Churn Prediction

Predicting whether a telecom customer will churn using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features).

## Project Overview

This project covers a full ML pipeline: EDA, data cleaning, feature engineering, encoding, handling class imbalance, model training, threshold tuning, and evaluation.

## Key EDA Findings

- Customers on month-to-month contracts churn far more than those on one/two-year contracts.
- Fiber optic internet users have a higher churn rate than DSL or no-internet customers.
- Customers without Online Security, Tech Support, or Device Protection are more likely to churn.
- Electronic check payment is associated with the highest churn; automatic payments show lower churn.
- Gender has almost no effect on churn (~26% churn rate for both) and was dropped.
- Low tenure + high monthly charges is the strongest churn signal.

## Preprocessing

- Fixed `TotalCharges` (converted to numeric, handled missing values).
- Train/test split performed **before** imputation and scaling to avoid data leakage.
- Categorical features encoded with `OrdinalEncoder` (Contract) and `OneHotEncoder` (remaining categorical columns) inside a `ColumnTransformer` / `Pipeline`.
- Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) scaled with `StandardScaler`.

## Models Trained

| Model | Accuracy | Recall | Precision | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 76.51% | 79.32% | 55.29% | 65.16% |
| Decision Tree | 77.22% | 74.06% | 55.29% | 69.15% |
| Random Forest | 79.16% | 74.06% | 60.75% | 72.74% |

Class imbalance was handled with `class_weight="balanced"`, and the classification threshold was tuned using the precision-recall curve (best threshold ≈ 0.35 for Logistic Regression).

## Most Important Features

Based on Random Forest feature importances: **Contract type, tenure, Internet Service, Monthly Charges**, and additional service subscriptions (Online Security, Tech Support, Payment Method).

## Tools Used

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GridSearchCV, Pipeline, ColumnTransformer)

## Files

- `01_Yasules.ipynb` — EDA, cleaning, initial exploration
- `02_Prediction.ipynb` — Encoding, pipeline, model training, tuning, evaluation, prediction function

## How to Run

1. Open the notebooks in Google Colab.
2. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` to the Colab environment.
3. Run all cells in order.
