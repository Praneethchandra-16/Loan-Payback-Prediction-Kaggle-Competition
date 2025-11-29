# Loan-Payback-Prediction-Kaggle-Competition
Loan Default Probability Prediction – Kaggle Playground Series S5E11

This repository contains my complete solution for the Kaggle competition
Playground Series – Season 5, Episode 11
:
Predicting the probability that a borrower will repay their loan.

Final Public Leaderboard Score: ROC-AUC = 0.92103

1. Problem Overview

The objective of this challenge is to estimate the likelihood that a borrower will fully repay their loan.

Task Type: Binary Classification

Target Variable: loan_paid_back (1 = paid back, 0 = default)

Evaluation Metric: Area Under ROC Curve (ROC-AUC)

Data Size: ~594k rows (train), ~255k rows (test)

A higher ROC-AUC reflects better ranking of risky versus low-risk borrowers.

2. Dataset
Key Columns

Numerical Features

annual_income

debt_to_income_ratio

credit_score

loan_amount

interest_rate

Categorical Features

gender

marital_status

education_level

employment_status

loan_purpose

grade_subgrade (combined letter+number, e.g., C3)

Place the original CSVs in the data/ folder:

data/train.csv  
data/test.csv

3. Repository Structure
.
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── Loan_Payback_ML.ipynb
├── models/
│   └── final_xgb_pipeline.joblib
├── submissions/
│   └── submission_final_xgb.csv
├── src/
│   └── utils.py
├── requirements.txt
└── README.md

4. Exploratory Data Analysis (EDA)
4.1 Target Distribution

Approximately 80% of loans were fully repaid.

Approximately 20% defaulted.

Stratified splitting was used to maintain class balance. ROC-AUC was preferred over accuracy due to imbalance.

4.2 Numerical Feature Insights

debt_to_income_ratio: Strong positive correlation with default risk.

credit_score: Higher scores strongly correlate with repayment likelihood.

interest_rate: Higher values indicate higher risk.

annual_income, loan_amount: Weak individually; stronger after ratios/interaction features are engineered.

4.3 Categorical Feature Insights

employment_status: Clear risk differences (Unemployed/Students riskier; Retired/Employed safer).

grade_subgrade: One of the strongest predictors after splitting into grade and subgrade.

5. Feature Engineering

Feature engineering was applied consistently to both train and test datasets.

5.1 Engineered Numerical Features

log_annual_income = log1p(annual_income)

log_loan_amount = log1p(loan_amount)

income_to_loan = annual_income / loan_amount

credit_to_loan = credit_score / loan_amount

dti_x_interest = debt_to_income_ratio * interest_rate

loan_x_rate = loan_amount * interest_rate

These transformations reduce skew, encode ratios, and capture borrower risk exposure.

5.2 Engineered Categorical / Risk Features

From grade_subgrade:

grade_letter

subgrade_number

grade_letter_ord (ordinal mapping A→6 … F→1)

Employment-based indicators:

is_unemployed

is_student

is_retired

5.3 Final Feature Groups

Numerical (raw + engineered)
Annual income, loan amount, ratios, interactions, logs.

Categorical
Gender, marital status, education, employment status, loan purpose, grade letter.

Ordinal / Risk Flags
Grade ordinal, subgrade number, employment risk flags.

6. Modeling Approach
6.1 Train/Validation Split

Used stratified 80/20 split to retain class proportions.

6.2 Preprocessing Pipeline

Implemented via Pipeline + ColumnTransformer:

Numerical

Median imputation

Standard scaling

Categorical

Mode imputation

One-hot encoding

6.3 Baseline Models

Logistic Regression

XGBoost

Simple Neural Network

XGBoost achieved the best performance.

6.4 Hyperparameter Tuning

Used GridSearchCV / RandomizedSearchCV with stratified K-fold cross-validation.

Key parameters tuned:

n_estimators

learning_rate

max_depth

subsample

colsample_bytree

min_child_weight

6.5 Final Model Training

Trained full pipeline on all training data using optimal hyperparameters.

Saved model:

from joblib import dump
dump(final_pipeline, "models/final_xgb_pipeline.joblib")

7. Evaluation
7.1 Train vs Validation ROC-AUC

Train ROC-AUC: ~0.927

Validation ROC-AUC: ~0.926

Close values indicate strong generalization.

7.2 ROC Curve

Validation ROC curve remains consistently above baseline, implying strong ranking capability.

7.3 Kaggle Leaderboard Performance

Generated predictions:

test_proba = final_pipeline.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_proba
})

submission.to_csv("submissions/submission_final_xgb.csv", index=False)


Public Leaderboard ROC-AUC: 0.92103

8. Model Explainability with SHAP
8.1 Global Interpretability

Top predictors include:

is_unemployed

debt_to_income_ratio

credit_score

dti_x_interest

grade_letter_ord

8.2 Local Interpretability

SHAP waterfall plots reveal exactly why a borrower is classified as high- or low-risk, enabling transparent credit decisions.

9. How to Run

Clone the repository:

git clone https://github.com/<your-username>/kaggle-loan-default-s5e11.git
cd kaggle-loan-default-s5e11


Install dependencies:

pip install -r requirements.txt


Download Kaggle data and place it in the data/ folder.

Run the notebook:

jupyter notebook notebooks/Loan_Payback_ML.ipynb


This will:

Execute EDA

Generate engineered features

Train and evaluate models

Produce the submission CSV

10. Possible Extensions

Cross-validation ensembling

LightGBM / CatBoost models

Target encoding for categorical variables

Probability calibration

Fairness analysis across demographic subgroups

11. Acknowledgements

Competition and data by Kaggle (Playground Series – S5E11)

Libraries: Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn

About Me

I am Praneeth Chandra Budala, a Data Science and Artificial Intelligence enthusiast with strong hands-on experience in machine learning, NLP, data engineering, and full project lifecycle execution. I build real-world, end-to-end analytics and ML solutions across cloud platforms including Azure, AWS, Databricks, and Oracle Fusion.

LinkedIn: https://www.linkedin.com/in/praneeth-chandra-budala

GitHub: https://github.com/Praneethchandra-16

Portfolio Website: https://praneeths-ai-career.lovable.app

If you would like to discuss this project or collaborate on future work, feel free to connect.
