# Loan Default Probability Prediction – Kaggle Playground Series S5E11

This repository contains my complete solution for the Kaggle competition **Playground Series – Season 5, Episode 11**, focused on predicting the probability that a borrower will repay their loan.

**Final Public Leaderboard Score:** ROC-AUC = 0.92103

---

## 1. Problem Overview

The goal is to estimate the probability that a borrower will fully repay their loan.

**Task Type:** Binary Classification  
**Target Variable:** `loan_paid_back` (1 = repaid, 0 = default)  
**Metric:** ROC-AUC  
**Dataset Size:** ~594k (train), ~255k (test)

---

## 2. Dataset

### Numerical Features
- annual_income  
- debt_to_income_ratio  
- credit_score  
- loan_amount  
- interest_rate  

### Categorical Features
- gender  
- marital_status  
- education_level  
- employment_status  
- loan_purpose  
- grade_subgrade  


---

## 3. Repository Structure

<img width="1190" height="438" alt="image" src="https://github.com/user-attachments/assets/7874ab94-9b54-4920-a703-9d027a95e71f" />

---

## 4. Feature Engineering

### Numerical Engineering
- Log transforms for income and loan amount  
- Ratio features: income_to_loan, credit_to_loan  
- Interaction features: dti_x_interest, loan_x_rate  

### Categorical Engineering
- grade_letter, subgrade_number, grade_letter_ord  
- employment indicators (is_unemployed, is_student, is_retired)

These were applied consistently to both train and test sets.

---

## 5. Modeling

### Preprocessing Pipeline
Implemented via scikit-learn `Pipeline` + `ColumnTransformer`:

- Numerical: median imputation + standard scaling  
- Categorical: mode imputation + one-hot encoding  

### Models Tested
- Logistic Regression  
- XGBoost  
- Simple Neural Network  

**XGBoost delivered the best performance.**

### Final Model Training
Full training data was used with tuned hyperparameters.

Saved model:
```python
from joblib import dump
dump(final_pipeline, "models/final_xgb_pipeline.joblib")
**## 6. Evaluation
### ROC-AUC Results**

Training ROC-AUC: ~0.927

Validation ROC-AUC: ~0.926

Kaggle Submission
test_proba = final_pipeline.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_proba
})

submission.to_csv("submissions/submission_final_xgb.csv", index=False)
**## 7. Explainability (SHAP)**

Top predictors:

debt_to_income_ratio

credit_score

is_unemployed

dti_x_interest

grade_letter_ord

SHAP plots were used for both global and local interpretability.

**8. How to Run**
git clone https://github.com/<your-username>/kaggle-loan-default-s5e11.git
cd kaggle-loan-default-s5e11
pip install -r requirements.txt
jupyter notebook notebooks/Loan_Payback_ML.ipynb


The notebook performs:

EDA

Feature engineering

Model training

Validation

Submission file creation

9. Future Enhancements

LightGBM / CatBoost models

Cross-validation ensembling

Target encoding improvements

Probability calibration

Fairness evaluation

About Me

I am Praneeth Chandra Budala, a Data Science and Artificial Intelligence practitioner with hands-on experience in machine learning, NLP, cloud data engineering, and end-to-end ML solution development across Azure, AWS, Databricks, and Oracle Fusion.

LinkedIn: https://www.linkedin.com/in/praneeth-chandra-budala

GitHub: https://github.com/Praneethchandra-16

Portfolio: https://praneeths-ai-career.lovable.app
