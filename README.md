# Employee Attrition Analysis & Prediction

This project predicts whether an employee might leave the company (**attrition**) and also predicts their **performance rating** based on historical HR data.

It uses:
- Data preprocessing & cleaning
- Machine Learning models
- A **Streamlit** web application to interactively make predictions

---

## **Goals**
- Identify employees at high risk of attrition to support HR retention strategies
- Predict employee performance ratings to support workforce planning

---

## **Project Structure**

Project/
├── app.py # Streamlit web app
├── preprocess.py # Preprocessing script
├── train_attrition_model.py # Train attrition prediction model
├── train_performance_model.py # Train performance rating model
├── models/ # Saved models, scalers & encoders
│ ├── best_model.pkl
│ ├── performance_model.pkl
│ ├── attrition_scaler.pkl
│ ├── performance_scaler.pkl
│ ├── encoders.pkl
│ └── performance_features.pkl
├── data/
│ └── Employee-Attrition - Employee-Attrition.csv
└── requirements.txt # Python dependencies

---

## **Dataset**
- Employee data: age, monthly income, job satisfaction, overtime, years at company, job level, job involvement, etc.
- Target variables:
  - `Attrition` (Yes/No → 1/0)
  - `PerformanceRating` (1–4)

---

## **How to run locally**

1) Clone the repo or copy the files

2) Install dependencies:

pip install -r requirements.txt

3) Preprocess data:

python preprocess.py

4) Train models:

python train_attrition_model.py
python train_performance_model.py

5) Run Streamlit app:
streamlit run app.py

---

**Features used**
1) Attrition prediction:

Age

MonthlyIncome

JobSatisfaction

OverTime

2) Performance prediction:

YearsAtCompany

JobLevel

JobInvolvement

MonthlyIncome

---

**Results & Metrics**
Random Forest models trained with accuracy & ROC-AUC scores printed in console

Streamlit app displays:

Whether an employee will likely leave (and probability)

Predicted performance rating

---

**Technologies & Libraries**
Python

pandas

scikit-learn

joblib

Streamlit


---

## **requirements.txt**

pandas
scikit-learn
joblib
streamlit
