import streamlit as st
import pandas as pd
import joblib

st.title("🚀 Employee Analytics App")

attrition_model = joblib.load(r"C:\Users\Bala viknese\PycharmProjects\PythonProject3\models\best_model.pkl")
performance_model = joblib.load(r"C:\Users\Bala viknese\PycharmProjects\PythonProject3\models\performance_model.pkl")
attrition_scaler = joblib.load(r"C:\Users\Bala viknese\PycharmProjects\PythonProject3\models\attrition_scaler.pkl")
performance_scaler = joblib.load(r"C:\Users\Bala viknese\PycharmProjects\PythonProject3\models\performance_scaler.pkl")
encoders = joblib.load(r"C:\Users\Bala viknese\PycharmProjects\PythonProject3\models\encoders.pkl")
attrition_features = joblib.load(r"C:\Users\Bala viknese\PycharmProjects\PythonProject3\models\attrition_features.pkl")
perf_features = joblib.load(r"C:\Users\Bala viknese\PycharmProjects\PythonProject3\models\performance_features.pkl")

tab1, tab2 = st.tabs(["Attrition Prediction", "Performance Rating Prediction"])

with tab1:
    st.header("Predict Attrition")
    age = st.number_input("Age", 18, 60, 30)
    monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    overtime_encoded = encoders['OverTime'].transform([overtime])[0]

    input_df = pd.DataFrame({
        'Age': [age],
        'MonthlyIncome': [monthly_income],
        'JobSatisfaction': [job_satisfaction],
        'OverTime': [overtime_encoded]
    })
    input_df = input_df.reindex(columns=attrition_features, fill_value=0)

    if st.button("Predict Attrition"):
        input_scaled = attrition_scaler.transform(input_df)
        prediction = attrition_model.predict(input_scaled)
        prob = attrition_model.predict_proba(input_scaled)[0][1]

        st.subheader(f"Prediction: {'Will Leave' if prediction[0]==1 else 'Will Stay'}")
        st.subheader(f"Attrition Probability: {prob*100:.2f}%")

with tab2:
    st.header("Predict Performance Rating")
    years_at_company = st.number_input("Years at Company", 0, 40, 5)
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
    monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000, key="perf_income")

    input_df2 = pd.DataFrame({
        'YearsAtCompany': [years_at_company],
        'JobLevel': [job_level],
        'JobInvolvement': [job_involvement],
        'MonthlyIncome': [monthly_income]
    })
    input_df2 = input_df2.reindex(columns=perf_features, fill_value=0)

    if st.button("Predict Performance Rating"):
        input_scaled2 = performance_scaler.transform(input_df2)
        pred = performance_model.predict(input_scaled2)
        st.subheader(f"Predicted Performance Rating: {pred[0]}")
