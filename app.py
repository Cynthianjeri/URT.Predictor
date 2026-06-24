import streamlit as st
import joblib
import pandas as pd
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "logistic_regression_model.joblib")
model = joblib.load(model_path)

st.title("Upper Respiratory Tract Infection Diagnosis Predictor")
st.write("Enter the patient's information below to predict the likely diagnosis.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, value=25)
gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
fever = st.selectbox("Fever", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
cough = st.selectbox("Cough", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
runny_nose = st.selectbox("Runny Nose", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
sore_throat = st.selectbox("Sore Throat", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
shortness_of_breath = st.selectbox("Shortness of Breath", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Predict
if st.button("Predict Diagnosis"):
    input_data = pd.DataFrame(
        [[age, gender, fever, cough, runny_nose, sore_throat, shortness_of_breath]],
        columns=["Age", "Gender", "Fever", "Cough", "Runny nose", "Sore throat", "Shortness of breath"]
    )
    prediction = model.predict(input_data)
    st.success(f"Predicted Diagnosis: {prediction[0]}")
