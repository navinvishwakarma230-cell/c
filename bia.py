import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load Model and Dummy Columns
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
df_dummies = pickle.load(open("data.pkl", "rb"))

# Get the column order model expects
model_columns = df_dummies.drop("Churn", axis=1).columns

st.title("📊 Customer Churn Prediction App")
st.write("Enter the customer details below to predict churn probability.")

# ----------------------------------------
# INPUT FIELDS
# ----------------------------------------

# Senior Citizen (Yes/No → 1/0)
SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
SeniorCitizen = 1 if SeniorCitizen == "Yes" else 0

tenure = st.number_input("Tenure (in months)", 0, 100, 10)
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

gender = st.selectbox("Gender", ["Female", "Male"])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

# ----------------------------------------------------
# Convert Inputs to DataFrame and Match Dummy Columns
# ----------------------------------------------------
input_dict = {
    "SeniorCitizen": SeniorCitizen,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "gender": gender,
    "Partner": Partner,
    "Dependents": Dependents,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod
}

input_df = pd.DataFrame([input_dict])

# One-hot encoding
input_encoded = pd.get_dummies(input_df)

# Align with model columns
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.error(f"❌ Customer is likely to **Churn**.\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Customer will **Not Churn**.\n\nProbability: {prob:.2f}")
