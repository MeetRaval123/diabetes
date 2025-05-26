import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pickle

# Load dataset and train model once
@st.cache_resource
def train_model():
    df = pd.read_csv("diabetes.csv")  # Ensure this file exists
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    imputer = SimpleImputer(strategy='mean')
    df[cols_with_zeros] = imputer.fit_transform(df[cols_with_zeros])

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, scaler, X.columns

model, scaler, feature_names = train_model()

# Hardcoded login credentials
USERNAME = "admin"
PASSWORD = "1234"

def login():
    st.title("🔐 Login to Diabetes Predictor")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.success("Login successful!")
            return True
        else:
            st.error("Invalid credentials")
            return False
    return False

def predict_type(input_data):
    age = input_data[7]
    bmi = input_data[5]
    pregnancies = input_data[0]
    if pregnancies > 0 and age < 40:
        return "🤰 Likely Gestational Diabetes"
    elif age < 30 and bmi < 25:
        return "🧒 Likely Type 1 Diabetes"
    else:
        return "👴 Likely Type 2 Diabetes"

def input_form():
    st.title("🩺 Patient Diabetes Predictor")

    with st.form("patient_form"):
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=180, value=70)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        Insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0)
        DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
        Age = st.number_input("Age", min_value=1, max_value=120, value=30)

        submitted = st.form_submit_button("Predict")
        if submitted:
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
            input_df = pd.DataFrame([input_data], columns=feature_names)
            input_scaled = scaler.transform(input_df)
            result = model.predict(input_scaled)[0]
            if result == 0:
                st.error("❌ The patient is unlikely to have diabetes.")
            else:
                st.success("✅ The patient is likely to have diabetes.")
                diabetes_type = predict_type(input_data)
                st.info(f"🔎 {diabetes_type}")

# App entry point
if __name__ == "__main__":
    if login():
        input_form()
