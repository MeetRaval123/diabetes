import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and train model (you can cache this)
@st.cache_data
def load_model():
    data = pd.read_csv("diabetes.csv")
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def main():
    st.title("Diabetes Prediction")

    model = load_model()  # Load model once

    # Input form for patient data
    with st.form("patient_form"):
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=180, value=70)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        Age = st.number_input("Age", min_value=1, max_value=120, value=30)

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            prediction = model.predict([input_data])[0]
            if prediction == 1:
                st.error("⚠️ The patient is likely to have diabetes.")
            else:
                st.success("✅ The patient is unlikely to have diabetes.")

if __name__ == "__main__":
    main()
