import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load model
@st.cache_data
def load_model():
    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Simple login check
def check_login(username, password):
    return username == "admin" and password == "1234"

def main():
    st.title("🔐 Diabetes Prediction App")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if check_login(username, password):
                st.success("✅ Logged in successfully")
                st.session_state.logged_in = True
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials")
        return

    # After login
    st.subheader("🩺 Enter Patient Details")

    model = load_model()

    with st.form("input_form"):
        Pregnancies = st.number_input("Pregnancies", 0, 20)
        Glucose = st.number_input("Glucose", 0, 300)
        BloodPressure = st.number_input("Blood Pressure", 0, 180)
        SkinThickness = st.number_input("Skin Thickness", 0, 100)
        Insulin = st.number_input("Insulin", 0, 900)
        BMI = st.number_input("BMI", 0.0, 70.0)
        DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
        Age = st.number_input("Age", 1, 120)

        submit = st.form_submit_button("Predict")

        if submit:
            input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]]
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("⚠️ The patient is likely to have diabetes.")
            else:
                st.success("✅ The patient is unlikely to have diabetes.")

if __name__ == "__main__":
    main()
