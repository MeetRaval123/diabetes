import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Dummy User Login ---
def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["authenticated"] = True
            st.success("✅ Logged in successfully!")
        else:
            st.error("❌ Invalid username or password.")

# --- Load and Train Model ---
@st.cache_data
def load_model():
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# --- Main App Logic ---
def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login()
        return

    st.title("🩺 Diabetes Prediction Form")

    model = load_model()

    # --- Patient Data Form ---
    with st.form("form1"):
        Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        Glucose = st.number_input("Glucose", 0, 300, 120)
        BloodPressure = st.number_input("Blood Pressure", 0, 180, 70)
        SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
        Insulin = st.number_input("Insulin", 0, 900, 79)
        BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
        DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        Age = st.number_input("Age", 1, 120, 33)

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]]
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("⚠️ Patient is likely to have Diabetes.")
            else:
                st.success("✅ Patient is unlikely to have Diabetes.")

# --- Run App ---
if __name__ == "__main__":
    main()
