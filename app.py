import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. --- User credentials ---
USER_CREDENTIALS = {
    "admin": "1234",
    "doctor": "doc@123",
    "nurse": "nurse@321"
}

# 2. --- Login Function ---
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.success(f"Welcome, {username}!")
            return True
        else:
            st.error("❌ Invalid credentials")
    return False

# 3. --- Load Model Function ---
@st.cache_data
def load_model():
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 4. --- Prediction Function ---
def predict_diabetes(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]

# 5. --- Main App ---
def main():
    if not login():
        st.stop()

    st.header("🩺 Patient Diabetes Prediction")

    model = load_model()

    # --- Input Form ---
    with st.form("patient_form"):
        st.subheader("Enter Patient Details")

        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=180, value=70)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        Age = st.number_input("Age", min_value=1, max_value=120, value=33)

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
            result = predict_diabetes(model, input_data)
            if result == 1:
                st.error("⚠️ The patient is likely to have Diabetes.")
            else:
                st.success("✅ The patient is unlikely to have Diabetes.")

if __name__ == "__main__":
    main()
