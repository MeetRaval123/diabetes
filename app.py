import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- 1. User login system ---
USER_CREDENTIALS = {
    "admin": "1234",
    "doctor": "doc@123",
    "nurse": "nurse@321"
}

def login():
    st.title("🔐 Login to Diabetes Predictor")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.success(f"Welcome, {username}!")
            return username
        else:
            st.error("Invalid credentials")
    return None

# --- 2. Load dataset and train model ---
@st.cache_data(show_spinner=False)
def load_and_train():
    data = pd.read_csv("diabetes.csv")  # Make sure diabetes.csv is in the same folder
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# --- 3. Predict diabetes based on user inputs ---
def predict_diabetes(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]  # 0 or 1

# --- 4. Main app ---
def main():
    # Login
    user = login()
    if not user:
        st.stop()  # Stop app if not logged in

    st.header("Patient Data Input")

    # Load model once
    model = load_and_train()

    # --- 5. Patient data input form ---
    with st.form("patient_form"):
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
            result = predict_diabetes(model, input_data)
            if result == 1:
                st.error("⚠️ The patient is likely to have Diabetes.")
            else:
                st.success("✅ The patient is unlikely to have Diabetes.")

if __name__ == "__main__":
    main()
