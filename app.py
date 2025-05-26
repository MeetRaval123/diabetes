import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Load and Train Model ---
@st.cache_data
def load_model():
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# --- Check login credentials ---
def check_login(username, password):
    credentials = {
        "admin": "1234",
        "doctor": "doc@123",
        "nurse": "nurse@321"
    }
    return credentials.get(username) == password

# --- Main App ---
def main():
    st.title("🔐 Diabetes Prediction App with Login")

    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Login Section
    if not st.session_state.logged_in:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_login(username, password):
                st.success(f"✅ Welcome, {username}!")
                st.session_state.logged_in = True
                st.ex
