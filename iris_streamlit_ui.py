
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="🌸 Iris Flower Classifier", layout="centered")
st.title("🌸 Iris Flower Classifier")
st.markdown("Enter the flower measurements below:")

# Input fields
sl = st.number_input("Sepal Length (cm)", 0.0, 10.0, step=0.1)
sw = st.number_input("Sepal Width (cm)", 0.0, 10.0, step=0.1)
pl = st.number_input("Petal Length (cm)", 0.0, 10.0, step=0.1)
pw = st.number_input("Petal Width (cm)", 0.0, 10.0, step=0.1)

if st.button("Predict"):
    features = np.array([[sl, sw, pl, pw]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    st.success(f"🌼 Predicted Species: **{prediction.capitalize()}**")