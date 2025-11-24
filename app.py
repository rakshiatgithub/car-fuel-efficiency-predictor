import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="🚗 Auto MPG Predictor",
    page_icon="🚘",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load DL Model & Preprocessor
# -----------------------------
dl_model = tf.keras.models.load_model("auto_mpg_dl_model_extended.h5", compile=False)
preprocessor = joblib.load("preprocessor_extended.joblib")

# -----------------------------
# App Header
# -----------------------------
st.markdown("""
    <div style="background-color:#4CAF50;padding:15px;border-radius:10px">
        <h1 style="color:white;text-align:center;">🚗 Auto MPG Prediction App</h1>
    </div>
""", unsafe_allow_html=True)

st.write("Predict your car's fuel efficiency (MPG) using a Deep Learning model!")
st.markdown("---")

# -----------------------------
# Sidebar Input
# -----------------------------
st.sidebar.header("⚙️ Set Car Features")

cylinders = st.sidebar.slider("Cylinders", 3, 8, 4)
displacement = st.sidebar.slider("Displacement", 50, 500, 150)
horsepower = st.sidebar.slider("Horsepower", 40, 250, 90)
weight = st.sidebar.slider("Weight", 1500, 5500, 3000)
acceleration = st.sidebar.slider("Acceleration", 8, 25, 15)
model_year = st.sidebar.slider("Model Year", 70, 82, 76)
origin = st.sidebar.selectbox("Origin", ["USA", "Europe", "Japan", "India"])

# Map origins to numbers used in training
origin_map = {"USA": 1, "Europe": 2, "Japan": 3, "India": 4}

input_data = pd.DataFrame([{
    "cylinders": cylinders,
    "displacement": displacement,
    "horsepower": horsepower,
    "weight": weight,
    "acceleration": acceleration,
    "model_year": model_year,
    "origin": origin_map[origin]
}])

# -----------------------------
# Preprocess Input
# -----------------------------
dl_input = preprocessor.transform(input_data)

# -----------------------------
# Prediction
# -----------------------------
mpg_prediction = dl_model.predict(dl_input)[0][0]

# -----------------------------
# Display Prediction
# -----------------------------
st.subheader("🔮 Predicted MPG")
st.metric(label="Estimated Fuel Efficiency", value=f"{mpg_prediction:.2f} MPG")

# Optional: Show entered features
with st.expander("🔎 View Input Features"):
    st.table(input_data.T.rename(columns={0: "Value"}))

st.success("Prediction completed! 🎉")
