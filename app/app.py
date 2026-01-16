import streamlit as st
import pandas as pd
import joblib

# Load model
import os


MODEL_PATH = os.path.join("model", "delivery_delay_model.pkl")
model = joblib.load(MODEL_PATH)

st.title("Delivery Delay Prediction")

st.write("Predict whether a delivery will be delayed before dispatch.")

# User inputs
dispatch_delay_hrs = st.number_input("Dispatch delay (hours)", 0.0, 10.0, 2.0)
distance_km = st.number_input("Distance (km)", 1.0, 200.0, 30.0)

traffic_level = st.selectbox(
    "Traffic Level",
    options=["Low", "Medium", "High"]
)

weather = st.selectbox(
    "Weather Condition",
    options=["Clear", "Rain", "Storm"]
)

courier_partner = st.selectbox(
    "Courier Partner",
    options=["A", "B", "C"]
)

promised_delivery_hrs = st.number_input(
    "Promised delivery time (hours)",
    1.0, 50.0, 10.0
)

# Encoding (must match training)
traffic_map = {"Low": 0, "Medium": 1, "High": 2}
weather_map = {"Clear": 0, "Rain": 1, "Storm": 2}
courier_map = {"A": 0, "B": 1, "C": 2}

input_df = pd.DataFrame([{
    "dispatch_delay_hrs": dispatch_delay_hrs,
    "distance_km": distance_km,
    "traffic_level": traffic_map[traffic_level],
    "weather": weather_map[weather],
    "courier_partner": courier_map[courier_partner],
    "promised_delivery_hrs": promised_delivery_hrs
}])

if st.button("Predict Delay"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Delivery likely to be DELAYED (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Delivery likely ON TIME (Probability: {1 - probability:.2f})")
