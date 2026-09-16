import streamlit as st
import pandas as pd
import joblib
import os

# Resolve paths relative to this script's own folder, not the working
# directory — Streamlit Cloud runs the app with the repo root as cwd,
# so a bare "model.pkl" fails even though the file sits right next to
# this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
model_columns = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))

st.title("Medical Insurance Cost Predictor")
st.write("Enter customer details to estimate insurance charges.")

# --- Input fields ---
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict Insurance Cost"):
    # Build a single-row input matching training encoding exactly
    input_dict = {
        "age": age,
        "sex": 1 if sex == "male" else 0,
        "bmi": bmi,
        "children": children,
        "smoker": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }

    # Reindex to guarantee the same column order as training (northeast = baseline, all zeros)
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.success(f"Estimated Insurance Cost: ${prediction:,.2f}")

    with st.expander("See input used for this prediction"):
        st.write(input_df)