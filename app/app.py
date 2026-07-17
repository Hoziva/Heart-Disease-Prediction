import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

pipeline = joblib.load("models/heart_disease_pipeline.pkl")

st.title("❤️ Heart Disease Prediction System")

st.write("""
This application predicts whether a patient is likely to have heart disease
using a Machine Learning model trained on clinical data.
""")

st.header("🩺 Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 20, 100, 45)

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=250,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0,
        max_value=700,
        value=230
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1]
    )

with col2:

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak",
        0.0,
        6.5,
        1.0,
        step=0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )


if st.button("🔍 Predict"):

    patient = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    prediction = pipeline.predict(patient)
    probability = pipeline.predict_proba(patient)[0][1]

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"### Risk Probability: {probability*100:.2f}%")

    st.progress(float(probability))

    st.subheader("📋 Patient Summary")
    st.dataframe(patient)

    st.subheader("💡 Health Recommendations")

    if prediction[0] == 1:
        st.warning("""
- Consult a cardiologist.
- Maintain a healthy diet.
- Exercise regularly.
- Monitor blood pressure.
- Reduce cholesterol.
- Schedule further medical tests.
""")
    else:
        st.success("""
- Continue your healthy lifestyle.
- Exercise regularly.
- Maintain a balanced diet.
- Keep routine medical checkups.
- Monitor your health periodically.
""")