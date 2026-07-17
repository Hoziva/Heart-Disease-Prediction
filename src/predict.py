import joblib
import pandas as pd

# Load the trained pipeline
pipeline = joblib.load("models/heart_disease_pipeline.pkl")

# Example patient
patient = {
    "Age": 45,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 120,
    "Cholesterol": 230,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 150,
    "ExerciseAngina": "N",
    "Oldpeak": 1.2,
    "ST_Slope": "Up"
}

# Convert to DataFrame
patient_df = pd.DataFrame([patient])

# Make prediction
prediction = pipeline.predict(patient_df)

# Prediction result
if prediction[0] == 1:
    print("⚠️ Heart Disease Detected")
else:
    print("✅ No Heart Disease")