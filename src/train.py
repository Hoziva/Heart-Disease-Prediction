import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/heart.csv")

print("Dataset loaded successfully!")

# Features and Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# ==========================
# Define Feature Types
# ==========================

numeric_features = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak"
]

categorical_features = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

# ==========================
# Create Preprocessor
# ==========================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ==========================
# Create Pipeline
# ==========================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ]
)

# ==========================
# Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Train Model
# ==========================

pipeline.fit(X_train, y_train)

# ==========================
# Evaluate Model
# ==========================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}\n")

print("Classification Report")
print(classification_report(y_test, y_pred))

# ==========================
# Save Pipeline
# ==========================

joblib.dump(
    pipeline,
    "models/heart_disease_pipeline.pkl"
)

print("\nPipeline saved successfully!")