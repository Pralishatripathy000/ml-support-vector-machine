import joblib
import pandas as pd

MODEL_PATH = "models/svm_heart_disease.pkl"
SCALER_PATH = "models/standard_scaler.pkl"
DATA_PATH = "data/raw/heart_disease_cleveland.csv"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)

sample = X.iloc[[0]]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)[0]
probabilities = model.predict_proba(sample_scaled)[0]

diagnosis = (
    "Heart Disease"
    if prediction == 1
    else "No Heart Disease"
)

print("\nPrediction:")
print(diagnosis)

print("\nEstimated Class Probabilities:")

for class_label, probability in zip(model.classes_, probabilities):
    class_name = (
        "Heart Disease"
        if class_label == 1
        else "No Heart Disease"
    )

    print(f"{class_name}: {probability:.6f}")

print(
    "\nNote: SVM class prediction is determined by the decision function; "
    "calibrated probability estimates may not always select the same class."
)