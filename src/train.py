import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

DATA_PATH = "data/raw/heart_disease_cleveland.csv"

df = pd.read_csv(DATA_PATH)

for column in df.columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].median())

df["target"] = (df["target"] > 0).astype(int)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = [
    {
        "kernel": ["linear"],
        "C": [0.1, 1, 10, 100]
    },
    {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1, 1]
    },
    {
        "kernel": ["poly"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "degree": [2, 3]
    }
]

grid_search = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

model = grid_search.best_estimator_

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob)
    ]
})

report = pd.DataFrame(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Heart Disease", "Heart Disease"],
        output_dict=True
    )
).transpose()

metrics.to_csv(
    "outputs/tables/final_evaluation_metrics.csv",
    index=False
)

report.to_csv(
    "outputs/tables/final_classification_report.csv"
)

joblib.dump(
    model,
    "models/svm_heart_disease.pkl"
)

joblib.dump(
    scaler,
    "models/standard_scaler.pkl"
)

print("\nBest Parameters:")
print(grid_search.best_params_)

print(
    "\nBest Cross-Validation ROC-AUC:",
    round(grid_search.best_score_, 6)
)

print("\nTest Set Performance:")
print(metrics.to_string(index=False))

print("\nModel and scaler saved successfully.")