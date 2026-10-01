<img width="1536" height="1024" alt="122e2c5b-3b44-4123-9089-3fc48290a98a" src="https://github.com/user-attachments/assets/1a21e569-b110-449a-8f0a-4395eec51f52" />


# ❤️ ML Support Vector Machine

> **ML Fundamentals #5** — Support Vector Machine classification for heart disease prediction using the Cleveland Heart Disease dataset.

## 📌 Overview

This project implements an end-to-end **Support Vector Machine (SVM)** classification workflow for predicting the presence of heart disease from clinical patient attributes.

Beyond building a predictive model, the project explores the core behavior of SVM through **kernel comparison, support-vector analysis, decision-boundary visualization, feature scaling, and hyperparameter optimization**.

The project forms part of my **ML Fundamentals** series, where each repository focuses on implementing and understanding a fundamental machine learning algorithm.

---

## 📊 Dataset

The project uses the **Cleveland Heart Disease dataset** from the UCI Machine Learning Repository.

- **Observations:** 303
- **Predictor Features:** 13
- **Target:** Heart disease diagnosis
- **Problem Type:** Binary Classification

The original diagnosis variable ranges from `0` to `4`.

For binary classification:

- `0` → No Heart Disease
- `1–4` → Heart Disease

### Features

| Feature | Description |
|---|---|
| `age` | Age |
| `sex` | Sex |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia-related test result |

---

## ⚙️ Project Workflow

```text
Cleveland Heart Disease Dataset
              │
              ▼
       Data Understanding
              │
              ▼
 Exploratory Data Analysis
              │
              ▼
     Missing Value Handling
              │
              ▼
 Binary Target Transformation
              │
              ▼
    Stratified Train/Test Split
              │
              ▼
       StandardScaler
              │
              ▼
        Baseline RBF SVM
              │
              ├───────────────┐
              ▼               ▼
      Model Evaluation   Kernel Comparison
                              │
                     Linear / Poly / RBF
                              │
                              ▼
                  Decision Boundary Analysis
                              │
                              ▼
                       GridSearchCV
                              │
                              ▼
                         Tuned SVM
                              │
                              ▼
                    Model Serialization
```

---

## 🔬 Exploratory Data Analysis

The EDA examines:

- Target class distribution
- Continuous clinical feature distributions
- Missing values
- Duplicate observations
- Feature correlations
- Age vs maximum heart rate relationships

All generated visualizations are stored in:

```text
outputs/figures/
```

and analytical tables are stored in:

```text
outputs/tables/
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

1. Identification of missing clinical measurements
2. Median imputation for missing values
3. Conversion of the original diagnosis into a binary target
4. Separation of predictors and target
5. Stratified 80/20 train-test split
6. Standardization using `StandardScaler`

The scaler is fitted **only on the training data** to prevent test-data leakage.

Feature scaling is particularly important for SVM because features with substantially different numerical magnitudes can influence the geometry of the separating hyperplane.

---

## 🧠 Support Vector Machine

The baseline classifier uses an **RBF kernel**:

```python
SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)
```

SVM attempts to construct a decision boundary that separates classes while maximizing the margin between them.

The observations that directly influence this boundary are known as **support vectors**.

---

## 🔀 Kernel Comparison

Three SVM kernels are compared under the same preprocessing and train-test conditions:

### Linear Kernel

Constructs a linear separating hyperplane.

### Polynomial Kernel

Allows nonlinear relationships through polynomial transformations of the feature space.

### RBF Kernel

Uses radial similarity to construct flexible nonlinear decision boundaries.

The comparison evaluates:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Number of support vectors

---

## 📐 Decision Boundary Analysis

One of the primary SVM-specific components of this project is visualization of the classifier's geometry.

Because the final classifier operates using **13 features**, its true decision boundary exists in a 13-dimensional feature space and cannot be directly visualized.

Separate explanatory SVM models are therefore trained using:

```text
Age
Maximum Heart Rate (thalach)
```

These models demonstrate:

- Decision regions
- Separating hyperplanes
- SVM margins
- Support vectors
- Differences between Linear, Polynomial, and RBF boundaries

The 2D models are used **only for visualization** and do not replace the complete 13-feature predictive model.

---

## 🔧 Hyperparameter Optimization

`GridSearchCV` with **5-fold cross-validation** is used to explore combinations of:

- Kernel
- `C`
- `gamma`
- Polynomial degree

The search uses **ROC-AUC** as the optimization metric.

The selected configuration is subsequently evaluated on the untouched test set.

---

## 📈 Final Test Results

| Metric | Score |
|---|---:|
| Accuracy | **88.52%** |
| Precision | **92.00%** |
| Recall | **82.14%** |
| F1 Score | **86.79%** |
| ROC-AUC | **96.32%** |

The final classifier achieves an ROC-AUC of approximately **0.963**, indicating strong discrimination between the two diagnostic classes on the held-out test set.

---

## 🔮 Example Prediction

Example patient:

```text
Age:                         63
Sex:                          1
Chest Pain Type:              1
Resting Blood Pressure:     145
Cholesterol:                233
Fasting Blood Sugar:          1
Resting ECG:                  2
Maximum Heart Rate:         150
Exercise-Induced Angina:      0
Oldpeak:                    2.3
Slope:                        3
Major Vessels:                0
Thal:                         6
```

Model output:

```text
Prediction:
No Heart Disease

Estimated Class Probabilities:
No Heart Disease: 0.232119
Heart Disease:    0.767881
```

`SVC.predict()` determines the class using the SVM decision function, while probability estimates from `SVC(probability=True)` are produced through probability calibration. Therefore, the predicted class and the class with the largest calibrated probability are not guaranteed to coincide.

---

## 💾 Model Persistence

The final pipeline serializes:

```text
models/svm_heart_disease.pkl
models/standard_scaler.pkl
```

The scaler must be applied to new observations before inference so that incoming data undergoes the same transformation used during model development.

---

## 📁 Repository Structure

```text
ml-support-vector-machine/
│
├── data/
│   ├── raw/
│   │   └── heart_disease_cleveland.csv
│   └── processed/
│       └── heart_disease_processed.csv
│
├── models/
│   ├── svm_heart_disease.pkl
│   └── standard_scaler.pkl
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── notebook.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

> Serialized `.pkl` model files are excluded from Git tracking and can be regenerated using `src/train.py`.

---

## 🚀 Running the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Run the prediction pipeline:

```bash
python src/predict.py
```

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## 📚 Key Concepts Demonstrated

- Support Vector Machines
- Maximum-margin classification
- Support vectors
- Linear and nonlinear kernels
- RBF kernel
- Polynomial kernel
- Feature standardization
- Decision boundaries
- Hyperparameter tuning
- Stratified cross-validation
- ROC-AUC analysis
- Model serialization
- Reusable inference pipelines

---

## 🎯 Conclusion

This project demonstrates an end-to-end application of **Support Vector Machines for binary classification**, while also examining the mathematical behavior that distinguishes SVM from other classification algorithms.

Kernel comparison and two-dimensional decision-boundary analysis illustrate how different kernel functions alter the geometry of the classifier, while Grid Search demonstrates the role of hyperparameter selection in model development.

The final workflow combines preprocessing, model training, evaluation, visualization, optimization, persistence, and standalone inference into a reproducible machine learning project.

---

## 📌 ML Fundamentals Series

**#5 — Support Vector Machine**

Previous implementations:

1. Linear Regression
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. **Support Vector Machine ← Current**