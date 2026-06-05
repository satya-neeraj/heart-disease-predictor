# Heart Disease Predictor

A machine learning web application that predicts the likelihood of heart disease in a patient based on clinical parameters. Built with Logistic Regression and deployed as an interactive Streamlit interface.

**Objective:** Given 11 clinical and physiological features (e.g., age, cholesterol, chest pain type, ST slope), classify whether a patient has heart disease (binary: `0` = No, `1` = Yes) — enabling early screening support for clinical decision-making.

---

## Features

- Exploratory data analysis with distribution plots, count plots, box plots, violin plots, and a correlation heatmap
- Automatic imputation of biologically impossible zero values in `Cholesterol` and `RestingBP` with column means
- One-hot encoding of all categorical features with `drop_first=True` to avoid multicollinearity
- Comparative benchmarking of five classification algorithms (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM)
- Trained model, scaler, and feature column list persisted via `joblib` for reproducible inference
- Interactive Streamlit web app with sliders, dropdowns, and real-time prediction output

---

## Workflow / Pipeline

```
Raw CSV Data
     │
     ▼
Data Loading & EDA
  (distributions, correlations, class balance)
     │
     ▼
Data Cleaning
  (replace 0-valued Cholesterol & RestingBP with column means)
     │
     ▼
Feature Engineering
  (One-Hot Encoding via pd.get_dummies → 15 features)
     │
     ▼
Train / Test Split  (80% / 20%, random_state=42)
     │
     ▼
Feature Scaling  (StandardScaler on train set)
     │
     ▼
Model Training & Comparison
  (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM)
     │
     ▼
Model Selection  (Logistic Regression — best accuracy & F1)
     │
     ▼
Artifact Serialization
  (logistic_heart.pkl, scaler.pkl, columns.pkl)
     │
     ▼
Streamlit Web App  (app.py — real-time user predictions)
```

---

## Dataset Information

| Property | Details |
|---|---|
| **Source** | [Heart Failure Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |
| **Size** | 918 rows × 12 columns |
| **Target Variable** | `HeartDisease` (0 = No Disease, 1 = Disease) |
| **Class Balance** | 508 positive (55.3%) / 410 negative (44.7%) |

**Input Features:**

| Feature | Type | Description |
|---|---|---|
| `Age` | Numerical | Patient age in years |
| `Sex` | Categorical | M / F |
| `ChestPainType` | Categorical | ATA, NAP, TA, ASY |
| `RestingBP` | Numerical | Resting blood pressure (mm Hg) |
| `Cholesterol` | Numerical | Serum cholesterol (mg/dl) |
| `FastingBS` | Binary | Fasting blood sugar > 120 mg/dl (0/1) |
| `RestingECG` | Categorical | Normal, ST, LVH |
| `MaxHR` | Numerical | Maximum heart rate achieved |
| `ExerciseAngina` | Categorical | Exercise-induced angina (Y/N) |
| `Oldpeak` | Numerical | ST depression induced by exercise |
| `ST_Slope` | Categorical | Slope of peak exercise ST segment (Up/Flat/Down) |

**Preprocessing performed:**
- Zero values in `Cholesterol` and `RestingBP` replaced with the non-zero column mean (biologically impossible values)
- All categorical variables one-hot encoded using `pd.get_dummies(drop_first=True)`, producing 15 model features
- No missing values present in the original dataset
- **Train/Test Split:** 80% / 20% (`test_size=0.20, random_state=42`)
- **Scaling:** `StandardScaler` fitted on training data; applied to both train and test sets

---

## Model(s) Used

Five classifiers were benchmarked under identical preprocessing and scaling conditions:

| Model | Accuracy | F1 Score |
|---|---|---|
| **Logistic Regression** | **0.8696** | **0.8846** |
| KNN | 0.8641 | 0.8815 |
| SVM (RBF kernel) | 0.8478 | 0.8667 |
| Naive Bayes | 0.8478 | 0.8614 |
| Decision Tree | 0.7826 | 0.8039 |

**Selected Model: Logistic Regression**

Chosen for its highest accuracy (86.96%) and F1 score (88.46%) among all evaluated models, as well as its interpretability, robustness on standardized tabular data, and suitability for a binary clinical classification task.

**Hyperparameters:** Default `sklearn` configuration — `solver='lbfgs'`, `max_iter=100`, `C=1.0`, `penalty='l2'`

---

## Performance / Results

Evaluated on the held-out test set (184 samples):

```
              precision    recall  f1-score   support

           0       0.82      0.88      0.85        77
           1       0.91      0.86      0.88       107

    accuracy                           0.87       184
   macro avg       0.87      0.87      0.87       184
weighted avg       0.87      0.87      0.87       184
```

**Confusion Matrix:**

```
               Predicted: No   Predicted: Yes
Actual: No         68               9
Actual: Yes        15              92
```

- **True Positives:** 92 — correctly identified heart disease cases
- **False Negatives:** 15 — missed heart disease cases (higher clinical risk)
- **False Positives:** 9 — healthy patients flagged incorrectly
- **True Negatives:** 68 — correctly identified healthy patients

The model achieves **91% precision** for the positive class, minimizing false alarms, while maintaining **86% recall** — a clinically acceptable balance for a screening tool.

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **ML / Data** | scikit-learn, pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Serialization** | joblib |
| **Web App / Deployment** | Streamlit |
| **Development** | Jupyter Notebook |

---

## Project Structure

```text
heart-disease-predictor/
├── Heart.ipynb              # Full EDA, preprocessing, model training & evaluation notebook
├── app.py                   # Streamlit web application for real-time prediction
├── heart.csv                # Source dataset (918 rows × 12 columns)
├── logistic_heart.pkl       # Serialized trained Logistic Regression model
├── scaler.pkl               # Serialized StandardScaler fitted on training data
├── columns.pkl              # Serialized list of 15 expected feature columns
└── requirements.txt         # Python dependencies
```

---

## Installation & Setup

**Prerequisites:** Python 3.8+

```bash
# Clone the repository
git clone https://github.com/<your-username>/heart-disease-predictor.git
cd heart-disease-predictor

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` installs: `streamlit`, `pandas`, `scikit-learn`, `joblib`, `numpy`

---

## Usage

### Run the Streamlit Web App

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

**Input:** Use the interactive UI to enter patient details:
- Age (slider: 18–100)
- Sex (Male / Female)
- Chest Pain Type (ATA / NAP / TA / ASY)
- Resting Blood Pressure (80–200 mm Hg)
- Serum Cholesterol (100–600 mg/dl)
- Fasting Blood Sugar > 120 mg/dl (0 / 1)
- Resting ECG (Normal / ST / LVH)
- Maximum Heart Rate (slider: 60–220)
- Exercise-Induced Angina (Y / N)
- Oldpeak / ST Depression (slider: 0.0–6.0)
- ST Slope (Up / Flat / Down)

**Output:**
- 🔴 **High likelihood of heart disease** — prompts the user to consult a doctor
- 🟢 **Low likelihood of heart disease** — confirms a healthy result

### Re-run the Training Notebook

```bash
jupyter notebook Heart.ipynb
```

Run all cells to reproduce EDA, preprocessing, model comparison, and artifact generation (`logistic_heart.pkl`, `scaler.pkl`, `columns.pkl`).

---

## Results

- Logistic Regression outperformed all other tested models on this dataset, achieving **86.96% accuracy** and an **F1 score of 0.8846** on the 20% test split.
- Replacing biologically impossible zero values in `Cholesterol` and `RestingBP` with column means measurably cleaned the feature distributions prior to modeling.
- One-hot encoding with `drop_first=True` expanded the 11 raw features into 15 binary/numeric model features.
- The trained model correctly identifies heart disease in **86% of actual positive cases** — viable as a first-pass screening aid.

---

## Future Improvements

- **Hyperparameter tuning:** Apply `GridSearchCV` or `RandomizedSearchCV` on Logistic Regression (`C`, `solver`, `penalty`) and SVM (`C`, `gamma`) to push accuracy further.
- **Ensemble methods:** Evaluate Random Forest and XGBoost, which often outperform single classifiers on tabular medical data.
- **Cross-validation:** Replace single train/test split with stratified k-fold CV for more robust performance estimates.
- **SHAP explainability:** Add SHAP value plots to the Streamlit app to surface which features are driving each individual prediction.
- **Input validation:** Add range checks and warnings in the app for physiologically extreme input values.
- **Docker deployment:** Containerize the Streamlit app for reproducible cloud deployment (e.g., on Render, Railway, or AWS ECS).
- **Larger/multi-source dataset:** Merge with additional heart disease cohorts to improve model generalizability across demographics.
