"""
Streamlit app for the ML Assignment 2 (BITS WILP, Machine Learning).

Lets the user upload test data (CSV), pick one of the 5 trained models,
and see evaluation metrics, a confusion matrix and the classification report.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
TARGET = "Revenue"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes (Gaussian)": "naive_bayes_gaussian.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

st.set_page_config(page_title="Online Shoppers Intention Classifier",
                   layout="wide")
st.title("Online Shoppers Purchasing Intention")
st.write("Binary classification on the UCI Online Shoppers Purchasing "
         "Intention dataset. Upload test data, pick a model and review "
         "its performance.")

# 1. dataset upload (test data only, as per assignment)
uploaded = st.file_uploader("Upload test data (CSV with a Revenue column)",
                            type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.caption(f"Using uploaded file: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    df = pd.read_csv(os.path.join(ROOT, "test_data.csv"))
    st.caption(f"No file uploaded, using bundled test_data.csv "
               f"({df.shape[0]} rows). Upload a CSV above to use your own.")

if TARGET not in df.columns:
    st.error(f"The CSV must contain a '{TARGET}' column with the true labels.")
    st.stop()

# tolerate TRUE/FALSE strings or booleans in the label column
if df[TARGET].dtype == object:
    df[TARGET] = df[TARGET].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
df[TARGET] = df[TARGET].astype(int)
if "Weekend" in df.columns and df["Weekend"].dtype == object:
    df["Weekend"] = df["Weekend"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})

X = df.drop(columns=[TARGET])
y = df[TARGET]

# 2. model selection dropdown
model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
pipe = joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[model_name]))

y_pred = pipe.predict(X)
y_prob = pipe.predict_proba(X)[:, 1]

# 3. evaluation metrics
st.subheader("Evaluation metrics")
cols = st.columns(6)
metric_values = [
    ("Accuracy", accuracy_score(y, y_pred)),
    ("AUC", roc_auc_score(y, y_prob)),
    ("Precision", precision_score(y, y_pred)),
    ("Recall", recall_score(y, y_pred)),
    ("F1", f1_score(y, y_pred)),
    ("MCC", matthews_corrcoef(y, y_pred)),
]
for col, (name, val) in zip(cols, metric_values):
    col.metric(name, f"{val:.4f}")

# 4. confusion matrix and classification report
left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No purchase", "Purchase"],
                yticklabels=["No purchase", "Purchase"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(y, y_pred,
                                   target_names=["No purchase", "Purchase"])
    st.code(report)

# comparison table across all models, computed on the same data
st.subheader("All models on this test data")
rows = []
for name, fname in MODEL_FILES.items():
    p = joblib.load(os.path.join(MODEL_DIR, fname))
    yp = p.predict(X)
    ypr = p.predict_proba(X)[:, 1]
    rows.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y, yp), 4),
        "AUC": round(roc_auc_score(y, ypr), 4),
        "Precision": round(precision_score(y, yp), 4),
        "Recall": round(recall_score(y, yp), 4),
        "F1": round(f1_score(y, yp), 4),
        "MCC": round(matthews_corrcoef(y, yp), 4),
    })
st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
