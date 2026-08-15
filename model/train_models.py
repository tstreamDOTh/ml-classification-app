"""
Trains 5 classification models on the UCI Online Shoppers Purchasing
Intention dataset and saves the fitted pipelines plus a metrics summary.

Run from the repo root:
    python model/train_models.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "online_shoppers_intention.csv")
MODEL_DIR = os.path.join(ROOT, "model")

CATEGORICAL = ["Month", "VisitorType"]
TARGET = "Revenue"


def load_data():
    df = pd.read_csv(DATA)
    # booleans to 0/1
    df["Weekend"] = df["Weekend"].astype(int)
    df[TARGET] = df[TARGET].astype(int)
    return df


def build_preprocessor(feature_cols):
    numeric = [c for c in feature_cols if c not in CATEGORICAL]
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, random_state=42),
    }


def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main():
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # keep the raw test split in the repo so the app has data to score
    test_out = X_test.copy()
    test_out[TARGET] = y_test
    test_out.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

    all_metrics = {}
    for name, clf in get_models().items():
        pipe = Pipeline([
            ("prep", build_preprocessor(list(X.columns))),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        all_metrics[name] = evaluate(y_test, y_pred, y_prob)

        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        joblib.dump(pipe, os.path.join(MODEL_DIR, fname))
        print(f"{name:26s} {all_metrics[name]}")

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\nSaved pipelines and metrics.json to model/")


if __name__ == "__main__":
    main()
