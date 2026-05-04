"""
Chronic Kidney Disease Classification
--------------------------------------
Trains 5 classifiers on the UCI kidney_disease dataset and reports
accuracy, F1, precision, recall, and confusion matrix for each.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# 1. Load data 

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "kidney_disease.csv")

data = pd.read_csv(DATA_PATH)
data.drop("id", axis=1, inplace=True)


# 2. Fix mixed-type columns 

MIXED_COLS = ["pcv", "wc", "rc"]
for col in MIXED_COLS:
    data[col] = pd.to_numeric(data[col], errors="coerce")


# 3. Impute missing values 

def fill_mean(df, col):
    df[col] = df[col].fillna(df[col].mean())

def fill_mode(df, col):
    df[col] = df[col].fillna(df[col].mode()[0])

num_cols = [c for c in data.columns if data[c].dtype != "object"]
cat_cols = [c for c in data.columns if data[c].dtype == "object"]

for col in num_cols:
    fill_mean(data, col)
for col in cat_cols:
    fill_mode(data, col)

missing = data.isnull().sum()
if missing.any():
    print("Remaining missing values:\n", missing[missing > 0])


# 4. Fix dirty string values 

data["classification"] = data["classification"].replace({"ckd\t": "ckd", "notckd": "not ckd"})
data["cad"] = data["cad"].replace({"\tno": "no"})
data["dm"]  = data["dm"].replace({"\tno": "no", " yes": "yes", "\tyes": "yes"})


# 5. Encode categorical columns

ENCODINGS = {
    "classification": {"ckd": 1, "not ckd": 0},
    "rbc":   {"normal": 1, "abnormal": 0},
    "pc":    {"normal": 1, "abnormal": 0},
    "pcc":   {"present": 1, "notpresent": 0},
    "ba":    {"present": 1, "notpresent": 0},
    "htn":   {"yes": 1, "no": 0},
    "dm":    {"yes": 1, "no": 0},
    "cad":   {"yes": 1, "no": 0},
    "appet": {"good": 1, "poor": 0},
    "pe":    {"yes": 1, "no": 0},
    "ane":   {"yes": 1, "no": 0},
}
for col, mapping in ENCODINGS.items():
    data[col] = data[col].map(mapping)

leaked = data.isnull().sum()
if leaked.any():
    print("NaN after encoding (dropping rows):\n", leaked[leaked > 0])
    data.dropna(inplace=True)


# 6. Correlation heatmap

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(20, 12))
sb.heatmap(data.corr(), annot=True, fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png", dpi=150)
plt.close()

top_features = (
    data.corr()["classification"]
    .abs()
    .sort_values(ascending=False)[1:]
)
print("\nTop features correlated with diagnosis:\n", top_features)


# 7. Train / test split + scaling

X = data.drop("classification", axis=1)
y = data["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   
X_test  = scaler.transform(X_test)       


# 8. Train models & evaluate

MODELS = [
    ("Decision Tree",  DecisionTreeClassifier(random_state=42)),
    ("KNN",            KNeighborsClassifier(n_neighbors=8)),
    ("SVM",            SVC(kernel="linear", random_state=42)),
    ("Random Forest",  RandomForestClassifier(random_state=42)),
    ("Naive Bayes",    GaussianNB()),
]

results = []

for name, model in MODELS:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    results.append({"Model": name, "Accuracy": acc, "F1": f1,
                    "Precision": prec, "Recall": rec})

    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(cm, display_labels=["Not CKD", "CKD"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
    plt.tight_layout()
    plt.savefig(f"results/cm_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


# ── 9. Summary table ──────────────────────────────────────────────────────────

summary = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
print("\n\n=== Summary (sorted by F1) ===")
print(summary.to_string(index=False))
summary.to_csv("results/model_summary.csv", index=False)
print("\nResults saved to results/")