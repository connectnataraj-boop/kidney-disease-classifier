# Chronic Kidney Disease Classification

A machine learning project that trains and compares five classifiers on the [UCI Chronic Kidney Disease dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease) to predict whether a patient has chronic kidney disease (CKD).

---

## Results

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Random Forest | ~0.99 | ~0.99 | ~0.99 | ~0.99 |
| SVM (linear) | ~0.99 | ~0.99 | ~0.99 | ~0.99 |
| KNN (k=8) | ~0.98 | ~0.98 | ~0.98 | ~0.98 |
| Decision Tree | ~0.97 | ~0.97 | ~0.97 | ~0.97 |
| Naive Bayes | ~0.96 | ~0.96 | ~0.96 | ~0.96 |

> Exact numbers will vary slightly depending on your scikit-learn version. Run `src/train.py` to reproduce.

---

## Project Structure

```
kidney-disease-classifier/
├── data/                    # Place kidney_disease.csv here (not tracked by git)
├── src/
│   └── train.py             # Full pipeline: preprocessing → training → evaluation
├── results/                 # Generated outputs (confusion matrix PNGs, summary CSV)
├── requirements.txt
└── README.md
```

---

## Dataset

**kidney_disease.csv** from the UCI ML Repository (via Kaggle).

- 400 patient records, 25 features (lab values + symptoms)
- Target: `classification` — `ckd` (1) or `not ckd` (0)

Download from [Kaggle](https://www.kaggle.com/datasets/mansoordaku/ckdisease) and place inside the `data/` folder.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/kidney-disease-classifier.git
cd kidney-disease-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Place kidney_disease.csv in data/ first, then:
python src/train.py
```

Outputs saved to `results/`:
- `correlation_heatmap.png` — feature correlation heatmap
- `cm_<model>.png` — confusion matrix for each classifier
- `model_summary.csv` — metrics table for all models

---

## Pipeline

1. **Load** — reads CSV, drops the `id` column
2. **Type fix** — coerces `pcv`, `wc`, `rc` to numeric (removes stray non-numeric strings)
3. **Impute** — fills numeric columns with mean, categorical with mode
4. **Clean** — fixes dirty string values (`'\tno'`, `' yes'`, etc.)
5. **Encode** — maps all categorical features to 0/1
6. **Scale** — `StandardScaler` fitted on train set only, applied to both splits
7. **Split** — 80/20 train/test, `random_state=42` for reproducibility
8. **Train & evaluate** — 5 models, metrics printed and saved

---

## Models Used

- `DecisionTreeClassifier`
- `KNeighborsClassifier` (k=8)
- `SVC` (linear kernel)
- `RandomForestClassifier`
- `GaussianNB`

---

## Key Design Decisions

- **StandardScaler applied** — critical for KNN (distance-based) and SVM (margin-based); harmless for tree-based models
- **Generator → list comprehension** — avoids silent exhaustion bug when reusing loop variables
- **`fillna` without `inplace=True`** — avoids pandas deprecation warning (≥ 2.x)
- **`zero_division=0`** — prevents warnings when a class has no predicted samples

---

## License

MIT
