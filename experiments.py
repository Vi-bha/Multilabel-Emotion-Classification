"""
Comparative Evaluation of Multi-Label Classification Methods
for Emotion Recognition

Paper: "Comparison of Multilabel Classification Techniques for Emotion Recognition"
Authors: Tummewar V., Shukla S., Gyanchandani M.
Published: Springer ICIoTCT 2025

Evaluates 4 transformation methods × 4 classifiers = 16 combinations
on the Emotions benchmark dataset (593 samples, 6 emotion labels).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.ensemble import RakelD

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, hamming_loss, f1_score,
    precision_score, recall_score, jaccard_score
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE    = 0.2
RESULTS_DIR  = Path("results")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_emotions():
    """Load and return the Emotions dataset as dense numpy arrays."""
    print("Loading Emotions dataset...")
    X, y, _, _ = load_dataset('emotions', 'undivided')
    X = X.toarray()
    y = y.toarray()
    print(f"  Features : {X.shape}")
    print(f"  Labels   : {y.shape}")
    print(f"  Classes  : anger, disgust, fear, joy, sadness, surprise\n")
    return X, y

# ---------------------------------------------------------------------------
# Classifiers & Methods
# ---------------------------------------------------------------------------

def get_classifiers():
    return {
        "Naive Bayes"        : GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM (Linear)"       : LinearSVC(random_state=RANDOM_STATE),
    }

def build_model(method_name, base_clf):
    """Instantiate the correct MLC wrapper for a given method and base classifier."""
    if method_name == "Binary Relevance":
        return BinaryRelevance(classifier=base_clf, require_dense=[True, True])

    elif method_name == "Classifier Chain":
        return ClassifierChain(classifier=base_clf, require_dense=[True, True])

    elif method_name == "Label Powerset":
        return LabelPowerset(classifier=base_clf, require_dense=[True, True])

    elif method_name == "RAkEL":
        # RakelD uses a different API — base_classifier instead of classifier
        return RakelD(
            base_classifier=base_clf,
            base_classifier_require_dense=[True, True],
            labelset_size=3,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred) -> dict:
    """Compute all MLC evaluation metrics for a set of predictions."""
    return {
        "Subset Accuracy"    : accuracy_score(y_true, y_pred),
        "Hamming Loss"       : hamming_loss(y_true, y_pred),
        "Precision (micro)"  : precision_score(y_true, y_pred, average='micro',   zero_division=0),
        "Recall (micro)"     : recall_score(y_true,    y_pred, average='micro',   zero_division=0),
        "F1 Score (micro)"   : f1_score(y_true,        y_pred, average='micro',   zero_division=0),
        "Precision (macro)"  : precision_score(y_true, y_pred, average='macro',   zero_division=0),
        "Recall (macro)"     : recall_score(y_true,    y_pred, average='macro',   zero_division=0),
        "F1 Score (macro)"   : f1_score(y_true,        y_pred, average='macro',   zero_division=0),
        "Jaccard (micro)"    : jaccard_score(y_true,   y_pred, average='micro'),
        "Jaccard (samples)"  : jaccard_score(y_true,   y_pred, average='samples'),
    }

# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run_experiments(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Run all 16 method × classifier combinations and return a results DataFrame.
    Each row is one combination; columns are the evaluation metrics.
    """
    classifiers = get_classifiers()
    methods     = ["Binary Relevance", "Classifier Chain", "Label Powerset", "RAkEL"]
    records     = []

    total = len(classifiers) * len(methods)
    run   = 0

    for clf_name, base_clf in classifiers.items():
        for method_name in methods:
            run += 1
            print(f"[{run:02d}/{total}] {method_name:<20} + {clf_name}")

            model = build_model(method_name, base_clf)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = evaluate(y_test, y_pred)
            metrics["Classifier"] = clf_name
            metrics["Method"]     = method_name
            records.append(metrics)

    df = pd.DataFrame(records)
    # Reorder columns: identifiers first, then metrics
    id_cols     = ["Method", "Classifier"]
    metric_cols = [c for c in df.columns if c not in id_cols]
    return df[id_cols + metric_cols]

# ---------------------------------------------------------------------------
# Save & display
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame):
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "results.csv"
    df.round(4).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

def print_summary(df: pd.DataFrame):
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    display_cols = ["Method", "Classifier", "Subset Accuracy",
                    "Hamming Loss", "F1 Score (micro)", "F1 Score (macro)"]
    print(df[display_cols].round(4).to_string(index=False))

    print("\n" + "="*70)
    print("BEST COMBINATIONS PER METRIC")
    print("="*70)

    highlight = {
        "Subset Accuracy" : "max",
        "Hamming Loss"    : "min",
        "F1 Score (micro)": "max",
        "F1 Score (macro)": "max",
    }

    for metric, direction in highlight.items():
        idx = df[metric].idxmin() if direction == "min" else df[metric].idxmax()
        row = df.loc[idx]
        print(f"  {metric:<22} → {row['Method']} + {row['Classifier']}  ({row[metric]:.4f})")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load data
    X, y = load_emotions()

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples\n")

    # 3. Run all 16 combinations
    print("Running experiments...")
    results_df = run_experiments(X_train, X_test, y_train, y_test)

    # 4. Display and save
    print_summary(results_df)
    save_results(results_df)
