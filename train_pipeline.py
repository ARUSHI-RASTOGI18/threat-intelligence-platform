"""
ML Training & Artifact Serialization Pipeline
Location: ./train_pipeline.py
Execution: python train_pipeline.py
Purpose: Trains a Random Forest attack type classifier using temporal out-of-time holdout validation,
         computes multiclass confusion matrix, macro/weighted metrics, and serializes artifacts.
"""

import os
import json
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, classification_report, confusion_matrix
)

from src.data_loader import load_analytical_data
from src.preprocessing import build_preprocessor, prepare_data_for_training, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def main():
    print("\n=======================================================")
    print("      GTI-ARP ACADEMIC ML TRAINING & VALIDATION       ")
    print("=======================================================")

    # 1. Load Data
    print("[1/5] Ingesting dataset...")
    df = load_analytical_data(force_reload=True)
    min_yr = int(df["year"].min())
    max_yr = int(df["year"].max())
    print(f"      Loaded dataset: {len(df):,} rows | Horizon: {min_yr} – {max_yr}")

    # 2. Prepare Features & Target
    print("[2/5] Preparing pre-event feature space (no outcome leakage)...")
    X, y = prepare_data_for_training(df, min_samples_per_class=50)
    
    # Encode Target Labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes_list = label_encoder.classes_.tolist()

    # 3. Temporal Out-of-Time Validation Split
    # Academic justification: Time-series/event data must be evaluated on future unseen periods.
    print("[3/5] Applying Temporal Out-of-Time Holdout Split...")
    split_year = max_yr - 3  # e.g., if max is 2017, train on <= 2014, test on 2015-2017
    
    train_mask = df.loc[X.index, "year"] <= split_year
    test_mask = df.loc[X.index, "year"] > split_year

    # Fallback to stratified split if temporal split produces degenerate test distributions
    if test_mask.sum() < 500 or len(np.unique(y_encoded[test_mask])) < len(classes_list):
        print("      [NOTE] Temporal split unbalanced; applying Stratified 80/20 Holdout Split.")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
        )
        val_strategy = "Stratified Random Holdout (80/20)"
    else:
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_encoded[train_mask], y_encoded[test_mask]
        val_strategy = f"Temporal Out-of-Time Holdout (Train: <= {split_year}, Test: {split_year+1}–{max_yr})"

    print(f"      Validation Strategy: {val_strategy}")
    print(f"      Training Samples: {len(X_train):,} | Test Samples: {len(X_test):,}")

    # Build and Fit Preprocessor
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # 4. Train Random Forest Classifier
    print("[4/5] Training RandomForestClassifier (n_estimators=150, balanced subsampling)...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_split=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_trans, y_train)

    # 5. Comprehensive Metric Evaluation
    print("[5/5] Computing multi-metric evaluation & confusion matrix...")
    y_pred = clf.predict(X_test_trans)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, target_names=classes_list, output_dict=True, zero_division=0)

    # Class sample counts
    class_distribution = {}
    for c_idx, c_name in enumerate(classes_list):
        class_distribution[c_name] = {
            "train_samples": int((y_train == c_idx).sum()),
            "test_samples": int((y_test == c_idx).sum()),
            "precision": round(float(clf_report[c_name]["precision"] * 100), 2),
            "recall": round(float(clf_report[c_name]["recall"] * 100), 2),
            "f1_score": round(float(clf_report[c_name]["f1-score"] * 100), 2)
        }

    print("\n---------------- MODEL PERFORMANCE REPORT ----------------")
    print(f" Validation Strategy   : {val_strategy}")
    print(f" Overall Accuracy      : {acc * 100:.2f}%")
    print(f" Balanced Accuracy     : {bal_acc * 100:.2f}%")
    print(f" Macro Precision       : {prec_macro * 100:.2f}%")
    print(f" Macro Recall          : {rec_macro * 100:.2f}%")
    print(f" Macro F1-Score        : {f1_macro * 100:.2f}%")
    print(f" Weighted F1-Score     : {f1_weighted * 100:.2f}%")
    print("----------------------------------------------------------\n")

    # 6. Serialize Artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, os.path.join("models", "attack_classifier.joblib"))
    joblib.dump(preprocessor, os.path.join("models", "preprocessor.joblib"))
    joblib.dump(label_encoder, os.path.join("models", "label_encoder.joblib"))

    metadata = {
        "model_name": "RandomForestClassifier",
        "problem_type": "Multi-Class Incident Tactic Classification",
        "validation_strategy": val_strategy,
        "training_timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(float(acc * 100), 2),
        "balanced_accuracy": round(float(bal_acc * 100), 2),
        "macro_precision": round(float(prec_macro * 100), 2),
        "macro_recall": round(float(rec_macro * 100), 2),
        "macro_f1": round(float(f1_macro * 100), 2),
        "weighted_f1": round(float(f1_weighted * 100), 2),
        "input_features": CATEGORICAL_FEATURES + NUMERICAL_FEATURES,
        "target_variable": "attack_type",
        "target_classes": classes_list,
        "confusion_matrix": cm,
        "class_distribution": class_distribution,
        "temporal_coverage": f"{min_yr} – {max_yr}",
        "limitations": [
            "Evaluates pre-event categorical and tactical signals; does not predict exact timing or coordinates.",
            "Historical reporting density variations across decades affect baseline class distributions.",
            "Feature importances represent statistical model reliance and must not be interpreted as direct causal drivers."
        ]
    }

    with open(os.path.join("models", "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] All model artifacts & verification telemetry saved to ./models/.")

if __name__ == "__main__":
    main()