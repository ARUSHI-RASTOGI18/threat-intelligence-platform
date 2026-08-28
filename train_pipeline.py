"""
Multi-Model Machine Learning Training & Temporal Validation Pipeline
Location: ./train_pipeline.py
Execution: python train_pipeline.py
"""

import os
import json
import joblib
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

from src.data_loader import load_analytical_data
from src.preprocessing import build_preprocessor, prepare_data_for_training, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_attack_classifier.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

def main():
    print("\n=======================================================")
    print("      GTI-ARP MULTI-MODEL ML BENCHMARK PIPELINE        ")
    print("=======================================================")

    print("[1/6] Ingesting dataset...")
    df = load_analytical_data(force_reload=True)
    min_yr = int(df["year"].min())
    max_yr = int(df["year"].max())
    print(f"      Dataset: {len(df):,} records | Horizon: {min_yr} – {max_yr}")

    print("[2/6] Preparing pre-event feature space (no outcome leakage)...")
    X, y = prepare_data_for_training(df, min_samples_per_class=50)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes_list = label_encoder.classes_.tolist()

    print("[3/6] Configuring Temporal Train / Validation / Test Splits...")
    train_end_yr = max_yr - 6
    val_end_yr = max_yr - 3

    train_mask = df.loc[X.index, "year"] <= train_end_yr
    val_mask = (df.loc[X.index, "year"] > train_end_yr) & (df.loc[X.index, "year"] <= val_end_yr)
    test_mask = df.loc[X.index, "year"] > val_end_yr

    if val_mask.sum() < 200 or test_mask.sum() < 200:
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
        val_strategy = "Stratified Random Holdout (70/15/15)"
        split_periods = {"train": f"{min_yr}–{max_yr}", "val": f"{min_yr}–{max_yr}", "test": f"{min_yr}–{max_yr}"}
    else:
        X_train, y_train = X[train_mask], y_encoded[train_mask]
        X_val, y_val = X[val_mask], y_encoded[val_mask]
        X_test, y_test = X[test_mask], y_encoded[test_mask]
        val_strategy = f"Temporal Out-of-Time (Train: <= {train_end_yr}, Val: {train_end_yr+1}–{val_end_yr}, Test: {val_end_yr+1}–{max_yr})"
        split_periods = {"train": f"{min_yr}–{train_end_yr}", "val": f"{train_end_yr+1}–{val_end_yr}", "test": f"{val_end_yr+1}–{max_yr}"}

    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    print("[4/6] Training and benchmarking multi-model suite...")
    models = {
        "Dummy Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (L2 Regularized)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(max_iter=100, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_split=4, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    }

    benchmark_results = []
    models_evaluation = {}
    trained_models = {}

    for name, clf in models.items():
        clf.fit(X_train_trans, y_train)
        
        # Validation evaluation
        y_val_pred = clf.predict(X_val_trans)
        v_acc = accuracy_score(y_val, y_val_pred)
        v_bal_acc = balanced_accuracy_score(y_val, y_val_pred)
        v_macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        v_weighted_f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

        # Test evaluation
        y_test_pred = clf.predict(X_test_trans)
        t_acc = accuracy_score(y_test, y_test_pred)
        t_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
        t_macro_prec = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
        t_macro_rec = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
        t_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
        t_weighted_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
        cm_test = confusion_matrix(y_test, y_test_pred).tolist()

        benchmark_results.append({
            "Model": name,
            "Accuracy (%)": round(float(v_acc * 100), 2),
            "Balanced Accuracy (%)": round(float(v_bal_acc * 100), 2),
            "Macro F1 (%)": round(float(v_macro_f1 * 100), 2),
            "Weighted F1 (%)": round(float(v_weighted_f1 * 100), 2)
        })

        models_evaluation[name] = {
            "validation_accuracy": round(float(v_acc * 100), 2),
            "test_metrics": {
                "accuracy": round(float(t_acc * 100), 2),
                "balanced_accuracy": round(float(t_bal_acc * 100), 2),
                "macro_precision": round(float(t_macro_prec * 100), 2),
                "macro_recall": round(float(t_macro_rec * 100), 2),
                "macro_f1": round(float(t_macro_f1 * 100), 2),
                "weighted_f1": round(float(t_weighted_f1 * 100), 2)
            },
            "confusion_matrix": cm_test
        }
        trained_models[name] = clf

    benchmark_df = pd.DataFrame(benchmark_results).sort_values("Macro F1 (%)", ascending=False).reset_index(drop=True)
    best_model_name = benchmark_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print("[5/6] Computing Permutation Importance for Selected Best Model...")
    perm_sample_size = min(2000, len(X_test_trans))
    perm_res = permutation_importance(best_model, X_test_trans[:perm_sample_size], y_test[:perm_sample_size], n_repeats=3, random_state=42, n_jobs=-1)
    
    feature_names = preprocessor.get_feature_names_out()
    perm_df = pd.DataFrame({"raw_feature": feature_names, "importance": perm_res.importances_mean})
    
    def root_name(col: str) -> str:
        clean = col.replace("cat__", "").replace("num__", "")
        for base in ["weapon_type", "target_type", "region", "suicide", "success"]:
            if clean.startswith(base):
                return base.replace("_", " ").title()
        return "Other Context"

    perm_df["root_feature"] = perm_df["raw_feature"].apply(root_name)
    perm_agg = perm_df.groupby("root_feature")["importance"].sum().clip(lower=0.0).reset_index()
    perm_total = perm_agg["importance"].sum()
    perm_agg["importance_pct"] = ((perm_agg["importance"] / (perm_total if perm_total > 0 else 1.0)) * 100).round(2)
    perm_sorted = perm_agg.sort_values("importance_pct", ascending=False).to_dict(orient="records")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    metadata = {
        "best_model_name": best_model_name,
        "benchmark_comparison": benchmark_results,
        "models_evaluation": models_evaluation,
        "validation_strategy": val_strategy,
        "split_periods": split_periods,
        "training_timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "best_model_metrics": models_evaluation[best_model_name]["test_metrics"],
        "target_classes": classes_list,
        "confusion_matrix": models_evaluation[best_model_name]["confusion_matrix"],
        "permutation_importance": perm_sorted,
        "temporal_coverage": f"{min_yr} – {max_yr}"
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[SUCCESS] Pipeline Complete. Model artifacts and multi-model metadata saved to ./models/.")

if __name__ == "__main__":
    main()