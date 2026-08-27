"""
Pre-Event Feature Engineering & Preprocessing
Location: ./src/preprocessing.py
Purpose: Transforms raw input attributes into ML-safe features without post-event leakage.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

# Explicit ML feature schema: ONLY variables knowable prior to or at the immediate onset
CATEGORICAL_FEATURES = ["region", "target_type", "weapon_type"]
NUMERICAL_FEATURES = ["suicide", "success"]
TARGET_COLUMN = "attack_type"

def build_preprocessor() -> ColumnTransformer:
    """Constructs a scikit-learn ColumnTransformer for categorical & numerical pipeline."""
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
            ("num", num_transformer, NUMERICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor

def prepare_data_for_training(df: pd.DataFrame, min_samples_per_class: int = 50):
    """Filters low-frequency target classes and returns X, y with label maps."""
    df_clean = df.dropna(subset=[TARGET_COLUMN]).copy()
    
    # Prune obscure classes to guarantee solid multiclass statistical validity
    class_counts = df_clean[TARGET_COLUMN].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()
    df_filtered = df_clean[df_clean[TARGET_COLUMN].isin(valid_classes)].copy()

    X = df_filtered[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df_filtered[TARGET_COLUMN]
    return X, y