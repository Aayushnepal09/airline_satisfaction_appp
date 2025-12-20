# housing_pipeline.py
"""Shared ML pipeline components for the project.

IMPORTANT:
- This file name is intentionally kept as `housing_pipeline.py` to match the
  starter repository structure and to provide a stable module path for joblib
  pickles.
- Even though the dataset is airline passenger satisfaction, keeping the module
  name stable avoids unpickling issues.

This module contains:
- Preprocessing builder (ColumnTransformer)
- Optional PCA support (inside numeric pipeline)
- Estimator factory for the required models

Dataset: Airline Passenger Satisfaction (binary classification)
Target: satisfaction_binary (1 = satisfied, 0 = neutral/dissatisfied)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =============================================================================
# Columns
# =============================================================================

CATEGORICAL_COLS: List[str] = [
    "gender",
    "customer_type",
    "type_of_travel",
    "travel_class",
]

NUMERIC_COLS: List[str] = [
    "age",
    "flight_distance",
    "inflight_wifi_service",
    "departure_arrival_time_convenient",
    "ease_of_online_booking",
    "gate_location",
    "food_and_drink",
    "online_boarding",
    "seat_comfort",
    "inflight_entertainment",
    "on_board_service",
    "leg_room_service",
    "baggage_handling",
    "checkin_service",
    "inflight_service",
    "cleanliness",
    "departure_delay_minutes",
    "arrival_delay_minutes",
]

# numeric columns to log-transform (skewed, non-negative)
SKEW_COLS: List[str] = [
    "flight_distance",
    "departure_delay_minutes",
    "arrival_delay_minutes",
]


# =============================================================================
# Custom helpers
# =============================================================================

def safe_log1p(x: np.ndarray) -> np.ndarray:
    """Safe log1p for non-negative numeric arrays."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, a_min=0, a_max=None)
    return np.log1p(x)


# =============================================================================
# Preprocessing
# =============================================================================


def build_preprocessing(*, use_pca: bool = False, pca_components: float = 0.95) -> ColumnTransformer:
    """Build the preprocessing ColumnTransformer.

    - Categorical: most_frequent impute + OneHotEncoder
    - Numeric:
        * SKEW_COLS: median impute + log1p + standardize (+ optional PCA)
        * Other numeric: median impute + standardize (+ optional PCA)

    PCA is applied *inside the numeric pipelines* (so categorical one-hot features
    are not reduced). This keeps the meaning of categorical features and avoids
    PCA on sparse/one-hot blocks.
    """

    # Split numeric cols into skew vs other
    skew_cols = [c for c in SKEW_COLS if c in NUMERIC_COLS]
    other_num_cols = [c for c in NUMERIC_COLS if c not in skew_cols]

    skew_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(safe_log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ]
    other_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    if use_pca:
        skew_steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
        other_steps.append(("pca", PCA(n_components=pca_components, random_state=42)))

    skew_pipeline = Pipeline(skew_steps)
    other_pipeline = Pipeline(other_steps)

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num_skew", skew_pipeline, skew_cols),
            ("num", other_pipeline, other_num_cols),
            ("cat", cat_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessing


# =============================================================================
# Estimator factory
# =============================================================================


def make_estimator_for_name(name: str):
    """Return a classifier instance given a short model name."""
    name = name.strip().lower()

    if name in {"logreg", "logistic", "logistic_regression"}:
        return LogisticRegression(max_iter=2000, n_jobs=-1)

    if name == "ridge":
        return RidgeClassifier()

    if name in {"histgradientboosting", "hgb"}:
        return HistGradientBoostingClassifier(random_state=42)

    if name in {"xgboost", "xgb"}:
        return XGBClassifier(
            objective="binary:logistic",
            random_state=42,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="logloss",
        )

    if name in {"lightgbm", "lgbm"}:
        return LGBMClassifier(
            random_state=42,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            verbose=-1,
        )

    raise ValueError(f"Unknown model name: {name}")


def build_pipeline(model_name: str, *, use_pca: bool = False, pca_components: float = 0.95) -> Pipeline:
    """Convenience helper: preprocessing + estimator."""
    preprocessing = build_preprocessing(use_pca=use_pca, pca_components=pca_components)
    estimator = make_estimator_for_name(model_name)
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])
