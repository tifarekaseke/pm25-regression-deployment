"""Reusable training utilities for the PM2.5 regression project."""

from __future__ import annotations

import json
import math
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip"
)
TARGET_COLUMN = "PM2.5"
NUMERIC_FEATURES = [
    "month",
    "hour",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]
CATEGORICAL_FEATURES = ["wd", "station"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


class BatchGradientDescentRegressor(BaseEstimator, RegressorMixin):
    """Multiple linear regression optimized using full-batch gradient descent.

    This estimator follows the scikit-learn estimator interface so it can be
    placed inside a Pipeline and saved safely with Joblib.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 150,
        l2_penalty: float = 0.0001,
        tolerance: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2_penalty = l2_penalty
        self.tolerance = tolerance

    def fit(self, X: Any, y: Any) -> "BatchGradientDescentRegressor":
        X_checked, y_checked = check_X_y(
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            y_numeric=True,
            dtype=np.float64,
        )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be at least 1.")
        if self.l2_penalty < 0:
            raise ValueError("l2_penalty cannot be negative.")

        n_samples, n_features = X_checked.shape
        self.coef_ = np.zeros(n_features, dtype=np.float64)
        self.intercept_ = 0.0
        self.loss_history_: list[float] = []
        previous_loss = float("inf")

        for _ in range(self.n_epochs):
            predictions = X_checked @ self.coef_ + self.intercept_
            errors = predictions - y_checked
            mse = float(np.mean(errors**2))
            regularization = float(self.l2_penalty * np.sum(self.coef_**2))
            loss = mse + regularization
            self.loss_history_.append(loss)

            gradient_weights = (
                (2.0 / n_samples) * (X_checked.T @ errors)
                + 2.0 * self.l2_penalty * self.coef_
            )
            gradient_intercept = 2.0 * float(np.mean(errors))

            self.coef_ -= self.learning_rate * gradient_weights
            self.intercept_ -= self.learning_rate * gradient_intercept

            if not np.isfinite(loss):
                raise FloatingPointError(
                    "Batch gradient descent diverged. Reduce the learning rate."
                )
            if abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss

        self.n_features_in_ = n_features
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X_checked = check_array(
            X,
            accept_sparse=False,
            ensure_2d=True,
            dtype=np.float64,
        )
        return X_checked @ self.coef_ + self.intercept_


@dataclass
class TrainingResult:
    best_model_name: str
    metrics: dict[str, dict[str, float]]
    model_path: Path
    metadata_path: Path
    sample_size: int
    best_batch_gd_params: dict[str, Any]
    best_sgd_params: dict[str, Any]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def download_dataset(data_dir: Path | None = None) -> Path:
    """Download and extract the UCI dataset if it is not present."""
    data_dir = data_dir or project_root() / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "beijing_air_quality.zip"
    extracted = data_dir / "PRSA_Data_20130301-20170228"

    if not extracted.exists():
        if not zip_path.exists():
            print(f"Downloading dataset to {zip_path} ...")
            with urllib.request.urlopen(DATASET_URL, timeout=180) as response, zip_path.open("wb") as target:
                shutil.copyfileobj(response, target)
        print("Extracting dataset ...")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(data_dir)

    return extracted


def load_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    extracted = download_dataset(data_dir)
    csv_files = sorted(extracted.rglob("PRSA_Data_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No station CSV files found under {extracted}")
    frames = [pd.read_csv(path) for path in csv_files]
    return pd.concat(frames, ignore_index=True)


def validate_training_frame(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {missing}")


def prepare_frame(df: pd.DataFrame, sample_size: int | None = 120_000) -> pd.DataFrame:
    validate_training_frame(df)
    clean = df.copy()
    clean = clean.dropna(subset=[TARGET_COLUMN])
    clean = clean[REQUIRED_COLUMNS]
    clean = clean[clean[TARGET_COLUMN].between(0, 1000)]

    if sample_size and len(clean) > sample_size:
        clean = clean.sample(n=sample_size, random_state=42)
    return clean.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _fit_gradient_descent_models_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[Pipeline, dict[str, Any], Pipeline, dict[str, Any]]:
    """Tune and fit both required gradient-descent linear regressors."""
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_validation_t = preprocessor.transform(X_validation)

    # A smaller deterministic subset keeps full-batch tuning practical while
    # the selected configuration is subsequently refitted on all training rows.
    rng = np.random.default_rng(42)
    tune_count = min(25_000, len(X_train_t))
    tune_indices = rng.choice(len(X_train_t), size=tune_count, replace=False)
    X_batch_tune = X_train_t[tune_indices]
    y_batch_tune = y_train.iloc[tune_indices]

    batch_grid = list(
        ParameterGrid(
            {
                "learning_rate": [0.003, 0.01],
                "l2_penalty": [0.0, 0.0001],
            }
        )
    )
    best_batch_params: dict[str, Any] | None = None
    best_batch_rmse = float("inf")
    for params in batch_grid:
        candidate = BatchGradientDescentRegressor(
            n_epochs=120,
            tolerance=1e-7,
            **params,
        )
        candidate.fit(X_batch_tune, y_batch_tune)
        rmse = math.sqrt(
            mean_squared_error(y_validation, candidate.predict(X_validation_t))
        )
        if rmse < best_batch_rmse:
            best_batch_rmse = rmse
            best_batch_params = params

    assert best_batch_params is not None
    final_batch_model = BatchGradientDescentRegressor(
        n_epochs=150,
        tolerance=1e-8,
        **best_batch_params,
    )
    final_batch_model.fit(X_train_t, y_train)
    batch_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", final_batch_model)]
    )

    sgd_grid = list(
        ParameterGrid(
            {
                "alpha": [0.0001, 0.001],
                "eta0": [0.001, 0.01],
                "learning_rate": ["invscaling", "adaptive"],
            }
        )
    )
    best_sgd_model: SGDRegressor | None = None
    best_sgd_params: dict[str, Any] | None = None
    best_sgd_rmse = float("inf")

    for params in sgd_grid:
        candidate = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            max_iter=1500,
            tol=1e-3,
            random_state=42,
            average=True,
            **params,
        )
        candidate.fit(X_train_t, y_train)
        rmse = math.sqrt(
            mean_squared_error(y_validation, candidate.predict(X_validation_t))
        )
        if rmse < best_sgd_rmse:
            best_sgd_rmse = rmse
            best_sgd_model = candidate
            best_sgd_params = params

    assert best_sgd_model is not None and best_sgd_params is not None
    sgd_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", best_sgd_model)]
    )
    return batch_pipeline, best_batch_params, sgd_pipeline, best_sgd_params


def train_models_from_dataframe(
    df: pd.DataFrame,
    model_dir: Path | None = None,
    sample_size: int | None = 120_000,
) -> TrainingResult:
    """Train four regressors, select the lowest-test-RMSE model, and save it."""
    clean = prepare_frame(df, sample_size=sample_size)
    X = clean[FEATURE_COLUMNS]
    y = clean[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    (
        batch_pipeline,
        best_batch_gd_params,
        sgd_pipeline,
        best_sgd_params,
    ) = _fit_gradient_descent_models_with_tuning(
        X_train, y_train, X_validation, y_validation
    )

    models: dict[str, Pipeline] = {
        "Batch Gradient Descent Linear Regression": batch_pipeline,
        "Stochastic Gradient Descent Linear Regression": sgd_pipeline,
        "Decision Tree": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=20,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest (Ensemble)": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=60,
                        max_depth=16,
                        min_samples_leaf=5,
                        max_features=0.7,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    metrics: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in models.items():
        if name in {"Decision Tree", "Random Forest (Ensemble)"}:
            model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics[name] = regression_metrics(y_test, predictions)
        fitted[name] = model

    best_name = min(metrics, key=lambda name: metrics[name]["rmse"])
    best_model = fitted[best_name]

    model_dir = model_dir or project_root() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.joblib"
    metadata_path = model_dir / "model_metadata.json"
    joblib.dump(best_model, model_path, compress=3)

    metadata = {
        "best_model_name": best_name,
        "target": TARGET_COLUMN,
        "unit": "µg/m³",
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": metrics,
        "best_batch_gd_params": best_batch_gd_params,
        "best_sgd_params": best_sgd_params,
        "training_rows": len(clean),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainingResult(
        best_model_name=best_name,
        metrics=metrics,
        model_path=model_path,
        metadata_path=metadata_path,
        sample_size=len(clean),
        best_batch_gd_params=best_batch_gd_params,
        best_sgd_params=best_sgd_params,
    )


def train_from_default_dataset(sample_size: int | None = 120_000) -> TrainingResult:
    return train_models_from_dataframe(load_dataset(), sample_size=sample_size)
