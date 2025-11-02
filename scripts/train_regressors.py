"""Train multiple regression models for AQI prediction and export them as joblib artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

DEFAULT_MODELS: Dict[str, object] = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
}

NUMERIC_FEATURES: List[str] = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene", "Year", "Month", "Day", "DayOfWeek",
    "Hour", "PM2.5_NO2"
]

CATEGORICAL_FEATURES: List[str] = ["City"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multiple regression models for AQI prediction")
    parser.add_argument("--input", default="notebooks/aqi_dataset_engineered.csv", help="Training dataset with engineered features")
    parser.add_argument("--output-dir", default="models/regression", help="Directory to save trained models")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out ratio for test split")
    return parser.parse_args()


def load_dataset(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    df = pd.read_csv(path)
    if "AQI" not in df.columns:
        raise ValueError("Dataset must contain 'AQI' column")
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def train_models(df: pd.DataFrame, output_dir: Path, test_size: float) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    preprocessor = build_preprocessor()

    for model_name, model in DEFAULT_MODELS.items():
        print(f"Training {model_name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        joblib.dump(pipeline, output_dir / f"{model_name}.pkl")
        print(f"Saved {model_name}.pkl")

        results.append({
            "model": model_name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "regression_results.csv", index=False)
    return results_df


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)
    results_df = train_models(df, Path(args.output_dir), args.test_size)
    print("Training complete. Results:")
    print(results_df)


if __name__ == "__main__":
    main()
