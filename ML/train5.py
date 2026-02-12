import os
from datetime import datetime
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    HAS_PLOTS = True
except Exception:
    plt = None
    HAS_PLOTS = False
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")


def load_data() -> pd.DataFrame:
    data_path = Path("data") / "rent_processed.csv"
    df = pd.read_csv(data_path)
    print(f"Rent Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nBasic statistics:")
    print(df.describe())
    return df


def prepare_features(df: pd.DataFrame):
    target_col = "price"
    X = df.drop(columns=[target_col, "price_normalized", "property_type_cluster"])
    y = df[target_col]

    numeric_cols = ["surface", "rooms", "bathrooms"]
    categorical_cols = [
        "region",
        "property_type",
        "city",
        "price_segment",
        "has_piscine",
        "has_garage",
        "has_jardin",
        "has_terrasse",
        "has_ascenseur",
        "is_meuble",
        "has_chauffage",
        "has_climatisation",
    ]

    print(f"Target column: {target_col}")
    print(f"Number of features: {X.shape[1]}")
    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")

    return X, y, numeric_cols, categorical_cols, target_col


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    print("✓ Preprocessing pipeline created")
    return preprocessor


def build_models():
    models = {
        "Ridge": Ridge(random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, random_state=42
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        print("✓ XGBoost added")
    except Exception as exc:
        print(f"⚠ XGBoost not available: {exc}")

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
        )
        print("✓ LightGBM added")
    except Exception as exc:
        print(f"⚠ LightGBM not available: {exc}")

    print(f"\nTotal models to train: {len(models)}")
    print(f"Models: {list(models.keys())}")

    return models


def train_models(models, preprocessor, X_train, y_train):
    print("Training models...\n")
    trained_pipelines = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...", end=" ")

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline

        print("✓")

    print(f"\n✓ All {len(models)} models trained successfully")
    return trained_pipelines


def evaluate_models(trained_pipelines, X_test, y_test):
    print("Evaluating models on test set...\n")
    results = []

    for model_name, pipeline in trained_pipelines.items():
        preds = pipeline.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append(
            {
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "R²": r2,
            }
        )

        print(
            f"{model_name:20} | RMSE: {rmse:12,.2f} | MAE: {mae:12,.2f} | R²: {r2:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print("\n" + "=" * 70)
    print("Sorted by RMSE (best at top):")
    print("=" * 70)
    print(results_df.to_string(index=False))

    return results_df


def visualize_results(results_df, trained_pipelines, X_test, y_test):
    if not HAS_PLOTS:
        print("⚠ Skipping plots (matplotlib not available)")
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    results_sorted = results_df.sort_values("RMSE")

    ax1 = axes[0, 0]
    colors = ["green" if i == 0 else "steelblue" for i in range(len(results_sorted))]
    ax1.barh(results_sorted["Model"], results_sorted["RMSE"], color=colors)
    ax1.set_xlabel("RMSE (Root Mean Squared Error)")
    ax1.set_title("rent Model Comparison: RMSE (Lower is Better)")
    ax1.invert_yaxis()
    for i, v in enumerate(results_sorted["RMSE"]):
        ax1.text(v, i, f" {v:,.0f}", va="center")

    ax2 = axes[0, 1]
    results_mae = results_df.sort_values("MAE")
    colors = [
        "green" if results_mae.iloc[i]["Model"] == results_sorted.iloc[0]["Model"] else "steelblue"
        for i in range(len(results_mae))
    ]
    ax2.barh(results_mae["Model"], results_mae["MAE"], color=colors)
    ax2.set_xlabel("MAE (Mean Absolute Error)")
    ax2.set_title("Rent Model Comparison: MAE (Lower is Better)")
    ax2.invert_yaxis()
    for i, v in enumerate(results_mae["MAE"]):
        ax2.text(v, i, f" {v:,.0f}", va="center")

    ax3 = axes[1, 0]
    results_r2 = results_df.sort_values("R²", ascending=False)
    colors = [
        "green" if results_r2.iloc[i]["Model"] == results_sorted.iloc[0]["Model"] else "steelblue"
        for i in range(len(results_r2))
    ]
    ax3.barh(results_r2["Model"], results_r2["R²"], color=colors)
    ax3.set_xlabel("R² Score")
    ax3.set_title("rent Model Comparison: R² (Higher is Better)")
    ax3.invert_yaxis()
    for i, v in enumerate(results_r2["R²"]):
        ax3.text(v, i, f" {v:.4f}", va="center")

    ax4 = axes[1, 1]
    ax4.axis("off")
    table_data = []
    for _, row in results_sorted.iterrows():
        table_data.append(
            [
                row["Model"],
                f"{row['RMSE']:,.0f}",
                f"{row['MAE']:,.0f}",
                f"{row['R²']:.4f}",
            ]
        )

    table = ax4.table(
        cellText=table_data,
        colLabels=["Model", "RMSE", "MAE", "R²"],
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    for i in range(4):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(4):
        table[(1, i)].set_facecolor("#90EE90")

    ax4.set_title("Performance Summary - RENT", fontweight="bold", pad=20)

    plt.tight_layout()
    plt.show()

    best_model_name = results_sorted.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    best_preds = best_pipeline.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(y_test, best_preds, alpha=0.5, s=20)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax1.set_xlabel("Actual Price")
    ax1.set_ylabel("Predicted Price")
    ax1.set_title(f"Best Rent Model ({best_model_name}): Predictions vs Actual")
    ax1.grid(True, alpha=0.3)

    residuals = y_test - best_preds
    ax2 = axes[1]
    ax2.scatter(best_preds, residuals, alpha=0.5, s=20)
    ax2.axhline(y=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Predicted Price")
    ax2.set_ylabel("Residuals")
    ax2.set_title(f"Best RENT Model ({best_model_name}): Residual Plot")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_best_model(results_df, trained_pipelines):
    best_model_name = results_df.sort_values("RMSE").iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]

    base_dir = Path(__file__).resolve().parent
    output_dir = (base_dir.parent / "back" / "models" / "rent_model").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"

    joblib.dump(best_pipeline, model_path)
    print(f"✓ Best model saved to: {model_path}")


def log_with_mlflow(trained_pipelines, X_test, y_test, X_train, target_col, experiment_name):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    if tracking_uri.startswith("file:"):
        print(f"MLflow tracking directory: {Path('mlruns').resolve()}")

    print("Logging RENT experiments to MLflow...\n")

    for model_name, pipeline in trained_pipelines.items():
        with mlflow.start_run(run_name=model_name):
            preds = pipeline.predict(X_test)

            rmse = root_mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            mlflow.log_param("model", model_name)
            mlflow.log_param("dataset", "rent_processed")
            mlflow.log_param("target_col", target_col)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(pipeline, "model")

            print(f"✓ {model_name} logged to MLflow")

    print("\n✓ All RENT models logged successfully!")
    print("\nTo view MLflow UI, run: mlflow ui")


def main():
    df = load_data()
    X, y, numeric_cols, categorical_cols, target_col = prepare_features(df)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(
        f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)"
    )
    print(
        f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)"
    )

    models = build_models()
    trained_pipelines = train_models(models, preprocessor, X_train, y_train)
    results_df = evaluate_models(trained_pipelines, X_test, y_test)
    visualize_results(results_df, trained_pipelines, X_test, y_test)
    save_best_model(results_df, trained_pipelines)
    experiment_name = f"real_estate_rent_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_with_mlflow(trained_pipelines, X_test, y_test, X_train, target_col, experiment_name)


if __name__ == "__main__":
    main()
