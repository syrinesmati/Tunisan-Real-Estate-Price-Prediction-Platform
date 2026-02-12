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

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    sns = None
    HAS_SEABORN = False

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in columns",
)

CV_FOLDS = 3
N_ITER = 10
SKIP_TUNING = True


def _create_onehot_encoder():
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def load_data() -> pd.DataFrame:
    data_path = Path("data") / "sale_processed.csv"
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nBasic statistics:")
    print(df.describe())

    return df


def analyze_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if HAS_PLOTS:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].hist(df["price"], bins=50, edgecolor="black")
        axes[0, 0].set_title("Price Distribution")
        axes[0, 0].set_xlabel("Price (TND)")
        axes[0, 0].set_ylabel("Frequency")

        axes[0, 1].boxplot(df["price"])
        axes[0, 1].set_title("Price Boxplot (Outlier Detection)")
        axes[0, 1].set_ylabel("Price (TND)")

        axes[1, 0].hist(df["surface"], bins=50, edgecolor="black")
        axes[1, 0].set_title("Surface Distribution")
        axes[1, 0].set_xlabel("Surface (m²)")
        axes[1, 0].set_ylabel("Frequency")

        axes[1, 1].scatter(df["surface"], df["price"], alpha=0.5)
        axes[1, 1].set_title("Price vs Surface")
        axes[1, 1].set_xlabel("Surface (m²)")
        axes[1, 1].set_ylabel("Price (TND)")

        plt.tight_layout()
        plt.show()

    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold_low = Q1 - 3 * IQR
    outlier_threshold_high = Q3 + 3 * IQR

    outliers = df[(df["price"] < outlier_threshold_low) | (df["price"] > outlier_threshold_high)]
    print(f"\nOutliers detected: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"Outlier threshold: [{outlier_threshold_low:.0f}, {outlier_threshold_high:.0f}]")

    print("\nSuspicious patterns:")
    df["price_per_sqm"] = df["price"] / df["surface"]
    suspicious = df[df["price_per_sqm"] > df["price_per_sqm"].quantile(0.99)]
    print(f"Very high price per m²: {len(suspicious)} properties")
    print(suspicious[["region", "price", "surface", "rooms", "property_type", "price_per_sqm"]].head(10))

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy()

    leakage_features = ["price_normalized"]
    print("Removing data leakage features:")
    for col in leakage_features:
        if col in df_fe.columns:
            print(f"  - {col}")
            df_fe = df_fe.drop(columns=[col])

    print("\nCreating engineered features:")

    df_fe["surface_per_room"] = df_fe["surface"] / df_fe["rooms"].replace(0, 1)
    df_fe["bathroom_ratio"] = df_fe["bathrooms"] / df_fe["rooms"].replace(0, 1)
    print("  ✓ Space features: surface_per_room, bathroom_ratio, total_rooms")

    amenity_cols = [
        "has_piscine",
        "has_garage",
        "has_jardin",
        "has_terrasse",
        "has_ascenseur",
        "has_chauffage",
        "has_climatisation",
    ]
    df_fe["amenity_score"] = df_fe[amenity_cols].sum(axis=1)
    df_fe["luxury_score"] = (
        df_fe["has_piscine"].astype(int) * 3
        + df_fe["has_jardin"].astype(int) * 2
        + df_fe["has_garage"].astype(int) * 2
        + df_fe["has_climatisation"].astype(int) * 1.5
        + df_fe["has_terrasse"].astype(int) * 1
        + df_fe["has_ascenseur"].astype(int) * 1
        + df_fe["has_chauffage"].astype(int) * 1
    )
    print("  ✓ Amenity features: amenity_score, luxury_score")

    df_fe["size_category"] = pd.cut(
        df_fe["surface"],
        bins=[0, 75, 120, 180, 1000],
        labels=["Small", "Medium", "Large", "Very_Large"],
    )
    print("  ✓ Categorical feature: size_category")

    df_fe["room_density"] = df_fe["rooms"] / df_fe["surface"].replace(0, 1)
    print("  ✓ Density feature: room_density")

    print(f"\nTotal features after engineering: {df_fe.shape[1]}")
    print(f"Original features: {df.shape[1]}")
    print(f"New features created: {df_fe.shape[1] - df.shape[1] + len(leakage_features)}")

    return df_fe


def correlation_analysis(df_fe: pd.DataFrame):
    if not HAS_PLOTS:
        print("⚠ Skipping correlation plots (matplotlib not available)")
        return
    if not HAS_SEABORN:
        print("⚠ Skipping correlation heatmap (seaborn not available)")

    numeric_features = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    if "price" in numeric_features:
        correlations = df_fe[numeric_features].corr()["price"].sort_values(ascending=False)
        print("Feature Correlations with Price:")
        print(correlations)

        plt.figure(figsize=(10, 8))
        top_features = correlations.abs().sort_values(ascending=False)[1:16]
        correlations[top_features.index].plot(kind="barh")
        plt.title("Top 15 Feature Correlations with Price")
        plt.xlabel("Correlation Coefficient")
        plt.tight_layout()
        plt.show()

        if HAS_SEABORN:
            engineered_cols = [
                "surface",
                "rooms",
                "bathrooms",
                "surface_per_room",
                "bathroom_ratio",
                "amenity_score",
                "luxury_score",
                "room_density",
                "price",
            ]
            available_cols = [col for col in engineered_cols if col in df_fe.columns]

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                df_fe[available_cols].corr(),
                annot=True,
                cmap="coolwarm",
                center=0,
                fmt=".2f",
                square=True,
            )
            plt.title("Correlation Matrix: Key Features")
            plt.tight_layout()
            plt.show()


def prepare_features(df_fe: pd.DataFrame):
    target_col = "price"
    y = df_fe[target_col]

    exclude_cols = [target_col, "price_per_sqm", "price_normalized"]
    if "price_per_sqm" in df_fe.columns:
        df_fe = df_fe.drop(columns=["price_per_sqm"])
    if "price_normalized" in df_fe.columns:
        df_fe = df_fe.drop(columns=["price_normalized"])

    feature_cols = [col for col in df_fe.columns if col not in exclude_cols]
    X = df_fe[feature_cols].copy()

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeatures to use ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col} ({X[col].dtype})")

    return X, y, feature_cols, target_col


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    print(f"Numeric features ({len(numeric_features)}): {numeric_features[:10]}...")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _create_onehot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    print("\n✓ Preprocessing pipeline created")
    return preprocessor, numeric_features, categorical_features


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print("\nTarget variable statistics:")
    print(f"  Train - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    print(f"  Test  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")

    return X_train, X_test, y_train, y_test


def train_baseline_models(preprocessor, X_train, X_test, y_train, y_test):
    baseline_models = {
        "Ridge": Ridge(random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
    }

    try:
        from xgboost import XGBRegressor

        baseline_models["XGBoost"] = XGBRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    except Exception as exc:
        print(f"⚠ XGBoost not available: {exc}")

    try:
        from lightgbm import LGBMRegressor

        baseline_models["LightGBM"] = LGBMRegressor(
            n_estimators=100, random_state=42, verbose=-1, n_jobs=-1
        )
    except Exception as exc:
        print(f"⚠ LightGBM not available: {exc}")

    baseline_results = []
    baseline_pipelines = {}

    print("Training baseline models...\n")

    for name, model in baseline_models.items():
        print(f"Training {name}...")

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        baseline_results.append(
            {
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R²": r2,
                "MAPE (%)": mape,
                "CV R² Mean": cv_mean,
                "CV R² Std": cv_std,
            }
        )

        baseline_pipelines[name] = pipeline

        print(
            f"  ✓ {name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, CV R²: {cv_mean:.4f} (±{cv_std:.4f})\n"
        )

    baseline_df = pd.DataFrame(baseline_results)
    baseline_df = baseline_df.sort_values("R²", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("BASELINE MODEL RESULTS (No Hyperparameter Tuning)")
    print("=" * 80)
    print(baseline_df.to_string(index=False))
    print("=" * 80)

    return baseline_df, baseline_pipelines


def tune_models(preprocessor, X_train, y_train):
    tuned_models = {}
    tuning_info = {}

    print("Tuning Ridge Regression...")
    ridge_params = {"model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]}

    ridge_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", Ridge(random_state=42)),
        ]
    )

    ridge_grid = GridSearchCV(
        ridge_pipeline, ridge_params, cv=CV_FOLDS, scoring="r2", n_jobs=-1, verbose=1
    )
    ridge_grid.fit(X_train, y_train)
    print(f"Best Ridge params: {ridge_grid.best_params_}")
    print(f"Best Ridge CV score: {ridge_grid.best_score_:.4f}\n")

    tuned_models["Ridge (Tuned)"] = ridge_grid.best_estimator_
    tuning_info["Ridge (Tuned)"] = ridge_grid

    print("Tuning Random Forest (RandomizedSearchCV)...")
    rf_params = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 15, 20, 25, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    }

    rf_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    rf_random = RandomizedSearchCV(
        rf_pipeline,
        rf_params,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    rf_random.fit(X_train, y_train)
    print(f"Best RF params: {rf_random.best_params_}")
    print(f"Best RF CV score: {rf_random.best_score_:.4f}\n")

    tuned_models["RandomForest (Tuned)"] = rf_random.best_estimator_
    tuning_info["RandomForest (Tuned)"] = rf_random

    print("Tuning Gradient Boosting...")
    gb_params = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 5, 7],
        "model__min_samples_split": [2, 5, 10],
        "model__subsample": [0.8, 0.9, 1.0],
    }

    gb_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )

    gb_random = RandomizedSearchCV(
        gb_pipeline,
        gb_params,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    gb_random.fit(X_train, y_train)
    print(f"Best GB params: {gb_random.best_params_}")
    print(f"Best GB CV score: {gb_random.best_score_:.4f}\n")

    tuned_models["GradientBoosting (Tuned)"] = gb_random.best_estimator_
    tuning_info["GradientBoosting (Tuned)"] = gb_random

    try:
        from xgboost import XGBRegressor

        print("Tuning XGBoost...")
        xgb_params = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7, 9],
            "model__min_child_weight": [1, 3, 5],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.8, 0.9, 1.0],
        }

        xgb_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", XGBRegressor(random_state=42, n_jobs=-1)),
            ]
        )

        xgb_random = RandomizedSearchCV(
            xgb_pipeline,
            xgb_params,
            n_iter=N_ITER,
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        xgb_random.fit(X_train, y_train)
        print(f"Best XGBoost params: {xgb_random.best_params_}")
        print(f"Best XGBoost CV score: {xgb_random.best_score_:.4f}\n")

        tuned_models["XGBoost (Tuned)"] = xgb_random.best_estimator_
        tuning_info["XGBoost (Tuned)"] = xgb_random
    except Exception as exc:
        print(f"⚠ XGBoost tuning skipped: {exc}")

    try:
        from lightgbm import LGBMRegressor

        print("Tuning LightGBM...")
        lgbm_params = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7, -1],
            "model__num_leaves": [15, 31, 63],
            "model__min_child_samples": [10, 20, 30],
            "model__subsample": [0.8, 0.9, 1.0],
        }

        lgbm_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)),
            ]
        )

        lgbm_random = RandomizedSearchCV(
            lgbm_pipeline,
            lgbm_params,
            n_iter=N_ITER,
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        lgbm_random.fit(X_train, y_train)
        print(f"Best LightGBM params: {lgbm_random.best_params_}")
        print(f"Best LightGBM CV score: {lgbm_random.best_score_:.4f}\n")

        tuned_models["LightGBM (Tuned)"] = lgbm_random.best_estimator_
        tuning_info["LightGBM (Tuned)"] = lgbm_random
    except Exception as exc:
        print(f"⚠ LightGBM tuning skipped: {exc}")

    return tuned_models, tuning_info


def evaluate_tuned_models(tuned_models, X_train, X_test, y_train, y_test):
    tuned_results = []

    print("Evaluating tuned models on test set...\n")

    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        cv_scores = cross_val_score(
            model, X_train, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        tuned_results.append(
            {
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R²": r2,
                "MAPE (%)": mape,
                "CV R² Mean": cv_mean,
                "CV R² Std": cv_std,
            }
        )

        print(f"✓ {name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    tuned_df = pd.DataFrame(tuned_results)
    tuned_df = tuned_df.sort_values("R²", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("TUNED MODEL RESULTS")
    print("=" * 90)
    print(tuned_df.to_string(index=False))
    print("=" * 90)

    return tuned_df


def compare_baseline_tuned(baseline_df: pd.DataFrame, tuned_df: pd.DataFrame):
    comparison_data = []

    for _, row in baseline_df.iterrows():
        model_name = row["Model"]
        tuned_name = model_name + " (Tuned)"

        if tuned_name in tuned_df["Model"].values:
            tuned_row = tuned_df[tuned_df["Model"] == tuned_name].iloc[0]

            comparison_data.append(
                {
                    "Model": model_name,
                    "Baseline R²": row["R²"],
                    "Tuned R²": tuned_row["R²"],
                    "Improvement": tuned_row["R²"] - row["R²"],
                    "Baseline RMSE": row["RMSE"],
                    "Tuned RMSE": tuned_row["RMSE"],
                    "RMSE Reduction": row["RMSE"] - tuned_row["RMSE"],
                }
            )

    comparison_df = pd.DataFrame(comparison_data)

    print("\n" + "=" * 100)
    print("BASELINE vs TUNED COMPARISON")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("=" * 100)

    if HAS_PLOTS and not comparison_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(comparison_df))
        width = 0.35

        axes[0].bar(
            x - width / 2,
            comparison_df["Baseline R²"],
            width,
            label="Baseline",
            alpha=0.8,
        )
        axes[0].bar(
            x + width / 2,
            comparison_df["Tuned R²"],
            width,
            label="Tuned",
            alpha=0.8,
        )
        axes[0].set_xlabel("Model")
        axes[0].set_ylabel("R² Score")
        axes[0].set_title("R² Score: Baseline vs Tuned")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(comparison_df["Model"], rotation=45, ha="right")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(
            x - width / 2,
            comparison_df["Baseline RMSE"],
            width,
            label="Baseline",
            alpha=0.8,
        )
        axes[1].bar(
            x + width / 2,
            comparison_df["Tuned RMSE"],
            width,
            label="Tuned",
            alpha=0.8,
        )
        axes[1].set_xlabel("Model")
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("RMSE: Baseline vs Tuned (Lower is Better)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(comparison_df["Model"], rotation=45, ha="right")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return comparison_df


def feature_importance_analysis(tuned_df, tuned_models, numeric_features, categorical_features):
    if tuned_df.empty:
        return None, None

    best_model_name = tuned_df.iloc[0]["Model"]
    best_model = tuned_models[best_model_name]

    print(f"Analyzing feature importance for: {best_model_name}\n")

    if hasattr(best_model.named_steps["model"], "feature_importances_"):
        preprocessor_fitted = best_model.named_steps["preprocessor"]
        importances = best_model.named_steps["model"].feature_importances_

        feature_names = []
        if hasattr(preprocessor_fitted, "get_feature_names_out"):
            try:
                feature_names = preprocessor_fitted.get_feature_names_out().tolist()
            except Exception:
                feature_names = []

        if not feature_names:
            feature_names = numeric_features.copy()
            if categorical_features:
                cat_encoder = preprocessor_fitted.named_transformers_["cat"].named_steps["onehot"]
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
                feature_names.extend(cat_feature_names)

        if len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        print("Top 20 Most Important Features:")
        print(importance_df.head(20).to_string(index=False))

        if HAS_PLOTS:
            plt.figure(figsize=(10, 8))
            top_n = 15
            top_features = importance_df.head(top_n)
            plt.barh(range(top_n), top_features["Importance"])
            plt.yticks(range(top_n), top_features["Feature"])
            plt.xlabel("Importance")
            plt.title(f"Top {top_n} Feature Importances - {best_model_name}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        return best_model_name, best_model

    print("Could not extract feature importances (model does not expose feature_importances_)")
    return best_model_name, best_model


def detailed_diagnostics(best_model, best_model_name, X_test, y_test):
    if not HAS_PLOTS:
        print("⚠ Skipping diagnostics plots (matplotlib not available)")
        return

    y_pred_best = best_model.predict(X_test)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred_best, alpha=0.5, s=20)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax1.set_xlabel("Actual Price (TND)")
    ax1.set_ylabel("Predicted Price (TND)")
    ax1.set_title(f"{best_model_name}\nPredictions vs Actual")
    ax1.grid(True, alpha=0.3)

    r2_best = r2_score(y_test, y_pred_best)
    ax1.text(
        0.05,
        0.95,
        f"R² = {r2_best:.4f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    residuals = y_test - y_pred_best
    ax2 = axes[0, 1]
    ax2.scatter(y_pred_best, residuals, alpha=0.5, s=20)
    ax2.axhline(y=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Predicted Price (TND)")
    ax2.set_ylabel("Residuals (TND)")
    ax2.set_title("Residual Plot")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    ax3.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax3.axvline(x=0, color="r", linestyle="--", lw=2)
    ax3.set_xlabel("Residuals (TND)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Residuals Distribution")
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 0]
    price_ranges = pd.qcut(
        y_test, q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    abs_errors = np.abs(residuals)
    error_by_range = pd.DataFrame(
        {"Price Range": price_ranges, "Absolute Error": abs_errors}
    )
    error_by_range.boxplot(column="Absolute Error", by="Price Range", ax=ax4)
    ax4.set_xlabel("Price Range")
    ax4.set_ylabel("Absolute Error (TND)")
    ax4.set_title("Prediction Error by Price Range")
    plt.sca(ax4)
    plt.xticks(rotation=45)

    ax5 = axes[1, 1]
    percentage_errors = (residuals / y_test) * 100
    ax5.hist(percentage_errors, bins=50, edgecolor="black", alpha=0.7)
    ax5.axvline(x=0, color="r", linestyle="--", lw=2)
    ax5.set_xlabel("Percentage Error (%)")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Percentage Error Distribution")
    ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 2]
    try:
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=ax6)
        ax6.set_title("Q-Q Plot (Residuals Normality Check)")
        ax6.grid(True, alpha=0.3)
    except Exception as exc:
        ax6.set_title("Q-Q Plot unavailable")
        print(f"⚠ Q-Q Plot skipped: {exc}")

    plt.tight_layout()
    plt.show()

    print("\nDIAGNOSTIC STATISTICS")
    print("=" * 60)
    print(f"Mean Residual: {residuals.mean():.2f} TND")
    print(f"Std of Residuals: {residuals.std():.2f} TND")
    print(f"Mean Absolute Error: {np.abs(residuals).mean():.2f} TND")
    print(f"Median Absolute Error: {np.median(np.abs(residuals)):.2f} TND")
    print(f"Mean Percentage Error: {percentage_errors.mean():.2f}%")
    print(f"Median Percentage Error: {np.median(percentage_errors):.2f}%")
    print(
        f"\nPercentage of predictions within 10% of actual: {(np.abs(percentage_errors) <= 10).mean()*100:.1f}%"
    )
    print(
        f"Percentage of predictions within 20% of actual: {(np.abs(percentage_errors) <= 20).mean()*100:.1f}%"
    )
    print("=" * 60)


def performance_by_characteristics(X_test, y_test, y_pred_best, residuals):
    analysis_df = X_test.copy()
    analysis_df["actual_price"] = y_test.values
    analysis_df["predicted_price"] = y_pred_best
    analysis_df["error"] = residuals.values
    analysis_df["abs_error"] = np.abs(residuals.values)
    analysis_df["pct_error"] = (residuals.values / y_test.values) * 100

    if "property_type" in analysis_df.columns:
        print("\nPERFORMANCE BY PROPERTY TYPE")
        print("=" * 60)
        by_type = (
            analysis_df.groupby("property_type")
            .agg({
                "abs_error": ["mean", "median"],
                "pct_error": ["mean", "median"],
                "actual_price": "count",
            })
            .round(2)
        )
        by_type.columns = ["Mean AE", "Median AE", "Mean PE%", "Median PE%", "Count"]
        print(by_type)
        print("=" * 60)

    if "region" in analysis_df.columns:
        print("\nPERFORMANCE BY REGION (Top 10)")
        print("=" * 60)
        by_region = (
            analysis_df.groupby("region")
            .agg({
                "abs_error": ["mean", "median"],
                "pct_error": ["mean", "median"],
                "actual_price": "count",
            })
            .round(2)
        )
        by_region.columns = ["Mean AE", "Median AE", "Mean PE%", "Median PE%", "Count"]
        by_region = by_region.sort_values("Mean AE", ascending=False).head(10)
        print(by_region)
        print("=" * 60)

    if "rooms" in analysis_df.columns:
        print("\nPERFORMANCE BY NUMBER OF ROOMS")
        print("=" * 60)
        by_rooms = (
            analysis_df.groupby("rooms")
            .agg({
                "abs_error": ["mean", "median"],
                "pct_error": ["mean", "median"],
                "actual_price": "count",
            })
            .round(2)
        )
        by_rooms.columns = ["Mean AE", "Median AE", "Mean PE%", "Median PE%", "Count"]
        print(by_rooms)
        print("=" * 60)


def log_to_mlflow(
    tuned_df,
    tuned_models,
    tuning_info,
    X,
    X_train,
    X_test,
    target_col,
    experiment_name,
):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print("Logging experiments to MLflow...\n")

    for _, row in tuned_df.iterrows():
        model_name = row["Model"]
        pipeline = tuned_models[model_name]

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("dataset", "sale_processed_enhanced")
            mlflow.log_param("target_col", target_col)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("cv_folds", CV_FOLDS)
            mlflow.log_param("hyperparameter_tuning", "Yes")

            mlflow.log_metric("test_rmse", row["RMSE"])
            mlflow.log_metric("test_mae", row["MAE"])
            mlflow.log_metric("test_r2", row["R²"])
            mlflow.log_metric("test_mape", row["MAPE (%)"])
            mlflow.log_metric("cv_r2_mean", row["CV R² Mean"])
            mlflow.log_metric("cv_r2_std", row["CV R² Std"])

            mlflow.sklearn.log_model(pipeline, "model")

            if model_name in tuning_info:
                mlflow.log_params(tuning_info[model_name].best_params_)

            print(f"✓ {model_name} logged to MLflow")

    print("\n✓ All models logged successfully!")
    print("\nTo view MLflow UI, run: mlflow ui")
    print("Then navigate to: http://localhost:5000")


def save_best_model(best_model_name, best_model, feature_cols, numeric_features, categorical_features, target_col):
    import joblib

    base_dir = Path(__file__).resolve().parent
    output_dir = (base_dir.parent / "back" / "models" / "sale_model").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"

    joblib.dump(best_model, model_path)
    print(f"✓ Best model saved to: {model_path}")

    feature_info = {
        "feature_columns": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": target_col,
    }
    joblib.dump(feature_info, output_dir / "feature_info.pkl")
    print("✓ Feature information saved as: feature_info.pkl")

    print(f"\nModel: {best_model_name}")
    return model_path


def summary_report(
    df,
    X_train,
    X_test,
    feature_cols,
    leakage_features,
    baseline_df,
    tuned_df,
    comparison_df,
):
    print("\n" + "=" * 90)
    print("ENHANCED ML PROJECT SUMMARY")
    print("=" * 90)

    print("\n1. DATA PROCESSING")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    print(f"   - Total features: {len(feature_cols)}")
    print(
        f"   - Engineered features: {len(feature_cols) - len(df.columns) + len(leakage_features) + 1}"
    )

    print("\n2. DATA LEAKAGE PREVENTION")
    print(f"   ✓ Removed features: {', '.join(leakage_features)}")

    print("\n3. FEATURE ENGINEERING APPLIED")
    print("   ✓ Space-related features (surface_per_room, bathroom_ratio, total_rooms)")
    print("   ✓ Amenity scores (amenity_score, luxury_score)")
    print("   ✓ Property categorization (size_category, is_premium_location)")
    print("   ✓ Density features (room_density)")

    print("\n4. MODEL PERFORMANCE COMPARISON")
    print("\n   Baseline (No Tuning):")
    for _, row in baseline_df.iterrows():
        print(f"   - {row['Model']:20s}: R² = {row['R²']:.4f}, RMSE = {row['RMSE']:.2f}")

    print("\n   Tuned (With Hyperparameter Optimization):")
    for _, row in tuned_df.iterrows():
        print(f"   - {row['Model']:25s}: R² = {row['R²']:.4f}, RMSE = {row['RMSE']:.2f}")

    print("\n5. BEST MODEL")
    best_row = tuned_df.iloc[0]
    print(f"   Model: {best_row['Model']}")
    print(f"   Test R²: {best_row['R²']:.4f}")
    print(f"   Test RMSE: {best_row['RMSE']:.2f} TND")
    print(f"   Test MAE: {best_row['MAE']:.2f} TND")
    print(f"   Test MAPE: {best_row['MAPE (%)']:.2f}%")
    print(f"   CV R² (mean ± std): {best_row['CV R² Mean']:.4f} ± {best_row['CV R² Std']:.4f}")

    if not comparison_df.empty:
        print("\n6. IMPROVEMENTS ACHIEVED")
        avg_improvement = comparison_df["Improvement"].mean()
        avg_rmse_reduction = comparison_df["RMSE Reduction"].mean()
        print(f"   Average R² improvement: {avg_improvement:.4f}")
        print(f"   Average RMSE reduction: {avg_rmse_reduction:.2f} TND")

    print("\n7. NEXT STEPS & RECOMMENDATIONS")
    print("   • Monitor model performance on new data")
    print("   • Consider ensemble methods (stacking best models)")
    print("   • Collect more data for underrepresented segments")
    print("   • Update model quarterly with new rental data")
    print("   • Implement prediction intervals for uncertainty quantification")
    print("   • A/B test model predictions against current pricing strategy")

    print("\n" + "=" * 90)
    print("END OF ENHANCED ML PROJECT")
    print("=" * 90)


def main():
    df = load_data()
    df = analyze_outliers(df)

    df_fe = feature_engineering(df)
    correlation_analysis(df_fe)

    X, y, feature_cols, target_col = prepare_features(df_fe)
    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    X_train, X_test, y_train, y_test = split_data(X, y)

    baseline_df, baseline_pipelines = train_baseline_models(
        preprocessor, X_train, X_test, y_train, y_test
    )

    if SKIP_TUNING:
        print("\n⚠ Skipping hyperparameter tuning. Using baseline models.")
        tuned_models = {f"{name} (Baseline)": pipeline for name, pipeline in baseline_pipelines.items()}
        tuned_df = baseline_df.copy()
        tuned_df["Model"] = tuned_df["Model"].astype(str) + " (Baseline)"
        tuning_info = {}
    else:
        tuned_models, tuning_info = tune_models(preprocessor, X_train, y_train)
        tuned_df = evaluate_tuned_models(tuned_models, X_train, X_test, y_train, y_test)

    comparison_df = compare_baseline_tuned(baseline_df, tuned_df)

    best_model_name, best_model = feature_importance_analysis(
        tuned_df, tuned_models, numeric_features, categorical_features
    )

    if best_model is not None:
        y_pred_best = best_model.predict(X_test)
        residuals = y_test - y_pred_best
        detailed_diagnostics(best_model, best_model_name, X_test, y_test)
        performance_by_characteristics(X_test, y_test, y_pred_best, residuals)

        experiment_name = f"real_estate_sale_regression_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_to_mlflow(
            tuned_df,
            tuned_models,
            tuning_info,
            X,
            X_train,
            X_test,
            target_col,
            experiment_name,
        )

        save_best_model(
            best_model_name,
            best_model,
            feature_cols,
            numeric_features,
            categorical_features,
            target_col,
        )

        leakage_features = ["price_normalized"]
        summary_report(
            df,
            X_train,
            X_test,
            feature_cols,
            leakage_features,
            baseline_df,
            tuned_df,
            comparison_df,
        )


if __name__ == "__main__":
    main()
