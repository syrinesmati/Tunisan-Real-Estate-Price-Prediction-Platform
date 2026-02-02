"""
train3.py - Enhanced with polynomial features and better feature engineering
"""
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape


def load_clustered_data_enhanced(transaction_type: str):
    """
    Load with enhanced feature engineering:
    - Polynomial features (surface², rooms²)
    - Interaction features (surface × rooms, rooms × bathrooms)
    - Log features (log surface)
    - Normalized numeric features
    - One-hot encoded categoricals
    """
    data_path = BASE_DIR / "data" / "processed" / f"{transaction_type}_clustered.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Clustered data not found: {data_path}")
    
    print(f"\n📂 Loading {transaction_type.upper()} (enhanced features)...")
    df = pd.read_csv(data_path)
    
    print(f"   Total samples: {len(df):,}")
    
    # Target variable
    y = df["price"].values
    
    # Base features
    feature_cols = ["city", "region", "surface", "rooms", "bathrooms", "property_type", "price_segment"]
    feature_cols = [col for col in feature_cols if col in df.columns]
    feature_df = df[feature_cols].copy()
    
    # Store numeric cols for later scaling
    numeric_cols = ["surface", "rooms", "bathrooms"]
    
    # ===== FEATURE ENGINEERING =====
    print(f"   Adding engineered features...")
    
    # 1. Polynomial features
    feature_df["surface_squared"] = df["surface"] ** 2
    feature_df["rooms_squared"] = df["rooms"] ** 2
    feature_df["bathrooms_squared"] = df["bathrooms"] ** 2
    
    # 2. Interaction features
    feature_df["surface_x_rooms"] = df["surface"] * df["rooms"]
    feature_df["surface_x_bathrooms"] = df["surface"] * df["bathrooms"]
    feature_df["rooms_x_bathrooms"] = df["rooms"] * df["bathrooms"]
    
    # 3. Log features (log(x+1) to handle zeros)
    feature_df["log_surface"] = np.log1p(df["surface"])
    feature_df["log_rooms"] = np.log1p(df["rooms"])
    feature_df["log_bathrooms"] = np.log1p(df["bathrooms"])
    
    # 4. Ratio features
    feature_df["rooms_per_sqm"] = df["rooms"] / (df["surface"] + 1)
    feature_df["bathrooms_per_room"] = df["bathrooms"] / (df["rooms"] + 1)
    feature_df["avg_room_size"] = df["surface"] / (df["rooms"] + 1)
    
    print(f"     ✓ Added: polynomial (3) + interaction (3) + log (3) + ratio (3) = 12 features")
    
    # Scale numeric features BEFORE train-test split
    scaler = StandardScaler()
    numeric_features = numeric_cols + [
        "surface_squared", "rooms_squared", "bathrooms_squared",
        "surface_x_rooms", "surface_x_bathrooms", "rooms_x_bathrooms",
        "log_surface", "log_rooms", "log_bathrooms",
        "rooms_per_sqm", "bathrooms_per_room", "avg_room_size"
    ]
    
    feature_df[numeric_features] = scaler.fit_transform(feature_df[numeric_features])
    
    # ONE-HOT encode categoricals
    cat_cols = [c for c in ["city", "region", "property_type", "price_segment"] if c in feature_df.columns]
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=False)
    
    # Train-test split
    X = feature_df.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    
    print(f"✅ Data prepared:")
    print(f"   Total features: {X.shape[1]} (base + polynomial + interaction + log + ratio)")
    print(f"   Train: {X_train.shape[0]:,} samples")
    print(f"   Test:  {X_test.shape[0]:,} samples")
    print(f"   Price range: {y_train.min():.0f} - {y_train.max():.0f} TND")
    
    return X_train, X_test, y_train, y_test


def get_models():
    return {
        'Ridge_Regression': Ridge(alpha=10.0, random_state=RANDOM_STATE),
        'Lasso_Regression': Lasso(alpha=1.0, random_state=RANDOM_STATE),
        'Random_Forest': RandomForestRegressor(
            n_estimators=200, 
            random_state=RANDOM_STATE,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            n_jobs=-1,
            max_features='sqrt'
        ),
        'Gradient_Boosting': GradientBoostingRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            max_depth=5,
            learning_rate=0.03,
            min_samples_split=3,
            subsample=0.8
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            max_depth=7,
            learning_rate=0.03,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            max_depth=7,
            learning_rate=0.03,
            num_leaves=50,
            min_data_in_leaf=3,
            verbose=-1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            lambda_l1=0.1,
            lambda_l2=0.1
        )
    }


def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test, transaction_type, experiment_name):
    print(f"\n  🤖 Training {model_name}...")
    
    with mlflow.start_run(run_name=f"{transaction_type}_{model_name}"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("transaction_type", transaction_type)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("feature_engineering", "poly+interaction+log+ratio+scaled")
        
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mape = calculate_mape(y_test, y_test_pred)
        
        overfitting_gap = train_r2 - test_r2
        
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mape", test_mape)
        mlflow.log_metric("overfitting_gap", overfitting_gap)
        
        mlflow.sklearn.log_model(
            model, 
            f"{transaction_type}_{model_name}",
            registered_model_name=f"{transaction_type}_enhanced_{model_name}"
        )
        
        print(f"     Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        print(f"     Test MAE: {test_mae:.2f} | RMSE: {test_rmse:.2f} | Gap: {overfitting_gap:.4f}")
        
        return {
            'model_name': model_name,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'overfitting_gap': overfitting_gap
        }


def run_experiment(transaction_type: str, experiment_name: str):
    print(f"\n{'='*80}")
    print(f"🏠 Running Experiment (ENHANCED): {transaction_type.upper()}")
    print(f"{'='*80}")
    
    X_train, X_test, y_train, y_test = load_clustered_data_enhanced(transaction_type)
    
    models = get_models()
    results = []
    
    for model_name, model in models.items():
        metrics = train_and_evaluate_model(
            model_name, model, X_train, X_test, y_train, y_test,
            transaction_type, experiment_name
        )
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_r2', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"📊 {transaction_type.upper()} Results (sorted by Test R²):")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    best_model_name = results_df.iloc[0]['model_name']
    best_r2 = results_df.iloc[0]['test_r2']
    print(f"\n🏆 Best Model: {best_model_name} (Test R² = {best_r2:.4f})")
    
    return results_df


def main():
    print("="*80)
    print("🏠 ENHANCED FEATURE ENGINEERING - POLYNOMIAL + LOG + INTERACTION + SCALING")
    print("="*80)
    print(f"📊 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    experiment_name = f"enhanced_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"📊 Experiment: {experiment_name}")
    print(f"📁 Experiment ID: {experiment.experiment_id}")
    
    # Train
    rent_results = run_experiment('rent', experiment_name)
    sale_results = run_experiment('sale', experiment_name)
    
    # Save
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rent_results.to_csv(output_dir / f"enhanced_rent_results_{timestamp}.csv", index=False)
    sale_results.to_csv(output_dir / f"enhanced_sale_results_{timestamp}.csv", index=False)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
