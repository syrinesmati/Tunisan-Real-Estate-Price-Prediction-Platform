"""
Train with one-hot encoded city/region for better location granularity
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


def load_clustered_data_onehot(transaction_type: str):
    """
    Load with ONE-HOT encoding for city and region for maximum location granularity
    """
    data_path = BASE_DIR / "data" / "processed" / f"{transaction_type}_clustered.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Clustered data not found: {data_path}")
    
    print(f"\n📂 Loading {transaction_type.upper()} clustered data (one-hot encoded)...")
    df = pd.read_csv(data_path)
    
    print(f"   Total samples: {len(df):,}")
    
    # Target variable
    y = df["price"].values
    
    # Features
    feature_cols = ["city", "region", "surface", "rooms", "bathrooms", "property_type", "property_type_cluster", "price_segment"]
    feature_cols = [col for col in feature_cols if col in df.columns]
    feature_df = df[feature_cols].copy()
    
    # Feature engineering (no price-based features!)
    feature_df["rooms_per_sqm"] = df["rooms"] / (df["surface"] + 1)
    feature_df["bathrooms_per_room"] = df["bathrooms"] / (df["rooms"] + 1)
    feature_df["avg_room_size"] = df["surface"] / (df["rooms"] + 1)
    
    # ONE-HOT encode city, region, property_type, price_segment
    cat_cols = [c for c in ["city", "region", "property_type", "price_segment"] if c in feature_df.columns]
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=False)
    
    # Train-test split
    X = feature_df.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    
    print(f"✅ Data prepared:")
    print(f"   Total features after one-hot: {X.shape[1]}")
    print(f"   Train: {X_train.shape[0]:,} samples")
    print(f"   Test:  {X_test.shape[0]:,} samples")
    print(f"   Price range: {y_train.min():.0f} - {y_train.max():.0f} TND")
    
    return X_train, X_test, y_train, y_test


def get_models():
    return {
        'Ridge_Regression': Ridge(alpha=100.0, random_state=RANDOM_STATE),
        'Random_Forest': RandomForestRegressor(
            n_estimators=100, 
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        ),
        'Gradient_Boosting': GradientBoostingRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=5
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=5,
            verbose=-1,
            feature_fraction=0.8
        )
    }


def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test, transaction_type, experiment_name):
    print(f"\n  🤖 Training {model_name}...")
    
    with mlflow.start_run(run_name=f"{transaction_type}_{model_name}"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("transaction_type", transaction_type)
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        
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
            registered_model_name=f"{transaction_type}_onehot_{model_name}"
        )
        
        print(f"     Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        print(f"     Test MAE: {test_mae:.2f} | RMSE: {test_rmse:.2f}")
        
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
    print(f"🏠 Running Experiment (ONE-HOT): {transaction_type.upper()}")
    print(f"{'='*80}")
    
    X_train, X_test, y_train, y_test = load_clustered_data_onehot(transaction_type)
    
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
    print("🏠 ONE-HOT ENCODING - LOCATION GRANULARITY APPROACH")
    print("="*80)
    print(f"📊 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    experiment_name = f"onehot_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    rent_results.to_csv(output_dir / f"onehot_rent_results_{timestamp}.csv", index=False)
    sale_results.to_csv(output_dir / f"onehot_sale_results_{timestamp}.csv", index=False)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
