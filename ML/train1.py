"""
Tunisian Real Estate Price Prediction - Clustered Data Training
Trains on rent and sale clustered datasets with price_segment labels
"""
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
RANDOM_STATE = 42

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape


def load_clustered_data(transaction_type: str):
    """
    Load clustered CSV data and prepare features with feature engineering.
    
    Features: city (target encoded), region (target encoded), surface, rooms, bathrooms, 
              property_type, property_type_cluster + engineered features
    
    Args:
        transaction_type: 'rent' or 'sale'
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    data_path = BASE_DIR / "data" / "processed" / f"{transaction_type}_clustered.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Clustered data not found: {data_path}")
    
    print(f"\n📂 Loading {transaction_type.upper()} clustered data...")
    df = pd.read_csv(data_path)
    
    print(f"   Total samples: {len(df):,}")
    
    # Target variable (actual price)
    y = df["price"].values
    
    # Base features
    feature_cols = ["city", "region", "surface", "rooms", "bathrooms", "property_type", "property_type_cluster"]
    
    # Select only available features
    feature_cols = [col for col in feature_cols if col in df.columns]
    feature_df = df[feature_cols].copy()
    
    # ===== FEATURE ENGINEERING =====
    # 1. Rooms per square meter (space density)
    feature_df["rooms_per_sqm"] = df["rooms"] / (df["surface"] + 1)
    
    # 2. Bathrooms per room (luxury indicator)
    feature_df["bathrooms_per_room"] = df["bathrooms"] / (df["rooms"] + 1)
    
    # 3. Average room size
    feature_df["avg_room_size"] = df["surface"] / (df["rooms"] + 1)
    
    print(f"   ✓ Added feature engineering: rooms_per_sqm, bathrooms_per_room, avg_room_size")
    
    # First split to avoid target leakage in encoding
    temp_indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        temp_indices, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    
    # Target encode city and region using ONLY training data
    for col in ["city", "region"]:
        if col in feature_df.columns:
            # Calculate mean price per category from training data only
            encoding_map = df.iloc[train_idx].groupby(col)["price"].mean().to_dict()
            global_mean = df.iloc[train_idx]["price"].mean()
            
            # Apply encoding (use global mean for unseen categories)
            feature_df[f"{col}_encoded"] = feature_df[col].map(encoding_map).fillna(global_mean)
            feature_df = feature_df.drop(columns=[col])
    
    # One-hot encode categorical features (only property_type)
    cat_cols = [c for c in ["property_type"] if c in feature_df.columns]
    if cat_cols:
        feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=False)
    
    # Split the encoded features
    X_train = feature_df.iloc[train_idx].values
    X_test = feature_df.iloc[test_idx].values
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"✅ Data prepared:")
    print(f"   Total features: {X_train.shape[1]} (7 base + 3 engineered)")
    print(f"   Train: {X_train.shape[0]:,} samples")
    print(f"   Test:  {X_test.shape[0]:,} samples")
    print(f"   Price range: {y_train.min():.0f} - {y_train.max():.0f} TND")
    
    return X_train, X_test, y_train, y_test


def get_models():
    """Return dictionary of models to train"""
    return {
        'Linear_Regression': LinearRegression(),
        'Ridge_Regression': Ridge(alpha=10.0, random_state=RANDOM_STATE),
        'Lasso_Regression': Lasso(alpha=10.0, random_state=RANDOM_STATE),
        'Decision_Tree': DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=5),
        'Random_Forest': RandomForestRegressor(
            n_estimators=50, 
            random_state=RANDOM_STATE,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1
        ),
        'Gradient_Boosting': GradientBoostingRegressor(
            n_estimators=50,
            random_state=RANDOM_STATE,
            max_depth=3,
            learning_rate=0.05,
            min_samples_split=10
        ),
        'XGBoost': XGBRegressor(
            n_estimators=50,
            random_state=RANDOM_STATE,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=5
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=50,
            random_state=RANDOM_STATE,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=15,
            min_data_in_leaf=5,
            verbose=-1
        )
    }


def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test, transaction_type, experiment_name):
    """Train a model and log results to MLflow"""
    print(f"\n  🤖 Training {model_name}...")
    
    with mlflow.start_run(run_name=f"{transaction_type}_{model_name}"):
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("transaction_type", transaction_type)
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_param("n_test", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        train_mape = calculate_mape(y_train, y_train_pred)
        test_mape = calculate_mape(y_test, y_test_pred)
        
        overfitting_gap = train_r2 - test_r2
        
        # Log metrics
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mape", train_mape)
        mlflow.log_metric("test_mape", test_mape)
        mlflow.log_metric("overfitting_gap", overfitting_gap)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            f"{transaction_type}_{model_name}",
            registered_model_name=f"{transaction_type}_{model_name}"
        )
        
        # Print results
        print(f"     Train R²: {train_r2:.4f} | MAE: {train_mae:.2f}")
        print(f"     Test R²:  {test_r2:.4f} | MAE: {test_mae:.2f}")
        print(f"     Test RMSE: {test_rmse:.2f} | MAPE: {test_mape:.2f}%")
        print(f"     Overfitting gap: {overfitting_gap:.4f}")
        
        return {
            'model_name': model_name,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'overfitting_gap': overfitting_gap
        }


def run_experiment(transaction_type: str, experiment_name: str):
    """Run full experiment for a transaction type"""
    print(f"\n{'='*80}")
    print(f"🏠 Running Experiment: {transaction_type.upper()}")
    print(f"{'='*80}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_clustered_data(transaction_type)
    
    # Get models
    models = get_models()
    
    # Train and evaluate each model
    results = []
    trained_models = {}
    
    for model_name, model in models.items():
        metrics = train_and_evaluate_model(
            model_name, model, X_train, X_test, y_train, y_test,
            transaction_type, experiment_name
        )
        results.append(metrics)
        trained_models[model_name] = model
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_r2', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"📊 {transaction_type.upper()} Results Summary (sorted by Test R²):")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    # Find best model
    best_model_name = results_df.iloc[0]['model_name']
    best_r2 = results_df.iloc[0]['test_r2']
    print(f"\n🏆 Best Model: {best_model_name} (Test R² = {best_r2:.4f})")
    
    return results_df, trained_models


def main():
    """Main training pipeline"""
    print("="*80)
    print("🏠 TUNISIAN REAL ESTATE PRICE PREDICTION - CLUSTERED DATA TRAINING")
    print("="*80)
    print(f"📊 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"📂 Base Directory: {BASE_DIR}")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create experiment
    experiment_name = f"clustered_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"📊 Experiment: {experiment_name}")
    print(f"📁 Experiment ID: {experiment.experiment_id}")
    
    # Store all results
    all_results = {}
    
    # Train on RENT data
    rent_results, rent_models = run_experiment('rent', experiment_name)
    all_results['rent'] = rent_results
    
    # Train on SALE data
    sale_results, sale_models = run_experiment('sale', experiment_name)
    all_results['sale'] = sale_results
    
    # Save results to CSV
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rent_results.to_csv(output_dir / f"clustered_rent_results_{timestamp}.csv", index=False)
    sale_results.to_csv(output_dir / f"clustered_sale_results_{timestamp}.csv", index=False)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 View MLflow UI: {MLFLOW_TRACKING_URI}")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    results = main()
