"""
train4.py - Hyperparameter Optimization with GridSearchCV + Cross-Validation
"""
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape


def load_clustered_data_clean(transaction_type: str):
    """
    Load data and remove extreme outliers (top/bottom 2% of prices)
    """
    data_path = BASE_DIR / "data" / "new" / f"{transaction_type}_clustered.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Clustered data not found: {data_path}")
    
    print(f"\n📂 Loading {transaction_type.upper()} (cleaned, optimized)...")
    df = pd.read_csv(data_path)
    
    original_size = len(df)
    
    # Remove extreme outliers (bottom 2% and top 2%)
    lower_bound = df["price"].quantile(0.02)
    upper_bound = df["price"].quantile(0.98)
    df = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)].reset_index(drop=True)
    
    removed = original_size - len(df)
    print(f"   Removed {removed} outliers ({removed/original_size*100:.1f}%)")
    print(f"   Remaining samples: {len(df):,}")
    
    # Target variable
    y = df["price"].values
    
    # Features (exclude price and all price-derived columns to prevent data leakage)
    exclude_cols = ["price", "transaction", "price_normalized","price_segment"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_df = df[feature_cols].copy()
    
    # Lightweight feature engineering
    feature_df["rooms_per_sqm"] = df["rooms"] / (df["surface"] + 1)
    feature_df["bathrooms_per_room"] = df["bathrooms"] / (df["rooms"] + 1)
    feature_df["avg_room_size"] = df["surface"] / (df["rooms"] + 1)
    
    # ONE-HOT encode
    cat_cols = [c for c in ["region", "property_type"] if c in feature_df.columns]
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=False)
    
    X = feature_df.values
    
    print(f"   Total features: {X.shape[1]}")
    print(f"   Price range: {y.min():.0f} - {y.max():.0f} TND")
    
    return X, y


def hyperparameter_search(X, y, transaction_type: str, experiment_name: str):
    """
    Perform GridSearchCV on multiple models
    """
    print(f"\n{'='*80}")
    print(f"🔍 HYPERPARAMETER OPTIMIZATION: {transaction_type.upper()}")
    print(f"{'='*80}")
    
    # Split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    results = []
    
    # 1. Ridge Regression
    print(f"\n  🔧 Tuning Ridge_Regression...")
    ridge_params = {
        'alpha': [0.1, 1, 10, 100, 1000],
    }
    ridge = GridSearchCV(Ridge(random_state=RANDOM_STATE), ridge_params, cv=5, scoring='r2', n_jobs=-1)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_r2 = r2_score(y_test, ridge_pred)
    print(f"     Best params: {ridge.best_params_}")
    print(f"     CV R²: {ridge.best_score_:.4f} → Test R²: {ridge_r2:.4f}")
    results.append(('Ridge_Regression', ridge.best_estimator_, ridge_r2, ridge.best_params_))
    
    # 2. Random Forest
    print(f"\n  🔧 Tuning Random_Forest...")
    rf_params = {
        'max_depth': [8, 10, 12],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    rf = GridSearchCV(RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), 
                      rf_params, cv=5, scoring='r2', n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"     Best params: {rf.best_params_}")
    print(f"     CV R²: {rf.best_score_:.4f} → Test R²: {rf_r2:.4f}")
    results.append(('Random_Forest', rf.best_estimator_, rf_r2, rf.best_params_))
    
    # 3. Gradient Boosting
    print(f"\n  🔧 Tuning Gradient_Boosting...")
    gb_params = {
        'max_depth': [4, 5, 6],
        'learning_rate': [0.01, 0.03, 0.05],
        'min_samples_split': [2, 5]
    }
    gb = GridSearchCV(GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
                      gb_params, cv=5, scoring='r2', n_jobs=-1)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    print(f"     Best params: {gb.best_params_}")
    print(f"     CV R²: {gb.best_score_:.4f} → Test R²: {gb_r2:.4f}")
    results.append(('Gradient_Boosting', gb.best_estimator_, gb_r2, gb.best_params_))
    
    # 4. XGBoost
    print(f"\n  🔧 Tuning XGBoost...")
    xgb_params = {
        'max_depth': [5, 6, 7],
        'learning_rate': [0.01, 0.03, 0.05],
        'min_child_weight': [1, 3]
    }
    xgb = GridSearchCV(XGBRegressor(n_estimators=100, random_state=RANDOM_STATE),
                       xgb_params, cv=5, scoring='r2', n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_pred)
    print(f"     Best params: {xgb.best_params_}")
    print(f"     CV R²: {xgb.best_score_:.4f} → Test R²: {xgb_r2:.4f}")
    results.append(('XGBoost', xgb.best_estimator_, xgb_r2, xgb.best_params_))
    
    # 5. LightGBM
    print(f"\n  🔧 Tuning LightGBM...")
    lgb_params = {
        'max_depth': [5, 6, 7],
        'learning_rate': [0.01, 0.03, 0.05],
        'num_leaves': [20, 31]
    }
    lgb = GridSearchCV(LGBMRegressor(n_estimators=100, random_state=RANDOM_STATE, verbose=-1),
                       lgb_params, cv=5, scoring='r2', n_jobs=-1)
    lgb.fit(X_train, y_train)
    lgb_pred = lgb.predict(X_test)
    lgb_r2 = r2_score(y_test, lgb_pred)
    print(f"     Best params: {lgb.best_params_}")
    print(f"     CV R²: {lgb.best_score_:.4f} → Test R²: {lgb_r2:.4f}")
    results.append(('LightGBM', lgb.best_estimator_, lgb_r2, lgb.best_params_))
    
    # Log to MLflow
    for model_name, model, test_r2, best_params in results:
        with mlflow.start_run(run_name=f"{transaction_type}_{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_params(best_params)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.sklearn.log_model(model, f"{transaction_type}_{model_name}")
    
    # Summary
    results_df = pd.DataFrame([
        {
            'model_name': name,
            'test_r2': r2,
            'best_params': str(params)
        }
        for name, _, r2, params in results
    ])
    results_df = results_df.sort_values('test_r2', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"📊 {transaction_type.upper()} - Final Results (sorted by Test R²):")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    best_model = results_df.iloc[0]
    print(f"\n🏆 Best Model: {best_model['model_name']} (Test R² = {best_model['test_r2']:.4f})")
    
    return results_df


def main():
    print("="*80)
    print("🏠 HYPERPARAMETER OPTIMIZATION WITH GRIDSEARCHCV + 5-FOLD CV")
    print("="*80)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    experiment_name = f"optimized_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    
    # Train on rent and sale
    X_rent, y_rent = load_clustered_data_clean('rent')
    X_sale, y_sale = load_clustered_data_clean('sale')
    
    rent_results = hyperparameter_search(X_rent, y_rent, 'rent', experiment_name)
    sale_results = hyperparameter_search(X_sale, y_sale, 'sale', experiment_name)
    
    # Save results
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rent_results.to_csv(output_dir / f"optimized_rent_results_{timestamp}.csv", index=False)
    sale_results.to_csv(output_dir / f"optimized_sale_results_{timestamp}.csv", index=False)
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
