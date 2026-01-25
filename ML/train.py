"""
Tunisian Real Estate Price Prediction - ML Pipeline
Main training script with MLflow tracking
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Import custom modules
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_evaluation import ModelEvaluator

# Load environment variables
load_dotenv()

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'tunisian_real_estate_prediction')
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))


class RealEstatePipeline:
    """Main ML Pipeline for Real Estate Price Prediction"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        
        # Set up MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
            'lasso': Lasso(alpha=1.0, random_state=RANDOM_STATE),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=RANDOM_STATE
            ),
            'xgboost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                verbose=-1
            )
        }
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from CSV"""
        print(f"📂 Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} rows")
        return df
    
    def prepare_data(self, df: pd.DataFrame, transaction: str = 'all'):
        """Prepare data for training"""
        print(f"\n🔧 Preprocessing data for transaction type: {transaction}")
        
        # Filter by transaction type if specified
        if transaction != 'all':
            df = df[df['transaction'] == transaction].copy()
            print(f"  Filtered to {len(df)} {transaction} properties")
        
        # Preprocess
        df_clean = self.preprocessor.clean_data(df)
        df_featured = self.feature_engineer.create_features(df_clean)
        
        # Separate features and target
        target_col = 'price'
        X = df_featured.drop(columns=[target_col])
        y = df_featured[target_col]
        
        return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    def train_and_evaluate(self, model_name: str, model, X_train, X_test, y_train, y_test):
        """Train a model and log results to MLflow"""
        print(f"\n🤖 Training {model_name}...")
        
        with mlflow.start_run(run_name=model_name):
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            metrics = {
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'test_r2': r2_score(y_test, y_pred_test),
            }
            
            # Log parameters
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Print results
            print(f"  Test MAE: {metrics['test_mae']:.2f}")
            print(f"  Test RMSE: {metrics['test_rmse']:.2f}")
            print(f"  Test R²: {metrics['test_r2']:.4f}")
            
            return metrics
    
    def run_experiment(self, data_path: str, transaction: str = 'all'):
        """Run complete experiment"""
        print(f"\n{'='*60}")
        print(f"🚀 Starting ML Experiment: {transaction.upper()}")
        print(f"{'='*60}")
        
        # Load and prepare data
        df = self.load_data(data_path)
        X_train, X_test, y_train, y_test = self.prepare_data(df, transaction)
        
        print(f"\n📊 Dataset Summary:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {X_train.shape[1]}")
        
        # Train all models
        results = {}
        for model_name, model in self.models.items():
            try:
                metrics = self.train_and_evaluate(
                    model_name, model, X_train, X_test, y_train, y_test
                )
                results[model_name] = metrics
            except Exception as e:
                print(f"  ❌ Error training {model_name}: {e}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("📈 Experiment Summary")
        print(f"{'='*60}")
        
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('test_r2', ascending=False)
        print(results_df[['test_mae', 'test_rmse', 'test_r2']])
        
        best_model = results_df.index[0]
        print(f"\n🏆 Best Model: {best_model}")
        print(f"  R² Score: {results_df.loc[best_model, 'test_r2']:.4f}")
        
        return results


def main():
    """Main execution"""
    pipeline = RealEstatePipeline()
    
    # Check if data exists
    data_path = Path("data/raw/tunisia_real_estate.csv")
    
    if not data_path.exists():
        print("⚠️  No data found!")
        print(f"Please place your dataset at: {data_path.absolute()}")
        print("\nYou can:")
        print("  1. Download from Kaggle")
        print("  2. Run the web scraper from the backend")
        return
    
    # Run experiments for both transaction types
    print("\n" + "="*60)
    print("🏠 TRAINING MODELS FOR SALE PROPERTIES")
    print("="*60)
    pipeline.run_experiment(str(data_path), transaction='sale')
    
    print("\n" + "="*60)
    print("🏠 TRAINING MODELS FOR RENT PROPERTIES")
    print("="*60)
    pipeline.run_experiment(str(data_path), transaction='rent')
    
    print("\n✅ All experiments completed!")
    print(f"📊 View results at: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
