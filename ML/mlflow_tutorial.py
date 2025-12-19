"""
Simple MLflow Example - Understanding the Basics
This script demonstrates MLflow's core features:
1. Experiment tracking
2. Parameter logging
3. Metric logging
4. Model saving
"""
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set tracking URI (where MLflow will store data)
mlflow.set_tracking_uri("http://localhost:5000")

# Create or set experiment
mlflow.set_experiment("mlflow_tutorial")

print("🚀 MLflow Tutorial - Understanding Experiment Tracking\n")
print("="*60)

# Create sample data
print("📊 Creating sample regression dataset...")
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Features: {X_train.shape[1]}\n")

# Start MLflow run
with mlflow.start_run(run_name="linear_regression_example"):
    
    print("📝 Starting MLflow Run...")
    print(f"  Run ID: {mlflow.active_run().info.run_id}\n")
    
    # Log parameters
    print("⚙️  Logging parameters...")
    params = {
        "model_type": "LinearRegression",
        "n_samples": len(X_train),
        "n_features": X_train.shape[1],
        "test_size": 0.2,
        "random_state": 42
    }
    mlflow.log_params(params)
    print("  ✅ Parameters logged\n")
    
    # Train model
    print("🤖 Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("  ✅ Model trained\n")
    
    # Make predictions
    print("🔮 Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    print("📈 Calculating metrics...")
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Log metrics
    metrics = {
        "train_mse": train_mse,
        "train_r2": train_r2,
        "test_mse": test_mse,
        "test_r2": test_r2
    }
    mlflow.log_metrics(metrics)
    
    print(f"  Train MSE: {train_mse:.2f}")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test MSE: {test_mse:.2f}")
    print(f"  Test R²: {test_r2:.4f}")
    print("  ✅ Metrics logged\n")
    
    # Log model
    print("💾 Saving model to MLflow...")
    mlflow.sklearn.log_model(model, "model")
    print("  ✅ Model saved\n")
    
    # Log additional info as tags
    mlflow.set_tags({
        "project": "mlflow_tutorial",
        "purpose": "learning",
        "author": "ML-project"
    })
    
    print("="*60)
    print("✅ MLflow Run Complete!\n")
    print("📊 View your results:")
    print(f"  1. Open MLflow UI: http://localhost:5000")
    print(f"  2. Look for experiment: 'mlflow_tutorial'")
    print(f"  3. Click on the run to see parameters, metrics, and model")

print("\n🎯 Try running this script multiple times with different parameters!")
print("   Each run will be tracked and you can compare them in MLflow UI.\n")
