"""
MLflow Model Comparison Example
Compares multiple models and logs everything to MLflow
"""
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Set tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model_comparison")

print("🏆 MLflow Model Comparison Example\n")
print("="*60)

# Create data
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to compare
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Ridge (alpha=10.0)": Ridge(alpha=10.0),
    "Lasso (alpha=1.0)": Lasso(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)
}

results = []

# Train and log each model
for model_name, model in models.items():
    print(f"\n🤖 Training: {model_name}")
    
    with mlflow.start_run(run_name=model_name):
        
        # Time training
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predictions
        y_pred_test = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        # Log everything
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(model.get_params())
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("training_time", training_time)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Store results
        results.append({
            "model": model_name,
            "mse": mse,
            "r2": r2,
            "time": training_time
        })
        
        print(f"  ✅ Test R²: {r2:.4f} | MSE: {mse:.2f} | Time: {training_time:.3f}s")

# Print summary
print("\n" + "="*60)
print("📊 Results Summary (sorted by R² score):\n")
results_sorted = sorted(results, key=lambda x: x['r2'], reverse=True)

for i, result in enumerate(results_sorted, 1):
    print(f"{i}. {result['model']}")
    print(f"   R²: {result['r2']:.4f} | MSE: {result['mse']:.2f} | Time: {result['time']:.3f}s\n")

print("="*60)
print("🎯 Now open MLflow UI and compare all runs!")
print("   http://localhost:5000")
print("\n💡 In MLflow UI you can:")
print("   • Click 'Compare' to see side-by-side metrics")
print("   • Create charts to visualize performance")
print("   • Download any model to use in production")
