# MLflow Tutorial - Quick Start Guide

## What is MLflow?

MLflow is a platform for managing the ML lifecycle, including:
- **Tracking**: Log parameters, metrics, and models
- **Projects**: Package code for reproducibility
- **Models**: Deploy models to various platforms
- **Registry**: Store and version models

## Setup

### 1. Start MLflow Server

**Option A: Using Docker (Recommended)**
```powershell
cd C:\Users\Mediatek\Desktop\ML-project
docker-compose up mlflow -d
```

**Option B: Manually**
```powershell
cd ML
pip install mlflow
mlflow ui --port 5000
```

### 2. Access MLflow UI
Open browser: http://localhost:5000

## Running Examples

### Example 1: Basic Tracking
```powershell
cd C:\Users\Mediatek\Desktop\ML-project\ML
python mlflow_tutorial.py
```

**What it does:**
- Creates a simple regression model
- Logs parameters (model settings)
- Logs metrics (performance)
- Saves the trained model

**Check MLflow UI:**
1. Go to http://localhost:5000
2. Click on "mlflow_tutorial" experiment
3. See your run with all logged data

### Example 2: Model Comparison
```powershell
python model_comparison_example.py
```

**What it does:**
- Trains 5 different models
- Logs all results to MLflow
- Compares performance

**In MLflow UI:**
1. Check "model_comparison" experiment
2. Select multiple runs
3. Click "Compare" button
4. View side-by-side metrics

## Understanding MLflow Concepts

### 1. Experiments
- Container for runs
- Groups related ML work
- Example: "tunisian_real_estate_prediction"

### 2. Runs
- Single execution of your code
- Each run logs:
  - Parameters (inputs)
  - Metrics (outputs)
  - Artifacts (files, models)
  - Tags (metadata)

### 3. Parameters
- Input values that don't change during run
- Examples: learning_rate=0.01, n_estimators=100

### 4. Metrics
- Output values that measure performance
- Examples: accuracy=0.95, loss=0.23
- Can log multiple times (e.g., per epoch)

### 5. Artifacts
- Files produced during run
- Examples: models, plots, data files

## Key MLflow Functions

```python
# Set where MLflow stores data
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("my_experiment")

# Start a run
with mlflow.start_run(run_name="my_run"):
    
    # Log single parameter
    mlflow.log_param("learning_rate", 0.01)
    
    # Log multiple parameters
    mlflow.log_params({
        "batch_size": 32,
        "epochs": 10
    })
    
    # Log single metric
    mlflow.log_metric("accuracy", 0.95)
    
    # Log multiple metrics
    mlflow.log_metrics({
        "loss": 0.23,
        "val_loss": 0.25
    })
    
    # Log metric over time (e.g., per epoch)
    for epoch in range(10):
        mlflow.log_metric("loss", loss_value, step=epoch)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifact (file)
    mlflow.log_artifact("plot.png")
    
    # Set tags
    mlflow.set_tag("model_type", "RandomForest")
```

## MLflow UI Features

### 1. Experiments Page
- Lists all experiments
- Shows number of runs per experiment
- Click experiment to see runs

### 2. Runs Table
- Shows all runs in an experiment
- Columns: metrics, parameters, tags
- Sort and filter runs

### 3. Run Details
- Click a run to see everything:
  - Parameters used
  - Metrics achieved
  - Artifacts (including models)
  - System info

### 4. Compare Runs
- Select multiple runs
- Click "Compare"
- See parallel coordinates plots
- View scatter plots

### 5. Model Registry
- Promote models to registry
- Version models (v1, v2, v3...)
- Stage models (Staging, Production)

## Practical Workflow

1. **Development**
   ```python
   # Try different models/parameters
   with mlflow.start_run():
       model = train_model(params)
       mlflow.log_params(params)
       mlflow.log_metrics(evaluate(model))
       mlflow.sklearn.log_model(model, "model")
   ```

2. **Compare Results**
   - Open MLflow UI
   - Look at all runs
   - Find best performing model

3. **Load Best Model**
   ```python
   # From UI, get run_id of best model
   model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
   
   # Use it
   predictions = model.predict(X_test)
   ```

4. **Deploy to Production**
   ```python
   # Register model
   mlflow.register_model(
       f"runs:/{run_id}/model",
       "RealEstatePriceModel"
   )
   
   # Later, load from registry
   model = mlflow.pyfunc.load_model(
       "models:/RealEstatePriceModel/Production"
   )
   ```

## Next Steps for Your Project

1. **Run the examples** to understand MLflow basics
2. **Run the main training** script:
   ```powershell
   cd ML
   # Put data in data/raw/tunisia_real_estate.csv
   python train.py
   ```
3. **Check MLflow UI** to see all model comparisons
4. **Select best model** and use it in your backend

## Useful Commands

```powershell
# View all experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-id 1

# Delete experiment
mlflow experiments delete --experiment-id 1

# Export run
mlflow runs export --run-id <run_id> --output run.json
```

## Tips

- **Run names**: Use descriptive names for easy identification
- **Tags**: Add tags for filtering (e.g., "production", "experiment")
- **Metrics over time**: Log validation loss per epoch to see training progress
- **Artifacts**: Save plots, confusion matrices, etc.
- **Nested runs**: Use for hyperparameter tuning

## Common Issues

**Issue: Can't connect to MLflow**
```powershell
# Check if server is running
curl http://localhost:5000
```

**Issue: Too many runs**
- Archive old experiments in UI
- Or delete: `mlflow experiments delete --experiment-id X`

**Issue: Large artifacts**
- MLflow stores everything locally by default
- For production, use cloud storage (S3, Azure Blob)

## Resources

- MLflow Docs: https://mlflow.org/docs/latest/index.html
- Tracking Guide: https://mlflow.org/docs/latest/tracking.html
- Model Registry: https://mlflow.org/docs/latest/model-registry.html
