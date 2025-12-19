# ML Pipeline Documentation

## Overview
Complete machine learning pipeline for training, evaluating, and deploying real estate price prediction models.

## Pipeline Architecture

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
```

## Components

### 1. Data Preprocessing (`src/data_preprocessing.py`)

#### Clustering-Based Imputation
Novel approach to handling missing values:
- Groups similar properties using K-Means clustering
- Fills missing values with cluster statistics
- More accurate than simple mean/median imputation

**Usage:**
```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(n_clusters=5)
df_clean = preprocessor.clean_data(df)
df_imputed = preprocessor.handle_missing_values(df_clean)
```

**Features:**
- Duplicate removal
- Outlier handling (3 IQR method)
- Clustering-based missing value imputation
- Fallback to median/mode for remaining nulls

### 2. Feature Engineering (`src/feature_engineering.py`)

#### Created Features
- `property_age`: Current year - construction year
- `is_new_construction`: Boolean (age <= 2 years)
- `area_per_room`: Area / (rooms + 1)
- `area_per_bedroom`: Area / (bedrooms + 1)
- `total_amenities`: Count of premium features
- `is_major_city`: Boolean for major Tunisian cities
- `is_premium_type`: Boolean for villa/duplex

#### Encoding Strategies
- **Label Encoding**: For low cardinality (<= 10 unique values)
- **Target Encoding**: For high cardinality features
- Preserves relationship with target variable

**Usage:**
```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_featured = engineer.create_features(df)
```

### 3. Model Training (`train.py`)

#### Supported Models
1. **Linear Regression** - Baseline
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization  
4. **Random Forest** - Ensemble learning
5. **Gradient Boosting** - Sequential boosting
6. **XGBoost** - Optimized gradient boosting
7. **LightGBM** - Fast gradient boosting

#### Training Process
```python
from train import RealEstatePipeline

pipeline = RealEstatePipeline()
results = pipeline.run_experiment(
    data_path="data/raw/tunisia_real_estate.csv",
    transaction_type="sale"  # or "rent"
)
```

#### MLflow Integration
All experiments automatically logged:
- Parameters (model hyperparameters)
- Metrics (MAE, RMSE, R²)
- Model artifacts
- Feature importance plots

### 4. Model Evaluation (`src/model_evaluation.py`)

#### Metrics
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R²** (R-squared): Proportion of variance explained
- **MAPE** (Mean Absolute Percentage Error): Percentage error

#### Visualizations
```python
from src.model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Prediction plot
evaluator.plot_predictions(y_true, y_pred)

# Residual plot
evaluator.plot_residuals(y_true, y_pred)

# Feature importance
evaluator.feature_importance(model, feature_names)
```

### 5. Automated Retraining (`retrain.py`)

#### Continuous Learning
- Monitors for new scraped data
- Automatically merges Kaggle + scraped datasets
- Removes duplicates
- Retrains all models
- Updates deployment models

**Usage:**
```bash
python retrain.py
```

Or set up as cron job:
```bash
# Retrain daily at 2 AM
0 2 * * * cd /path/to/ML && python retrain.py
```

## Data Requirements

### Minimum Columns
- `price` (target)
- `area`
- `governorate`
- `city`
- `property_type`
- `transaction_type`

### Optional Columns
- `rooms`, `bedrooms`, `bathrooms`
- `floor`, `construction_year`
- `has_elevator`, `has_parking`, `has_garden`, `has_pool`
- `is_furnished`
- `delegation`, `condition`

## Training Guide

### Step 1: Prepare Data
Place your CSV file in `ML/data/raw/tunisia_real_estate.csv`

Required format:
```csv
governorate,city,property_type,transaction_type,area,price,rooms,bedrooms,...
Tunis,La Marsa,apartment,sale,120,250000,4,3,...
```

### Step 2: Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### Step 3: Train Models
```bash
python train.py
```

### Step 4: View Results
Open MLflow UI:
```bash
mlflow ui --port 5000
```
Navigate to http://localhost:5000

### Step 5: Export Best Model
Models automatically saved in `models/` directory:
- `models/rent_model/model.pkl`
- `models/sale_model/model.pkl`

## Model Selection Criteria

Best model chosen based on:
1. **R² Score** (primary metric)
2. **RMSE** (lower is better)
3. **Training time** (if scores are close)

## Hyperparameter Tuning

Extend training with GridSearchCV:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    XGBRegressor(),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

## Model Deployment

### Option 1: Joblib
```python
import joblib

# Save
joblib.dump(model, 'models/sale_model/model.pkl')

# Load
model = joblib.load('models/sale_model/model.pkl')
```

### Option 2: MLflow Registry
```python
import mlflow

# Register model
mlflow.sklearn.log_model(model, "rent_model")

# Load from registry
model = mlflow.sklearn.load_model("models:/rent_model/latest")
```

## Performance Benchmarks

Typical performance on Tunisian dataset:
- **Linear Regression**: R² ~ 0.75
- **Random Forest**: R² ~ 0.85
- **XGBoost**: R² ~ 0.88
- **LightGBM**: R² ~ 0.89

## Troubleshooting

### Issue: Poor R² Score
- Check data quality
- Verify feature engineering
- Try different models
- Tune hyperparameters

### Issue: High RMSE
- Remove price outliers
- Log-transform target variable
- Use ensemble methods

### Issue: Overfitting
- Increase regularization
- Reduce model complexity
- Add more training data

## Advanced Topics

### Custom Loss Functions
```python
def custom_loss(y_true, y_pred):
    # Penalize underestimation more
    diff = y_true - y_pred
    return np.where(diff > 0, 2 * diff**2, diff**2).mean()
```

### Stacking Ensemble
```python
from sklearn.ensemble import StackingRegressor

stacking = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor()),
        ('xgb', XGBRegressor()),
        ('lgbm', LGBMRegressor())
    ],
    final_estimator=Ridge()
)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring='r2'
)
print(f"CV R² Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## Production Checklist

- [ ] Data validation pipeline
- [ ] Model versioning
- [ ] A/B testing setup
- [ ] Monitoring & alerting
- [ ] Automated retraining
- [ ] Model rollback strategy
- [ ] Performance logging
- [ ] Error handling

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
