"""
Model Evaluation Module
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Model evaluation utilities"""
    
    def __init__(self):
        pass
    
    def calculate_metrics(self, y_true, y_pred) -> dict:
        """Calculate regression metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def plot_predictions(self, y_true, y_pred, title="Predictions vs Actual"):
        """Plot predicted vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(title)
        plt.tight_layout()
        return plt
    
    def plot_residuals(self, y_true, y_pred, title="Residual Plot"):
        """Plot residuals"""
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title(title)
        plt.tight_layout()
        return plt
    
    def feature_importance(self, model, feature_names):
        """Get feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            return plt
        return None
