"""
ML Model Manager
Handles loading and serving ML models
"""
import joblib
import mlflow
from pathlib import Path
from typing import Optional, Dict
from app.core.config import settings


class ModelManager:
    """Manages ML models for prediction"""
    
    def __init__(self):
        self.models: Dict[str, any] = {}
        self.model_info: Dict[str, dict] = {}
        
    def load_models(self):
        """Load trained models from disk or MLflow"""
        try:
            # Load rent model
            rent_model_path = Path(settings.RENT_MODEL_PATH)
            if rent_model_path.exists():
                self.models["rent"] = joblib.load(rent_model_path / "model.pkl")
                self.model_info["rent"] = {
                    "loaded": True,
                    "path": str(rent_model_path)
                }
                print("✅ Rent model loaded")
            else:
                print("⚠️  Rent model not found")
                self.model_info["rent"] = {"loaded": False}
            
            # Load sale model
            sale_model_path = Path(settings.SALE_MODEL_PATH)
            if sale_model_path.exists():
                self.models["sale"] = joblib.load(sale_model_path / "model.pkl")
                self.model_info["sale"] = {
                    "loaded": True,
                    "path": str(sale_model_path)
                }
                print("✅ Sale model loaded")
            else:
                print("⚠️  Sale model not found")
                self.model_info["sale"] = {"loaded": False}
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            
    def load_from_mlflow(self, model_name: str, model_version: str = "latest"):
        """Load model from MLflow registry"""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            
            if model_version == "latest":
                model_uri = f"models:/{model_name}/latest"
            else:
                model_uri = f"models:/{model_name}/{model_version}"
            
            model = mlflow.sklearn.load_model(model_uri)
            self.models[model_name] = model
            self.model_info[model_name] = {
                "loaded": True,
                "source": "mlflow",
                "version": model_version
            }
            print(f"✅ Model {model_name} loaded from MLflow")
            
        except Exception as e:
            print(f"❌ Error loading model from MLflow: {e}")
    
    def get_model(self, transaction_type: str) -> Optional[any]:
        """Get model by transaction type"""
        return self.models.get(transaction_type)
    
    def is_loaded(self) -> bool:
        """Check if any models are loaded"""
        return len(self.models) > 0
    
    def get_models_info(self) -> dict:
        """Get information about loaded models"""
        return self.model_info


# Global model manager instance
model_manager = ModelManager()
