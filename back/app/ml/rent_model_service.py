"""
Rent Model Service - RandomForest pipeline from train5.py
"""
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from app.core.config import settings


class RentModelService:
    """Service for rent price prediction using train5 pipeline"""

    def __init__(self):
        self.model = None
        self.loaded = False
        self.last_error = None
        self.feature_columns = [
            "surface",
            "rooms",
            "bathrooms",
            "region",
            "property_type",
            "city",
            "price_segment",
            "has_piscine",
            "has_garage",
            "has_jardin",
            "has_terrasse",
            "has_ascenseur",
            "is_meuble",
            "has_chauffage",
            "has_climatisation",
        ]

    def load(self):
        """Load model pipeline from disk"""
        try:
            model_path = Path(settings.RENT_MODEL_PATH) / "model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")

            self.model = joblib.load(model_path)
            self.loaded = True
            self.last_error = None
            print("✅ Rent RandomForest model loaded successfully")
        except Exception as exc:
            self.loaded = False
            self.last_error = str(exc)
            print(f"❌ Error loading rent model: {exc}")
            raise

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        row = {
            "surface": float(payload["surface"]),
            "rooms": int(payload.get("rooms") or 0),
            "bathrooms": int(payload.get("bathrooms") or 0),
            "region": payload["region"],
            "property_type": payload["property_type"],
            "city": payload["city"],
            "price_segment": payload.get("price_segment") or "mid",
            "has_piscine": bool(payload.get("has_piscine")),
            "has_garage": bool(payload.get("has_garage")),
            "has_jardin": bool(payload.get("has_jardin")),
            "has_terrasse": bool(payload.get("has_terrasse")),
            "has_ascenseur": bool(payload.get("has_ascenseur")),
            "is_meuble": bool(payload.get("is_meuble")),
            "has_chauffage": bool(payload.get("has_chauffage")),
            "has_climatisation": bool(payload.get("has_climatisation")),
        }

        X = pd.DataFrame([[row[col] for col in self.feature_columns]], columns=self.feature_columns)
        prediction = float(self.model.predict(X)[0])

        return {
            "predicted_price": round(prediction, 2),
            "currency": "TND",
            "model": "rent_random_forest_train5",
        }


rent_model_service = RentModelService()
