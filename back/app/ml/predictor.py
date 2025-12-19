"""
Price Prediction Logic
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from app.models.schemas import PropertyFeatures


class PricePredictor:
    """Handles price prediction logic"""
    
    def __init__(self):
        self.feature_columns = [
            'area', 'rooms', 'bedrooms', 'bathrooms', 'floor',
            'has_elevator', 'has_parking', 'has_garden', 'has_pool',
            'is_furnished', 'construction_year'
        ]
        
    def prepare_features(self, features: PropertyFeatures) -> pd.DataFrame:
        """Convert PropertyFeatures to model input format"""
        
        # Extract numeric and boolean features
        feature_dict = {
            'area': features.area,
            'rooms': features.rooms or 0,
            'bedrooms': features.bedrooms or 0,
            'bathrooms': features.bathrooms or 0,
            'floor': features.floor or 0,
            'has_elevator': int(features.has_elevator or False),
            'has_parking': int(features.has_parking or False),
            'has_garden': int(features.has_garden or False),
            'has_pool': int(features.has_pool or False),
            'is_furnished': int(features.is_furnished or False),
            'construction_year': features.construction_year or 2000,
        }
        
        # TODO: Add encoding for categorical features (governorate, city, property_type)
        # This will be implemented once we have the actual trained models
        
        return pd.DataFrame([feature_dict])
    
    def predict(self, model: Any, features: PropertyFeatures) -> Dict[str, Any]:
        """Make price prediction"""
        
        # Prepare features
        X = self.prepare_features(features)
        
        # Make prediction
        predicted_price = float(model.predict(X)[0])
        
        # Calculate confidence interval (if model supports it)
        try:
            # For ensemble models with prediction intervals
            std_dev = predicted_price * 0.15  # Assume 15% standard deviation
            confidence_interval = {
                "lower": predicted_price - (1.96 * std_dev),
                "upper": predicted_price + (1.96 * std_dev)
            }
        except:
            confidence_interval = {
                "lower": predicted_price * 0.85,
                "upper": predicted_price * 1.15
            }
        
        # Generate market insights
        insights = self._generate_insights(features, predicted_price)
        
        return {
            "predicted_price": predicted_price,
            "confidence_interval": confidence_interval,
            "insights": insights
        }
    
    def _generate_insights(
        self,
        features: PropertyFeatures,
        predicted_price: float
    ) -> Dict[str, Any]:
        """Generate market insights based on property features"""
        
        insights = {
            "price_per_sqm": predicted_price / features.area if features.area else 0,
            "location": f"{features.city}, {features.governorate}",
            "property_type": features.property_type,
            "transaction_type": features.transaction_type.value
        }
        
        # Add value-adding features count
        premium_features = [
            features.has_elevator,
            features.has_parking,
            features.has_garden,
            features.has_pool,
            features.is_furnished
        ]
        insights["premium_features_count"] = sum(1 for f in premium_features if f)
        
        return insights
