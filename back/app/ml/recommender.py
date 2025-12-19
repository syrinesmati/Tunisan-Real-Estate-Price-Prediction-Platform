"""
Property Recommendation System using KNN
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import Dict, List
from app.models.schemas import PropertyFeatures, Property


class PropertyRecommender:
    """KNN-based property recommendation system"""
    
    def __init__(self):
        self.knn_model = None
        self.scaler = StandardScaler()
        self.properties_df = None
        self.feature_columns = [
            'area', 'rooms', 'bedrooms', 'bathrooms',
            'has_parking', 'has_garden', 'is_furnished'
        ]
        
        # Load scraped properties data
        self._load_properties_data()
        
    def _load_properties_data(self):
        """Load properties from scraped data"""
        # TODO: Load from database or CSV file
        # For now, create empty dataframe
        self.properties_df = pd.DataFrame()
        
        try:
            # Try to load from CSV if exists
            import os
            data_path = "data/scraped_properties.csv"
            if os.path.exists(data_path):
                self.properties_df = pd.read_csv(data_path)
                self._fit_knn_model()
        except Exception as e:
            print(f"Could not load properties data: {e}")
    
    def _fit_knn_model(self):
        """Fit KNN model on available properties"""
        if self.properties_df.empty:
            return
        
        # Prepare features
        X = self.properties_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit KNN
        self.knn_model = NearestNeighbors(
            n_neighbors=6,  # +1 because query point might be in dataset
            algorithm='ball_tree',
            metric='euclidean'
        )
        self.knn_model.fit(X_scaled)
    
    def find_similar(
        self,
        property_features: PropertyFeatures,
        n_recommendations: int = 5
    ) -> Dict[str, List]:
        """Find similar properties using KNN"""
        
        if self.properties_df.empty or self.knn_model is None:
            return {
                "properties": [],
                "scores": [],
                "message": "No properties available for recommendations yet"
            }
        
        # Prepare query features
        query_features = {
            'area': property_features.area,
            'rooms': property_features.rooms or 0,
            'bedrooms': property_features.bedrooms or 0,
            'bathrooms': property_features.bathrooms or 0,
            'has_parking': int(property_features.has_parking or False),
            'has_garden': int(property_features.has_garden or False),
            'is_furnished': int(property_features.is_furnished or False),
        }
        
        query_df = pd.DataFrame([query_features])
        query_scaled = self.scaler.transform(query_df)
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(
            query_scaled,
            n_neighbors=n_recommendations + 1
        )
        
        # Convert to similarity scores (1 / (1 + distance))
        similarities = 1 / (1 + distances[0])
        
        # Get recommended properties
        recommended_properties = []
        for idx in indices[0][1:n_recommendations + 1]:  # Skip first (query itself)
            prop_data = self.properties_df.iloc[idx]
            
            property_obj = Property(
                id=str(prop_data.get('id', '')),
                title=prop_data.get('title', 'Property'),
                price=float(prop_data.get('price', 0)),
                governorate=prop_data.get('governorate', ''),
                city=prop_data.get('city', ''),
                property_type=prop_data.get('property_type', ''),
                transaction_type=property_features.transaction_type,
                area=float(prop_data.get('area', 0)) if pd.notna(prop_data.get('area')) else None,
                rooms=int(prop_data.get('rooms', 0)) if pd.notna(prop_data.get('rooms')) else None,
                bedrooms=int(prop_data.get('bedrooms', 0)) if pd.notna(prop_data.get('bedrooms')) else None,
                url=prop_data.get('url', ''),
                image_url=prop_data.get('image_url'),
                description=prop_data.get('description'),
                posted_date=prop_data.get('posted_date')
            )
            recommended_properties.append(property_obj)
        
        return {
            "properties": recommended_properties,
            "scores": similarities[1:n_recommendations + 1].tolist()
        }
    
    def get_stats(self) -> dict:
        """Get statistics about available properties"""
        if self.properties_df.empty:
            return {
                "total_properties": 0,
                "status": "no_data"
            }
        
        return {
            "total_properties": len(self.properties_df),
            "by_type": self.properties_df.groupby('property_type').size().to_dict(),
            "by_governorate": self.properties_df.groupby('governorate').size().to_dict(),
            "status": "ready"
        }
