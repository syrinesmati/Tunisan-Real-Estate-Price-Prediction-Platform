"""
Feature Engineering Module
Creates new features from raw data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder


class FeatureEngineer:
    """Feature engineering for real estate data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.target_encoders = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        df = df.copy()
        
        # Age of property
        if 'construction_year' in df.columns:
            current_year = 2025
            df['property_age'] = current_year - df['construction_year']
            df['is_new_construction'] = (df['property_age'] <= 2).astype(int)
        
        # Area-based features
        if 'area' in df.columns:
            if 'rooms' in df.columns:
                df['area_per_room'] = df['area'] / (df['rooms'] + 1)
            
            if 'bedrooms' in df.columns:
                df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 1)
        
        # Feature counts
        feature_cols = ['has_elevator', 'has_parking', 'has_garden', 'has_pool', 'is_furnished']
        available_features = [col for col in feature_cols if col in df.columns]
        if available_features:
            df['total_amenities'] = df[available_features].sum(axis=1)
        
        # Location features
        if 'governorate' in df.columns:
            # Major cities indicator
            major_cities = ['Tunis', 'Sfax', 'Sousse', 'Ariana', 'Ben Arous']
            df['is_major_city'] = df['governorate'].isin(major_cities).astype(int)
        
        # Property type value indicator
        if 'property_type' in df.columns:
            premium_types = ['villa', 'duplex']
            df['is_premium_type'] = df['property_type'].isin(premium_types).astype(int)
        
        # Encode categorical variables
        df = self._encode_categoricals(df)
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        df = df.copy()
        
        # Columns to encode
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Exclude ID columns and other non-feature columns
        exclude_cols = ['id', 'title', 'url', 'description', 'image_url', 'posted_date']
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        for col in categorical_cols:
            if col in df.columns:
                # Use Label Encoding for ordinal or low cardinality
                if df[col].nunique() <= 10:
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # For high cardinality, use target encoding (if price available)
                    if 'price' in df.columns:
                        te = TargetEncoder(cols=[col])
                        df[f'{col}_encoded'] = te.fit_transform(df[col], df['price'])
                        self.target_encoders[col] = te
                    else:
                        # Fallback to label encoding
                        le = LabelEncoder()
                        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                        self.label_encoders[col] = le
        
        # Drop original categorical columns
        df = df.drop(columns=categorical_cols)
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> list:
        """Get list of feature names"""
        exclude = ['price', 'id']
        return [col for col in df.columns if col not in exclude]
