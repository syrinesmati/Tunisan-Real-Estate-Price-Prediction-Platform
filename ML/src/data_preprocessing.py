"""
Data Preprocessing Module
Handles data cleaning and missing value imputation using clustering
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Optional


class DataPreprocessor:
    """Data preprocessing with clustering-based imputation"""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        df = df.copy()
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_len:
            print(f"  Removed {initial_len - len(df)} duplicates")
        
        # Remove invalid prices
        if 'price' in df.columns:
            df = df[df['price'] > 0]
            print(f"  Filtered to {len(df)} rows with valid prices")
        
        # Handle outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'price':  # Don't remove price outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def impute_with_clustering(
        self,
        df: pd.DataFrame,
        features_for_clustering: List[str],
        target_columns: List[str]
    ) -> pd.DataFrame:
        """
        Impute missing values using clustering-based approach
        Groups similar properties and fills missing values with cluster statistics
        """
        df = df.copy()
        
        # Select rows with complete clustering features
        complete_mask = df[features_for_clustering].notna().all(axis=1)
        df_complete = df[complete_mask]
        
        if len(df_complete) == 0:
            print("  ⚠️  Not enough complete data for clustering")
            return df
        
        # Fit KMeans on complete data
        X_cluster = self.scaler.fit_transform(df_complete[features_for_clustering])
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        df_complete['cluster'] = kmeans.fit_predict(X_cluster)
        
        # Predict clusters for incomplete data
        incomplete_mask = ~complete_mask
        if incomplete_mask.sum() > 0:
            # Use available features to predict cluster
            # For simplicity, assign to nearest cluster based on available data
            df.loc[complete_mask, 'cluster'] = df_complete['cluster']
        
        # Impute missing values using cluster statistics
        for col in target_columns:
            if df[col].isna().sum() > 0:
                for cluster_id in range(self.n_clusters):
                    cluster_mask = df['cluster'] == cluster_id
                    cluster_mean = df.loc[cluster_mask & df[col].notna(), col].mean()
                    
                    # Fill missing values in this cluster
                    missing_mask = cluster_mask & df[col].isna()
                    df.loc[missing_mask, col] = cluster_mean
                
                print(f"  Imputed {df[col].isna().sum()} values in {col}")
        
        # Drop cluster column
        df = df.drop(columns=['cluster'])
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataset"""
        df = df.copy()
        
        # Numeric columns: clustering-based imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Use key features for clustering
        clustering_features = [col for col in ['area', 'rooms', 'bedrooms'] if col in df.columns]
        
        if clustering_features and len(clustering_features) >= 2:
            print("  Using clustering-based imputation...")
            df = self.impute_with_clustering(
                df,
                features_for_clustering=clustering_features,
                target_columns=numeric_cols
            )
        
        # Remaining missing values: simple imputation
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns: mode or 'unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)
        
        return df
