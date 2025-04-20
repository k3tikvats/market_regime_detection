# backend/ml_service/data_pipeline/feature_engineering/normalizer.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

class Normalizer:
    """
    Normalizes feature data using various methods and optionally applies dimensionality reduction.
    """
    
    def __init__(self, method='standard', pca_components=None):
        """
        Initialize the normalizer.
        
        Args:
            method (str): Normalization method ('standard', 'minmax', 'robust')
            pca_components (int): Number of PCA components for dimensionality reduction
        """
        self.method = method
        self.pca_components = pca_components
        self.scaler = None
        self.pca = None
        self.fitted = False
        self.feature_names = None
    
    def fit(self, features):
        """
        Fit the normalizer to the given features.
        
        Args:
            features (pd.DataFrame): Features to fit the normalizer to
            
        Returns:
            self: The fitted normalizer
        """
        if isinstance(features, pd.DataFrame):
            self.feature_names = features.columns
            features_array = features.values
        else:
            features_array = features
        
        # Initialize and fit the scaler
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        # Fit the scaler
        self.scaler.fit(features_array)
        
        # Apply PCA if specified
        if self.pca_components is not None:
            self.pca = PCA(n_components=self.pca_components)
            normalized_features = self.scaler.transform(features_array)
            self.pca.fit(normalized_features)
        
        self.fitted = True
        return self
    
    def transform(self, features):
        """
        Transform features using the fitted normalizer.
        
        Args:
            features (pd.DataFrame/ndarray): Features to transform
            
        Returns:
            pd.DataFrame/ndarray: Normalized features
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        is_dataframe = isinstance(features, pd.DataFrame)
        if is_dataframe:
            features_array = features.values
            original_index = features.index
        else:
            features_array = features
            original_index = None
        
        # Apply normalization
        normalized_features = self.scaler.transform(features_array)
        
        # Apply PCA if specified
        if self.pca is not None:
            normalized_features = self.pca.transform(normalized_features)
            
            # Create new feature names for PCA components
            if self.feature_names is not None:
                self.transformed_feature_names = [f"PC{i+1}" for i in range(normalized_features.shape[1])]
            else:
                self.transformed_feature_names = None
        else:
            self.transformed_feature_names = self.feature_names
        
        # Return as DataFrame if input was DataFrame
        if is_dataframe:
            if self.transformed_feature_names is not None:
                return pd.DataFrame(
                    normalized_features, 
                    index=original_index,
                    columns=self.transformed_feature_names[:normalized_features.shape[1]]
                )
            else:
                return pd.DataFrame(normalized_features, index=original_index)
        else:
            return normalized_features
    
    def fit_transform(self, features):
        """
        Fit to data, then transform it.
        
        Args:
            features (pd.DataFrame/ndarray): Features to fit and transform
            
        Returns:
            pd.DataFrame/ndarray: Normalized features
        """
        return self.fit(features).transform(features)
    
    def inverse_transform(self, normalized_features):
        """
        Transform data back to its original space.
        
        Args:
            normalized_features (pd.DataFrame/ndarray): Normalized features
            
        Returns:
            pd.DataFrame/ndarray: Features in original space
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        
        is_dataframe = isinstance(normalized_features, pd.DataFrame)
        if is_dataframe:
            features_array = normalized_features.values
            original_index = normalized_features.index
        else:
            features_array = normalized_features
            original_index = None
        
        # Inverse transform PCA if applied
        if self.pca is not None:
            features_array = self.pca.inverse_transform(features_array)
        
        # Inverse transform scaling
        original_features = self.scaler.inverse_transform(features_array)
        
        # Return as DataFrame if input was DataFrame
        if is_dataframe:
            if self.feature_names is not None:
                return pd.DataFrame(
                    original_features, 
                    index=original_index,
                    columns=self.feature_names[:original_features.shape[1]]
                )
            else:
                return pd.DataFrame(original_features, index=original_index)
        else:
            return original_features