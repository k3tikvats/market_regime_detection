# backend/ml_service/models/base_model.py
import numpy as np

class BaseClusterModel:
    """Base class for all clustering models"""
    
    def __init__(self, name="base_model"):
        self.name = name
        self.fitted = False
        self.features = None
    
    def fit(self, features):
        """Train the model on the provided features"""
        self.features = features
        self.fitted = True
        return self
    
    def predict(self, features):
        """Predict cluster labels for the provided features"""
        raise NotImplementedError("Subclasses must implement predict method")
    
    def get_model_params(self):
        """Return the model parameters"""
        return {"status": "not_fitted"} if not self.fitted else {}
    
    def get_labels(self):
        """Return the labels for the fitted data"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting labels")
        return self.predict(self.features)
    
    def evaluate(self, metrics=['silhouette']):
        """Evaluate the model using the specified metrics"""
        if not self.fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        from sklearn import metrics as sk_metrics
        labels = self.get_labels()
        results = {}
        
        # Skip noise points (label -1) if they exist
        valid_indices = np.where(labels != -1)[0] if -1 in labels else np.arange(len(labels))
        if len(valid_indices) < 2:
            return {"error": "Not enough valid points for evaluation"}
        
        valid_features = self.features.iloc[valid_indices] if hasattr(self.features, 'iloc') else self.features[valid_indices]
        valid_labels = labels[valid_indices]
        
        # Only evaluate if we have at least 2 clusters
        unique_labels = np.unique(valid_labels)
        if len(unique_labels) < 2:
            return {"error": "Need at least 2 clusters for evaluation"}
        
        for metric in metrics:
            if metric == 'silhouette':
                results['silhouette_score'] = sk_metrics.silhouette_score(valid_features, valid_labels)
            elif metric == 'davies_bouldin':
                results['davies_bouldin_score'] = sk_metrics.davies_bouldin_score(valid_features, valid_labels)
            elif metric == 'calinski_harabasz':
                results['calinski_harabasz_score'] = sk_metrics.calinski_harabasz_score(valid_features, valid_labels)
        
        return results