import numpy as np
import hdbscan
from ml_service.models.base_model import BaseClusterModel

class HDBSCANModel(BaseClusterModel):
    """HDBSCAN clustering model implementation"""
    
    def __init__(self, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0):
        super().__init__(name="hdbscan")
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            prediction_data=True  # Enable prediction data generation during fitting
        )
    
    def fit(self, features):
        """Train the HDBSCAN model on the provided features"""
        self.model.fit(features)
        self.labels_ = self.model.labels_
        self.probabilities_ = self.model.probabilities_
        self.fitted = True
        return self
    
    def predict(self, features):
        """Predict cluster labels for the provided features
        
        Note: HDBSCAN doesn't have a native predict method for new data,
        this is a simplified approach using the approximate_predict method
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use approximate prediction for HDBSCAN
        labels, strengths = hdbscan.approximate_predict(self.model, features)
        return labels
    
    def get_model_params(self):
        """Return the model parameters"""
        if not self.fitted:
            return {"status": "not_fitted"}
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
            "n_clusters": len(set(self.labels_)) - (1 if -1 in self.labels_ else 0),
            "outlier_percentage": 100 * np.sum(self.labels_ == -1) / len(self.labels_)
        }