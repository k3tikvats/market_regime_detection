import numpy as np
from sklearn.mixture import GaussianMixture
from ml_service.models.base_model import BaseClusterModel

class GMMModel(BaseClusterModel):
    """Gaussian Mixture Model implementation"""
    
    def __init__(self, n_components=5, covariance_type='full'):
        super().__init__(name="gmm")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=42
        )
    
    def fit(self, features):
        """Train the GMM model on the provided features"""
        self.model.fit(features)
        self.means_ = self.model.means_
        self.covariances_ = self.model.covariances_
        self.fitted = True
        return self
    
    def predict(self, features):
        """Predict cluster labels for the provided features"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(features)
    
    def predict_proba(self, features):
        """Return probability of each component for the provided features"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(features)
    
    def get_model_params(self):
        """Return the model parameters"""
        if not self.fitted:
            return {"status": "not_fitted"}
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "means": self.means_.tolist(),
            "bic": self.model.bic(features) if hasattr(self, 'features') else None,
            "aic": self.model.aic(features) if hasattr(self, 'features') else None
        }
