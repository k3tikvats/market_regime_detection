import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ml_service.models.base_model import BaseClusterModel

class KMeansModel(BaseClusterModel):
    """K-Means clustering model implementation"""
    
    def __init__(self, n_clusters=5):
        super().__init__(name="kmeans")
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
    
    def fit(self, features):
        """Train the K-Means model on the provided features"""
        self.model.fit(features)
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        self.fitted = True
        return self
    
    def predict(self, features):
        """Predict cluster labels for the provided features"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(features)
    
    def get_model_params(self):
        """Return the model parameters"""
        if not self.fitted:
            return {"status": "not_fitted"}
        return {
            "n_clusters": self.n_clusters,
            "cluster_centers": self.cluster_centers_.tolist(),
            "inertia": self.model.inertia_
        }
    
    def find_optimal_clusters(self, features, max_clusters=10):
        """Find optimal number of clusters using the elbow method and silhouette score"""
        inertia_values = []
        silhouette_values = []
        k_values = range(2, max_clusters + 1)
        
        for k in k_values:
            temp_model = KMeans(n_clusters=k, random_state=42)
            temp_model.fit(features)
            inertia_values.append(temp_model.inertia_)
            
            # Calculate silhouette score (skip k=1 as it's not defined)
            if k > 1:
                labels = temp_model.labels_
                silhouette_values.append(silhouette_score(features, labels))
        
        # Find elbow point (not implemented here, would need a more sophisticated approach)
        # For now, we can use the max silhouette score as a heuristic
        best_k = k_values[np.argmax(silhouette_values)] if silhouette_values else 3
        
        # Update the model with the optimal k
        self.n_clusters = best_k
        self.model = KMeans(n_clusters=best_k, random_state=42)
        
        return {
            "k_values": list(k_values),
            "inertia_values": inertia_values,
            "silhouette_values": silhouette_values,
            "optimal_k": best_k
        }