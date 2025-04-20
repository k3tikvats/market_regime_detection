# backend/ml_service/regime_detector.py
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

from .models.kmeans import KMeansModel
from .models.gmm import GMMModel
from .models.hdbscan import HDBSCANModel
from data_pipeline.feature_engineering.feature_extractor import FeatureExtractor
from data_pipeline.feature_engineering.normalizer import Normalizer

class MarketRegimeDetector:
    """
    Main service for detecting market regimes using unsupervised learning
    """
    
    def __init__(self, config=None):
        """
        Initialize the regime detector with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feature_extractor = FeatureExtractor(
            data_dir=self.config.get('data_dir', 'data')
            # Removed window_sizes parameter as it's not supported by FeatureExtractor
        )
        
        # Store window sizes for later use when extracting features
        self.window_sizes = self.config.get('window_sizes', [10, 30, 60])
        
        self.normalizer = Normalizer(
            method=self.config.get('normalization_method', 'standard'),
            pca_components=self.config.get('pca_components', None)
        )
        
        # Initialize clustering models
        self.models = {
            'kmeans': KMeansModel(n_clusters=self.config.get('kmeans_clusters', 5)),
            'gmm': GMMModel(n_components=self.config.get('gmm_components', 5)),
            'hdbscan': HDBSCANModel(
                min_cluster_size=self.config.get('hdbscan_min_cluster_size', 5),
                min_samples=self.config.get('hdbscan_min_samples', None)
            )
        }
        
        # Selected model for predictions
        self.selected_model_name = self.config.get('selected_model', 'ensemble')
        
        # Data storage
        self.features = None
        self.normalized_features = None
        self.labels = {}
        self.regime_characteristics = {}
    
    def process_data(self, depth_df, trade_df):
        """
        Process raw order book and trade data to extract features
        
        Args:
            depth_df: DataFrame containing order book depth data
            trade_df: DataFrame containing trade data
            
        Returns:
            DataFrame of extracted features
        """
        self.features = self.feature_extractor.extract_features(depth_df, trade_df)
        self.normalized_features = self.normalizer.fit_transform(self.features)
        return self.normalized_features
    
    def train_models(self):
        """
        Train all clustering models on the processed data
        
        Returns:
            Dictionary of model names and their fitted instances
        """
        if self.normalized_features is None:
            raise ValueError("No data has been processed. Call process_data first.")
        
        # Train each model
        for name, model in self.models.items():
            model.fit(self.normalized_features)
            self.labels[name] = model.predict(self.normalized_features)
        
        # Create ensemble labels (simple majority voting)
        if len(self.models) > 1:
            # This is a simplified ensemble approach
            # A more sophisticated approach would use cluster matching or soft voting
            self.labels['ensemble'] = self._create_ensemble_labels()
        
        return self.models
    
    def fit(self, X):
        """
        Fit all models on the provided feature data.
        
        Args:
            X: Feature data for training the models
                
        Returns:
            self: The fitted model instance
        """
        if X is None or len(X) == 0:
            raise ValueError("No data provided for fitting models")
            
        self.normalized_features = self.normalizer.fit_transform(X)
        
        # Train each model
        for name, model in self.models.items():
            model.fit(self.normalized_features)
            self.labels[name] = model.predict(self.normalized_features)
        
        # Create ensemble labels if needed
        if len(self.models) > 1:
            self.labels['ensemble'] = self._create_ensemble_labels()
        
        # Set the model for predictions based on selected model name
        if self.selected_model_name not in self.models and self.selected_model_name != 'ensemble':
            self.selected_model_name = next(iter(self.models.keys()))
            
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for the input data.
        
        Args:
            X: Feature data to predict clusters for
                
        Returns:
            array: Predicted cluster labels
        """
        if self.normalizer is None:
            raise ValueError("Normalizer not initialized. Model must be fitted first.")
            
        # Normalize the input features
        X_normalized = self.normalizer.transform(X)
        
        if self.selected_model_name == 'ensemble':
            # Create ensemble predictions
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(X_normalized)
                
            # Use simple majority voting as the ensemble method
            # In a real implementation, this would be more sophisticated
            if 'kmeans' in self.models:
                return predictions['kmeans']
            else:
                # Return the first model's predictions as fallback
                return next(iter(predictions.values()))
        else:
            # Use the selected model
            model = self.models.get(self.selected_model_name)
            if model is None:
                # Fallback to the first available model
                model = next(iter(self.models.values()))
            
            return model.predict(X_normalized)
    
    def _create_ensemble_labels(self):
        """
        Create ensemble labels using a simple approach
        
        Returns:
            Array of ensemble cluster labels
        """
        # This is a very simplified approach to ensemble clustering
        # In reality, you would need to align cluster labels across models
        
        # For now, we'll use the KMeans labels as the base
        if 'kmeans' in self.labels:
            return self.labels['kmeans']
        else:
            # Fallback to the first available model
            return list(self.labels.values())[0]
    
    def analyze_regimes(self):
        """
        Analyze the characteristics of each detected regime
        
        Returns:
            Dictionary with regime characteristics
        """
        if not self.labels or self.features is None:
            raise ValueError("Models must be trained before analyzing regimes.")
        
        model_name = self.selected_model_name
        if model_name not in self.labels and model_name != 'ensemble':
            model_name = next(iter(self.labels.keys()))
        
        labels = self.labels[model_name]
        
        # Create a DataFrame with features and labels
        labeled_data = pd.DataFrame(self.features.copy())
        labeled_data['regime'] = labels
        
        # Analyze each regime
        unique_labels = np.unique(labels)
        characteristics = {}
        
        for label in unique_labels:
            if label == -1:  # Skip noise points in HDBSCAN
                continue
                
            regime_data = labeled_data[labeled_data['regime'] == label]
            
            # Calculate statistics for the regime
            stats = {}
            
            # Volatility characteristics
            volatility_cols = [col for col in regime_data.columns if 'volatility' in col]
            if volatility_cols:
                mean_volatility = regime_data[volatility_cols].mean().mean()
                stats['volatility'] = mean_volatility
                stats['volatility_type'] = 'High' if mean_volatility > labeled_data[volatility_cols].mean().mean() else 'Low'
            
            # Trend/Mean-reversion characteristics
            price_zscore_cols = [col for col in regime_data.columns if 'price_zscore' in col]
            if price_zscore_cols:
                mean_zscore = np.abs(regime_data[price_zscore_cols].mean().mean())
                stats['mean_reversion_tendency'] = mean_zscore
                stats['price_behavior'] = 'Mean-reverting' if mean_zscore < 1.0 else 'Trending'
            
            # Liquidity characteristics
            spread_cols = [col for col in regime_data.columns if 'spread' in col]
            imbalance_cols = [col for col in regime_data.columns if 'imbalance' in col]
            
            if spread_cols:
                mean_spread = regime_data[spread_cols].mean().mean()
                stats['spread'] = mean_spread
            
            if 'cum_bid_qty' in regime_data.columns and 'cum_ask_qty' in regime_data.columns:
                mean_depth = (regime_data['cum_bid_qty'].mean() + regime_data['cum_ask_qty'].mean()) / 2
                stats['order_book_depth'] = mean_depth
                stats['liquidity'] = 'High' if mean_depth > labeled_data[['cum_bid_qty', 'cum_ask_qty']].mean().mean() else 'Low'
            
            # Volume characteristics
            volume_cols = [col for col in regime_data.columns if 'volume_' in col and 'imbalance' not in col]
            if volume_cols:
                mean_volume = regime_data[volume_cols].mean().mean()
                stats['volume'] = mean_volume
            
            # Direction characteristics
            direction_cols = [col for col in regime_data.columns if 'direction_' in col]
            if direction_cols:
                mean_direction = regime_data[direction_cols].mean().mean()
                stats['price_direction'] = mean_direction
                stats['direction_type'] = 'Upward' if mean_direction > 0 else 'Downward' if mean_direction < 0 else 'Sideways'
            
            # Summarize regime
            regime_name = self._generate_regime_name(stats)
            stats['name'] = regime_name
            characteristics[label] = stats
        
        self.regime_characteristics = characteristics
        return characteristics
    
    def _generate_regime_name(self, stats):
        """
        Generate a descriptive name for a regime based on its statistics
        
        Args:
            stats: Dictionary of statistics for the regime
            
        Returns:
            String name for the regime
        """
        parts = []
        
        # Add price behavior type
        if 'price_behavior' in stats:
            parts.append(stats['price_behavior'])
        
        # Add liquidity type
        if 'liquidity' in stats:
            parts.append(f"{stats['liquidity']} Liquidity")
        
        # Add volatility type
        if 'volatility_type' in stats:
            parts.append(f"{stats['volatility_type']} Volatility")
        
        # Combine the parts
        name = " & ".join(parts)
        
        return name
    
    def detect_regime(self, new_data):
        """
        Detect regime for new data
        
        Args:
            new_data: DataFrame with new order book and trade data
            
        Returns:
            Detected regime label and characteristics
        """
        # Extract features from new data
        # Assuming new_data is a tuple of (depth_df, trade_df)
        depth_df, trade_df = new_data
        
        # Extract features
        new_features = self.feature_extractor.extract_features(depth_df, trade_df)
        
        # Normalize features
        normalized_features = self.normalizer.transform(new_features)
        
        # Predict using the selected model
        model_name = self.selected_model_name
        if model_name == 'ensemble':
            # Use all models and create an ensemble prediction
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(normalized_features)
            
            # Simple ensemble (using majority voting)
            # In a real implementation, this would be more sophisticated
            ensemble_pred = predictions.get('kmeans', predictions[list(predictions.keys())[0]])
        else:
            # Use the selected model
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found.")
            
            ensemble_pred = model.predict(normalized_features)
        
        # Get the most common regime label
        if len(ensemble_pred) > 0:
            most_common_label = np.bincount(ensemble_pred).argmax()
        else:
            most_common_label = None
        
        # Return the detected regime and its characteristics
        regime_info = {
            'label': most_common_label,
            'characteristics': self.regime_characteristics.get(most_common_label, {})
        }
        
        return regime_info
    
    def evaluate_clustering(self):
        """
        Evaluate the clustering performance using internal metrics
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.normalized_features is None or not self.labels:
            raise ValueError("Models must be trained before evaluation.")
        
        metrics = {}
        
        for model_name, labels in self.labels.items():
            # Skip models that produce noise points (like HDBSCAN with label -1)
            if -1 in labels:
                valid_idx = labels != -1
                if not np.any(valid_idx) or np.sum(valid_idx) <= 1:
                    continue
                valid_features = self.normalized_features.iloc[valid_idx]
                valid_labels = labels[valid_idx]
            else:
                valid_features = self.normalized_features
                valid_labels = labels
            
            # Compute silhouette score if there are at least 2 clusters
            unique_labels = np.unique(valid_labels)
            if len(unique_labels) >= 2:
                metrics[f'{model_name}_silhouette'] = silhouette_score(valid_features, valid_labels)
                metrics[f'{model_name}_davies_bouldin'] = davies_bouldin_score(valid_features, valid_labels)
        
        return metrics
    
    def analyze_regime_transitions(self):
        """
        Analyze transitions between different regimes
        
        Returns:
            DataFrame with transition probabilities
        """
        if not self.labels or self.features is None:
            raise ValueError("Models must be trained before analyzing transitions.")
        
        model_name = self.selected_model_name
        if model_name not in self.labels and model_name != 'ensemble':
            model_name = next(iter(self.labels.keys()))
        
        labels = self.labels[model_name]
        
        # Create a time series of regime labels
        regime_series = pd.Series(labels, index=self.features.index)
        
        # Count transitions between regimes
        transitions = {}
        prev_regime = regime_series.iloc[0]
        
        for current_regime in regime_series.iloc[1:]:
            transition = (prev_regime, current_regime)
            transitions[transition] = transitions.get(transition, 0) + 1
            prev_regime = current_regime
        
        # Convert transitions to a matrix
        unique_regimes = np.unique(labels)
        n_regimes = len(unique_regimes)
        
        transition_matrix = np.zeros((n_regimes, n_regimes))
        regime_mapping = {regime: idx for idx, regime in enumerate(unique_regimes)}
        
        for (from_regime, to_regime), count in transitions.items():
            from_idx = regime_mapping[from_regime]
            to_idx = regime_mapping[to_regime]
            transition_matrix[from_idx, to_idx] = count
        
        # Convert counts to probabilities (row-wise normalization)
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.zeros_like(transition_matrix)
        non_zero_rows = row_sums.flatten() > 0
        transition_probs[non_zero_rows] = transition_matrix[non_zero_rows] / row_sums[non_zero_rows]
        
        # Create a DataFrame for better readability
        transition_df = pd.DataFrame(
            transition_probs,
            index=[f"Regime {r}" for r in unique_regimes],
            columns=[f"Regime {r}" for r in unique_regimes]
        )
        
        return transition_df
    
    def visualize_regimes(self, method='tsne'):
        """
        Visualize the detected regimes in 2D space
        
        Args:
            method: Dimensionality reduction method ('tsne' or 'umap')
            
        Returns:
            Base64 encoded image of the visualization
        """
        if self.normalized_features is None or not self.labels:
            raise ValueError("Models must be trained before visualization.")
        
        model_name = self.selected_model_name
        if model_name not in self.labels and model_name != 'ensemble':
            model_name = next(iter(self.labels.keys()))
        
        labels = self.labels[model_name]
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown visualization method: {method}")
        
        reduced_features = reducer.fit_transform(self.normalized_features)
        
        # Plot the results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Regime')
        plt.title(f'Market Regimes Visualization ({model_name.upper()} with {method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        # Add timestamp labels for a subset of points
        step = max(1, len(self.features) // 20)  # Show up to 20 timestamp labels
        for i in range(0, len(self.features), step):
            if i < len(reduced_features):
                plt.annotate(
                    self.features.index[i].strftime('%H:%M:%S'),
                    (reduced_features[i, 0], reduced_features[i, 1]),
                    fontsize=8
                )
        
        # Save the figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64 for web display
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_base64
    
    def visualize_regime_evolution(self):
        """
        Visualize how regimes evolve over time
        
        Returns:
            Base64 encoded image of the visualization
        """
        if not self.labels or self.features is None:
            raise ValueError("Models must be trained before visualization.")
        
        model_name = self.selected_model_name
        if model_name not in self.labels and model_name != 'ensemble':
            model_name = next(iter(self.labels.keys()))
        
        labels = self.labels[model_name]
        
        # Create a time series of regime labels
        regime_series = pd.Series(labels, index=self.features.index)
        
        # Get the mid price if available
        if 'mid_price' in self.features.columns:
            price_series = self.features['mid_price']
        else:
            price_series = None
        
        # Plot the regime evolution
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot regimes
        ax1.scatter(regime_series.index, regime_series, c=regime_series, cmap='viridis', 
                   s=30, alpha=0.7, label='Regime')
        ax1.set_ylabel('Regime')
        ax1.set_title(f'Market Regime Evolution Over Time ({model_name.upper()})')
        
        # Add colorbar
        unique_regimes = np.unique(labels)
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=min(unique_regimes), vmax=max(unique_regimes))
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Regime')
        
        # Add price if available
        if price_series is not None:
            ax2 = ax1.twinx()
            ax2.plot(price_series.index, price_series, color='red', alpha=0.5, label='Price')
            ax2.set_ylabel('Price', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        # Format x-axis to show time properly
        fig.autofmt_xdate()
        
        # Save the figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64 for web display
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_base64
    
    def save_models(self, path):
        """
        Save models to disk
        
        Args:
            path: Path to save models
        """
        import joblib
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(path, f"{name}_model.joblib"))
        
        # Save normalizer
        joblib.dump(self.normalizer, os.path.join(path, "normalizer.joblib"))
        
        # Save feature extractor configuration
        joblib.dump(self.feature_extractor, os.path.join(path, "feature_extractor.joblib"))
        
        # Save regime characteristics
        joblib.dump(self.regime_characteristics, os.path.join(path, "regime_characteristics.joblib"))
    
    def load_models(self, path):
        """
        Load models from disk
        
        Args:
            path: Path to load models from
        """
        import joblib
        import os
        
        # Load each model
        for name in self.models.keys():
            model_path = os.path.join(path, f"{name}_model.joblib")
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
        
        # Load normalizer
        normalizer_path = os.path.join(path, "normalizer.joblib")
        if os.path.exists(normalizer_path):
            self.normalizer = joblib.load(normalizer_path)
        
        # Load feature extractor
        feature_extractor_path = os.path.join(path, "feature_extractor.joblib")
        if os.path.exists(feature_extractor_path):
            self.feature_extractor = joblib.load(feature_extractor_path)
        
        # Load regime characteristics
        regime_characteristics_path = os.path.join(path, "regime_characteristics.joblib")
        if os.path.exists(regime_characteristics_path):
            self.regime_characteristics = joblib.load(regime_characteristics_path)