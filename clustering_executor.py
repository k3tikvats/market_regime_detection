# clustering_executor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import umap
import seaborn as sns
from pathlib import Path
import logging

# Setup logger
logger = logging.getLogger(__name__)

from ml_service.models.kmeans import KMeansModel
from ml_service.models.gmm import GMMModel
from ml_service.models.hdbscan import HDBSCANModel
from data_pipeline.feature_engineering.normalizer import Normalizer

class ClusteringExecutor:
    """Execute and evaluate clustering models for market regime detection"""
    
    def __init__(self, features_df):
        """
        Initialize the clustering executor
        
        Args:
            features_df: DataFrame containing the extracted features
        """
        self.features = features_df
        self.normalizer = None
        self.normalized_features = None
        self.models = {}
        self.labels = {}
        self.evaluation_results = {}
        self.output_dir = Path('./results')
        self.output_dir.mkdir(exist_ok=True)
    
    def normalize_features(self, method='standard', pca_components=None):
        """
        Normalize features using the specified method
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
            pca_components: Number of PCA components (None for no dimensionality reduction)
            
        Returns:
            DataFrame of normalized features
        """
        print(f"Normalizing features using {method}...")
        self.normalizer = Normalizer(method=method, pca_components=pca_components)
        self.normalized_features = self.normalizer.fit_transform(self.features)
        
        # Print normalization stats
        print(f"Normalized feature statistics:")
        print(f"Shape: {self.normalized_features.shape}")
        print(f"Mean: {self.normalized_features.mean().mean():.4f}")
        print(f"Std: {self.normalized_features.std().mean():.4f}")
        
        # If PCA was applied, print explained variance
        if pca_components is not None:
            explained_variance = self.normalizer.pca.explained_variance_ratio_
            print(f"PCA explained variance: {sum(explained_variance):.4f}")
            
            # Plot explained variance
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, len(explained_variance) + 1), explained_variance)
            plt.plot(range(1, len(explained_variance) + 1), 
                    np.cumsum(explained_variance), 'r-*')
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Explained Variance')
            plt.savefig(self.output_dir / 'pca_explained_variance.png')
            plt.close()
        
        return self.normalized_features
    
    def setup_models(self):
        """
        Initialize all clustering models
        """
        # KMeans models with different cluster counts
        for n_clusters in [3, 5, 7]:
            model_name = f"kmeans_{n_clusters}"
            self.models[model_name] = KMeansModel(n_clusters=n_clusters)
        
        # GMM models with different component counts and covariance types
        for n_components in [3, 5]:
            for cov_type in ['full', 'diag']:
                model_name = f"gmm_{n_components}_{cov_type}"
                self.models[model_name] = GMMModel(n_components=n_components, 
                                                 covariance_type=cov_type)
        
        # HDBSCAN models with different parameters
        for min_cluster_size in [5, 10, 15]:
            model_name = f"hdbscan_{min_cluster_size}"
            self.models[model_name] = HDBSCANModel(min_cluster_size=min_cluster_size)
        
        print(f"Initialized {len(self.models)} clustering models")
        return self.models
    
    def run_clustering(self):
        """
        Run all clustering models on the normalized data
        
        Returns:
            Dictionary of model labels
        """
        if self.normalized_features is None:
            raise ValueError("Features must be normalized before clustering")
        
        print("Running clustering models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.normalized_features)
            self.labels[name] = model.predict(self.normalized_features)
            
            # Convert labels to pandas Series with the same index as features
            if isinstance(self.normalized_features, pd.DataFrame):
                self.labels[name] = pd.Series(
                    self.labels[name], 
                    index=self.normalized_features.index,
                    name=name
                )
        
        return self.labels
    
    def evaluate_models(self):
        """
        Evaluate each trained clustering model.
        
        Returns:
            pd.DataFrame: Evaluation metrics for each model
        """
        logger.info("Evaluating clustering models...")
        
        # Store evaluation metrics
        results = []
        
        # Track if any models had more than one cluster
        valid_model_found = False
        
        for model_name, labels in self.labels.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Only evaluate if there's more than one cluster
            if len(np.unique(labels)) < 2:
                logger.warning(f"  Skipping {model_name} - only one cluster detected")
                continue
            
            valid_model_found = True
            
            # Calculate metrics
            silhouette = silhouette_score(self.features, labels) if len(np.unique(labels)) > 1 else np.nan
            db_score = davies_bouldin_score(self.features, labels) if len(np.unique(labels)) > 1 else np.nan
            ch_score = calinski_harabasz_score(self.features, labels) if len(np.unique(labels)) > 1 else np.nan
            
            # Store results
            results.append({
                'model': model_name,
                'n_clusters': len(np.unique(labels)),
                'silhouette_score': silhouette,
                'davies_bouldin_score': db_score,
                'calinski_harabasz_score': ch_score
            })
            
            logger.info(f"  Silhouette Score: {silhouette:.4f}")
            logger.info(f"  Davies-Bouldin Index: {db_score:.4f}")
            logger.info(f"  Calinski-Harabasz Index: {ch_score:.4f}")
        
        # Create dataframe with results
        if results:
            self.evaluation_results = pd.DataFrame(results)
            # Sort by silhouette score (higher is better)
            self.evaluation_results = self.evaluation_results.sort_values(
                by='silhouette_score', ascending=False
            ).reset_index(drop=True)
        else:
            # Create an empty DataFrame with the expected columns if no valid models were found
            self.evaluation_results = pd.DataFrame(columns=[
                'model', 'n_clusters', 'silhouette_score', 
                'davies_bouldin_score', 'calinski_harabasz_score'
            ])
            logger.warning("No valid clustering models found - all detected only one cluster")
        
        return self.evaluation_results
    
    def _plot_evaluation_results(self):
        """Plot the evaluation results for all models"""
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x='model', 
            y='silhouette_score', 
            data=self.evaluation_results.sort_values('silhouette_score', ascending=False)
        )
        plt.title('Silhouette Score by Model (higher is better)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'silhouette_scores.png')
        plt.close()
        
        # Plot Davies-Bouldin scores
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x='model', 
            y='davies_bouldin_score', 
            data=self.evaluation_results.sort_values('davies_bouldin_score', ascending=True)
        )
        plt.title('Davies-Bouldin Score by Model (lower is better)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'davies_bouldin_scores.png')
        plt.close()
        
        # Plot Calinski-Harabasz scores
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x='model', 
            y='calinski_harabasz_score', 
            data=self.evaluation_results.sort_values('calinski_harabasz_score', ascending=False)
        )
        plt.title('Calinski-Harabasz Score by Model (higher is better)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calinski_harabasz_scores.png')
        plt.close()
    
    def visualize_clusters(self, method='umap', model_name=None):
        """
        Visualize clusters in 2D space using dimensionality reduction
        
        Args:
            method: Dimensionality reduction method ('umap', 'pca', 'tsne')
            model_name: Name of the model to visualize (None for the best model)
            
        Returns:
            Figure with the visualization
        """
        if not self.labels:
            raise ValueError("Models must be run before visualization")
        
        if model_name is None:
            # Use the model with the highest silhouette score
            if self.evaluation_results is not None and not self.evaluation_results.empty:
                model_name = self.evaluation_results.iloc[0]['model']
            else:
                model_name = list(self.labels.keys())[0]
        
        print(f"Visualizing clusters for {model_name} using {method}...")
        
        # Get labels for the selected model
        labels = self.labels[model_name]
        labels_array = labels.values if isinstance(labels, pd.Series) else labels
        
        # Apply dimensionality reduction
        if method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(self.normalized_features)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embedding = reducer.fit_transform(self.normalized_features)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embedding = reducer.fit_transform(self.normalized_features)
        else:
            raise ValueError(f"Unknown visualization method: {method}")
        
        # Create a DataFrame for plotting
        vis_df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'cluster': labels_array
        })
        
        # Get time index if available
        if isinstance(self.normalized_features, pd.DataFrame):
            vis_df['time'] = self.normalized_features.index
        
        # Plot clusters
        plt.figure(figsize=(12, 10))
        
        # Determine colormap based on number of clusters
        n_clusters = len(np.unique(labels_array))
        if -1 in np.unique(labels_array):  # Account for noise points in HDBSCAN
            cmap = plt.cm.get_cmap('viridis', n_clusters - 1)
            scatter = plt.scatter(
                embedding[:, 0], 
                embedding[:, 1], 
                c=labels_array, 
                cmap=cmap,
                alpha=0.7,
                s=50
            )
            # Add a color for noise points (-1)
            cmap_list = [cmap(i) for i in range(cmap.N)]
            cmap_list.insert(0, (0.5, 0.5, 0.5, 1.0))  # Gray for noise
            scatter.set_cmap(plt.matplotlib.colors.ListedColormap(cmap_list))
        else:
            scatter = plt.scatter(
                embedding[:, 0], 
                embedding[:, 1], 
                c=labels_array, 
                cmap='viridis',
                alpha=0.7,
                s=50
            )
        
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Cluster Visualization ({model_name} with {method.upper()})')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        # Add timestamp annotations for a subset of points if available
        if 'time' in vis_df.columns:
            step = max(1, len(vis_df) // 20)  # Show up to 20 timestamp labels
            for i in range(0, len(vis_df), step):
                plt.annotate(
                    vis_df['time'].iloc[i].strftime('%H:%M:%S'),
                    (vis_df['x'].iloc[i], vis_df['y'].iloc[i]),
                    fontsize=8
                )
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'cluster_visualization_{model_name}_{method}.png')
        plt.close()
        
        return vis_df
    
    def get_best_model(self, metric='silhouette_score'):
        """
        Get the best model based on the specified metric
        
        Args:
            metric: Metric to use for ranking ('silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score')
            
        Returns:
            Name of the best model
        """
        if self.evaluation_results is None or self.evaluation_results.empty:
            raise ValueError("Models must be evaluated first")
        
        # For Davies-Bouldin, lower is better
        ascending = metric == 'davies_bouldin_score'
        
        sorted_results = self.evaluation_results.sort_values(metric, ascending=ascending)
        best_model = sorted_results.iloc[0]['model']
        
        print(f"Best model based on {metric}: {best_model}")
        return best_model
    
    def save_labels(self, model_name=None):
        """
        Save cluster labels to CSV file
        
        Args:
            model_name: Name of the model to save labels for (None for all models)
            
        Returns:
            Path to the saved file
        """
        if not self.labels:
            raise ValueError("Models must be run before saving labels")
        
        if model_name is None:
            # Save all labels
            labels_df = pd.DataFrame(self.labels)
            file_path = self.output_dir / 'all_cluster_labels.csv'
        else:
            # Save labels for the specified model
            labels = self.labels[model_name]
            labels_df = pd.DataFrame({model_name: labels})
            file_path = self.output_dir / f'cluster_labels_{model_name}.csv'
        
        # Set index if available
        if isinstance(self.normalized_features, pd.DataFrame):
            labels_df.index = self.normalized_features.index
        
        labels_df.to_csv(file_path)
        print(f"Labels saved to {file_path}")
        return file_path