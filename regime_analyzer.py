# backend/regime_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
import umap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import io
import base64
from datetime import datetime

class RegimeAnalyzer:
    """
    Analyze market regimes detected by clustering algorithms and provide interpretable insights
    """
    
    def __init__(self, features_df, labels_dict, output_dir='./results'):
        """
        Initialize the regime analyzer
        
        Args:
            features_df: DataFrame containing the extracted features
            labels_dict: Dictionary of model names and their cluster labels
            output_dir: Directory to save output files
        """
        self.features = features_df
        self.labels = labels_dict
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.regime_characteristics = {}
        self.regime_names = {}
        self.transition_matrices = {}
    
    def analyze_regimes(self, model_name=None):
        """
        Analyze the characteristics of each regime detected by the specified model
        
        Args:
            model_name: Name of the model to analyze (None for all models)
            
        Returns:
            Dictionary of regime characteristics
        """
        if model_name is not None:
            if model_name not in self.labels:
                raise ValueError(f"Model {model_name} not found in labels")
            models_to_analyze = [model_name]
        else:
            models_to_analyze = self.labels.keys()
        
        for model in models_to_analyze:
            print(f"Analyzing regimes for model: {model}")
            labels = self.labels[model]
            
            # Create a DataFrame with features and labels
            labeled_data = self.features.copy()
            labeled_data['regime'] = labels if not isinstance(labels, pd.Series) else labels.values
            
            # Analyze each regime
            unique_labels = np.unique(labeled_data['regime'])
            characteristics = {}
            regime_names = {}
            
            for label in unique_labels:
                if label == -1:  # Skip noise points in HDBSCAN
                    regime_names[label] = "Noise"
                    continue
                    
                regime_data = labeled_data[labeled_data['regime'] == label]
                stats = self._calculate_regime_statistics(regime_data, labeled_data)
                characteristics[label] = stats
                
                # Generate a regime name
                regime_name = self._generate_regime_name(stats)
                regime_names[label] = regime_name
                stats['name'] = regime_name
            
            self.regime_characteristics[model] = characteristics
            self.regime_names[model] = regime_names
            
            # Save regime characteristics to CSV
            self._save_characteristics_to_csv(model, characteristics)
        
        return self.regime_characteristics
    
    def _calculate_regime_statistics(self, regime_data, all_data):
        """
        Calculate statistics for a specific regime
        
        Args:
            regime_data: DataFrame containing data for a specific regime
            all_data: DataFrame containing all data
            
        Returns:
            Dictionary of statistics for the regime
        """
        stats = {}
        
        # Basic statistics
        stats['size'] = len(regime_data) / len(all_data)  # Proportion of data in this regime
        
        # Volatility characteristics
        volatility_cols = [col for col in regime_data.columns if 'volatility' in col or 'std' in col]
        if volatility_cols:
            mean_volatility = regime_data[volatility_cols].mean().mean()
            all_volatility = all_data[volatility_cols].mean().mean()
            stats['volatility'] = mean_volatility
            stats['volatility_ratio'] = mean_volatility / all_volatility if all_volatility > 0 else 1.0
            stats['volatility_type'] = 'High' if stats['volatility_ratio'] > 1.1 else 'Low' if stats['volatility_ratio'] < 0.9 else 'Medium'
        
        # Trend/Mean-reversion characteristics
        returns_cols = [col for col in regime_data.columns if 'returns' in col or 'momentum' in col]
        if returns_cols:
            mean_returns = regime_data[returns_cols].mean().mean()
            stats['returns'] = mean_returns
            
            # Calculate auto-correlation for mean-reversion vs trending behavior
            if len(regime_data) > 10 and 'returns' in regime_data.columns:
                autocorr = regime_data['returns'].autocorr(lag=1)
                stats['autocorrelation'] = autocorr
                stats['price_behavior'] = 'Mean-reverting' if autocorr < -0.2 else 'Trending' if autocorr > 0.2 else 'Random'
            else:
                stats['price_behavior'] = 'Unknown'
        
        # Liquidity characteristics
        spread_cols = [col for col in regime_data.columns if 'spread' in col]
        depth_cols = [col for col in regime_data.columns if 'depth' in col or 'qty' in col]
        
        if spread_cols:
            mean_spread = regime_data[spread_cols].mean().mean() 
            all_spread = all_data[spread_cols].mean().mean()
            stats['spread'] = mean_spread
            stats['spread_ratio'] = mean_spread / all_spread if all_spread > 0 else 1.0
        
        if depth_cols:
            mean_depth = regime_data[depth_cols].mean().mean()
            all_depth = all_data[depth_cols].mean().mean()
            stats['depth'] = mean_depth
            stats['depth_ratio'] = mean_depth / all_depth if all_depth > 0 else 1.0
            
        # Combined liquidity measure
        if 'spread_ratio' in stats and 'depth_ratio' in stats:
            liquidity_score = (1/stats['spread_ratio']) * stats['depth_ratio']
            stats['liquidity_score'] = liquidity_score
            stats['liquidity'] = 'High' if liquidity_score > 1.1 else 'Low' if liquidity_score < 0.9 else 'Medium'
        
        # Volume characteristics
        volume_cols = [col for col in regime_data.columns if 'volume' in col and 'imbalance' not in col]
        if volume_cols:
            mean_volume = regime_data[volume_cols].mean().mean()
            all_volume = all_data[volume_cols].mean().mean()
            stats['volume'] = mean_volume
            stats['volume_ratio'] = mean_volume / all_volume if all_volume > 0 else 1.0
            stats['volume_type'] = 'High' if stats['volume_ratio'] > 1.1 else 'Low' if stats['volume_ratio'] < 0.9 else 'Medium'
        
        # Direction characteristics
        direction_cols = [col for col in regime_data.columns if 'direction' in col or 'momentum' in col]
        if direction_cols:
            mean_direction = regime_data[direction_cols].mean().mean()
            stats['direction'] = mean_direction
            stats['direction_type'] = 'Upward' if mean_direction > 0.001 else 'Downward' if mean_direction < -0.001 else 'Sideways'
        
        # Imbalance characteristics
        imbalance_cols = [col for col in regime_data.columns if 'imbalance' in col]
        if imbalance_cols:
            mean_imbalance = regime_data[imbalance_cols].mean().mean()
            stats['imbalance'] = mean_imbalance
            stats['imbalance_type'] = 'Buy Heavy' if mean_imbalance > 0.1 else 'Sell Heavy' if mean_imbalance < -0.1 else 'Balanced'
            
        # Calculate average feature values for this regime
        feature_cols = [col for col in regime_data.columns if col != 'regime' and not col.startswith('_')]
        stats['avg_features'] = {col: regime_data[col].mean() for col in feature_cols}
        
        # Calculate standard deviations for key features to measure regime coherence
        stats['std_features'] = {col: regime_data[col].std() for col in feature_cols}
        
        return stats
    
    def _generate_regime_name(self, stats):
        """
        Generate a descriptive name for a regime based on its statistics
        
        Args:
            stats: Dictionary of statistics for the regime
            
        Returns:
            String name for the regime
        """
        parts = []
        
        # Add price behavior first 
        if 'price_behavior' in stats:
            parts.append(stats['price_behavior'])
        elif 'direction_type' in stats:
            parts.append(stats['direction_type'])
            
        # Add volatility type
        if 'volatility_type' in stats:
            parts.append(f"{stats['volatility_type']} Volatility")
            
        # Add liquidity type
        if 'liquidity' in stats:
            parts.append(f"{stats['liquidity']} Liquidity")
        
        # Add volume type if we have it and it's interesting
        if 'volume_type' in stats and stats['volume_type'] != 'Medium':
            parts.append(f"{stats['volume_type']} Volume")
            
        # Add imbalance type if significant
        if 'imbalance_type' in stats and stats['imbalance_type'] != 'Balanced':
            parts.append(stats['imbalance_type'])
        
        # Combine the parts
        if parts:
            name = " & ".join(parts)
        else:
            name = "Regime"  # Default if we can't characterize the regime
        
        return name
    
    def _save_characteristics_to_csv(self, model_name, characteristics):
        """
        Save regime characteristics to a CSV file
        
        Args:
            model_name: Name of the model
            characteristics: Dictionary of regime characteristics
        """
        # Convert nested dictionary to DataFrame
        rows = []
        for regime_label, stats in characteristics.items():
            row = {'regime': regime_label, 'name': stats.get('name', f'Regime {regime_label}')}
            
            # Add simple stats directly
            for key, value in stats.items():
                if key != 'avg_features' and key != 'std_features' and not isinstance(value, dict):
                    row[key] = value
            
            # Add selected avg_features with prefix
            if 'avg_features' in stats:
                for feat, val in stats['avg_features'].items():
                    if any(keyword in feat for keyword in ['volatility', 'spread', 'depth', 'volume', 'direction', 'imbalance']):
                        row[f'avg_{feat}'] = val
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        file_path = self.output_dir / f'regime_characteristics_{model_name}.csv'
        df.to_csv(file_path, index=False)
        print(f"Saved regime characteristics to {file_path}")
    
    def analyze_transitions(self, model_name=None):
        """
        Analyze transitions between different regimes
        
        Args:
            model_name: Name of the model to analyze (None for the first model)
            
        Returns:
            DataFrame with transition probabilities
        """
        if model_name is None:
            model_name = next(iter(self.labels.keys()))
        
        if model_name not in self.labels:
            raise ValueError(f"Model {model_name} not found in labels")
        
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
        unique_regimes = sorted(np.unique(labels))
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
        
        # Get regime names if available
        if model_name in self.regime_names:
            regime_labels = [f"{unique_regimes[i]}: {self.regime_names[model_name].get(unique_regimes[i], f'Regime {unique_regimes[i]}')}" 
                           for i in range(n_regimes)]
        else:
            regime_labels = [f"Regime {r}" for r in unique_regimes]
        
        # Create a DataFrame for better readability
        transition_df = pd.DataFrame(
            transition_probs,
            index=regime_labels,
            columns=regime_labels
        )
        
        self.transition_matrices[model_name] = transition_df
        
        # Save transition matrix
        file_path = self.output_dir / f'transition_matrix_{model_name}.csv'
        transition_df.to_csv(file_path)
        print(f"Saved transition matrix to {file_path}")
        
        return transition_df
    
    def visualize_transitions(self, model_name=None):
        """
        Visualize the transition matrix as a heatmap
        
        Args:
            model_name: Name of the model to visualize (None for the first model)
            
        Returns:
            Path to the saved visualization
        """
        if model_name is None:
            model_name = next(iter(self.labels.keys()))
        
        if model_name not in self.transition_matrices:
            self.analyze_transitions(model_name)
        
        transition_matrix = self.transition_matrices[model_name]
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            transition_matrix, 
            annot=True, 
            cmap="YlGnBu",
            vmin=0, 
            vmax=1,
            fmt=".2f"
        )
        plt.title(f'Regime Transition Probabilities ({model_name})')
        plt.xlabel('To Regime')
        plt.ylabel('From Regime')
        plt.tight_layout()
        
        file_path = self.output_dir / f'transition_heatmap_{model_name}.png'
        plt.savefig(file_path)
        plt.close()
        
        print(f"Saved transition heatmap to {file_path}")
        return file_path
    
    def visualize_regime_evolution(self, model_name=None, overlay_price=True):
        """
        Visualize how regimes evolve over time
        
        Args:
            model_name: Name of the model to visualize (None for the first model)
            overlay_price: Whether to overlay price on the regime plot
            
        Returns:
            Path to the saved visualization
        """
        if model_name is None:
            model_name = next(iter(self.labels.keys()))
        
        if model_name not in self.labels:
            raise ValueError(f"Model {model_name} not found in labels")
        
        labels = self.labels[model_name]
        
        # Create a time series of regime labels
        regime_series = pd.Series(labels, index=self.features.index)
        
        # Get the mid price or price if available for overlay
        if 'mid_price' in self.features.columns:
            price_series = self.features['mid_price']
        elif 'price' in self.features.columns:
            price_series = self.features['price']
        else:
            price_series = None
            overlay_price = False
        
        # Plot the regime evolution
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Use regime names if available
        if model_name in self.regime_names:
            unique_regimes = sorted(np.unique(labels))
            cmap = plt.cm.get_cmap('viridis', len(unique_regimes))
            
            # Create custom labels for legend
            legend_elements = []
            for i, r in enumerate(unique_regimes):
                if r in self.regime_names[model_name]:
                    regime_name = self.regime_names[model_name][r]
                    color = cmap(i / len(unique_regimes))
                    legend_elements.append(
                        mpatches.Patch(color=color, label=f"{r}: {regime_name}")
                    )
            
            # Create a custom colormap if needed
            if -1 in unique_regimes:  # Handle HDBSCAN noise
                colors = [cmap(i / (len(unique_regimes) - 1)) for i in range(len(unique_regimes))]
                colors[unique_regimes.index(-1)] = (0.5, 0.5, 0.5, 1.0)  # Gray for noise
                custom_cmap = mcolors.ListedColormap(colors)
                scatter = ax1.scatter(regime_series.index, regime_series, c=regime_series, 
                                     cmap=custom_cmap, s=30, alpha=0.8)
            else:
                scatter = ax1.scatter(regime_series.index, regime_series, c=regime_series, 
                                     cmap='viridis', s=30, alpha=0.8)
            
            # Add legend
            ax1.legend(handles=legend_elements, title="Regimes", loc='upper left', 
                      bbox_to_anchor=(1.01, 1), borderaxespad=0)
        else:
            scatter = ax1.scatter(regime_series.index, regime_series, c=regime_series, 
                                 cmap='viridis', s=30, alpha=0.8)
            plt.colorbar(scatter, ax=ax1, label='Regime')
        
        ax1.set_ylabel('Regime')
        ax1.set_title(f'Market Regime Evolution Over Time ({model_name})')
        
        # Add price if available and requested
        if overlay_price and price_series is not None:
            ax2 = ax1.twinx()
            ax2.plot(price_series.index, price_series, color='red', alpha=0.5, label='Price')
            ax2.set_ylabel('Price', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        # Format x-axis to show time properly
        fig.autofmt_xdate()
        plt.tight_layout()
        
        file_path = self.output_dir / f'regime_evolution_{model_name}.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved regime evolution plot to {file_path}")
        return file_path
    
    def visualize_regime_distribution(self, model_name=None, method='umap'):
        """
        Visualize the distribution of regimes in feature space using dimensionality reduction
        
        Args:
            model_name: Name of the model to visualize (None for the first model)
            method: Dimensionality reduction method ('umap', 'tsne', or 'pca')
            
        Returns:
            Path to the saved visualization
        """
        if model_name is None:
            model_name = next(iter(self.labels.keys()))
        
        if model_name not in self.labels:
            raise ValueError(f"Model {model_name} not found in labels")
        
        labels = self.labels[model_name]
        
        # Apply dimensionality reduction
        if method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(self.features)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embedding = reducer.fit_transform(self.features)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(self.features)
        else:
            raise ValueError(f"Unknown visualization method: {method}")
        
        # Create a DataFrame for visualization
        vis_df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'regime': labels
        })
        
        # Get timestamps if available
        if isinstance(self.features, pd.DataFrame) and isinstance(self.features.index, pd.DatetimeIndex):
            vis_df['time'] = self.features.index
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Use regime names if available
        if model_name in self.regime_names:
            unique_regimes = sorted(np.unique(labels))
            cmap = plt.cm.get_cmap('viridis', len(unique_regimes))
            
            # Create custom labels for legend
            legend_elements = []
            for i, r in enumerate(unique_regimes):
                if r in self.regime_names[model_name]:
                    regime_name = self.regime_names[model_name][r]
                    color = cmap(i / len(unique_regimes))
                    legend_elements.append(
                        mpatches.Patch(color=color, label=f"{r}: {regime_name}")
                    )
            
            # Create a custom colormap if needed
            if -1 in unique_regimes:  # Handle HDBSCAN noise
                colors = [cmap(i / (len(unique_regimes) - 1)) for i in range(len(unique_regimes))]
                colors[unique_regimes.index(-1)] = (0.5, 0.5, 0.5, 1.0)  # Gray for noise
                custom_cmap = mcolors.ListedColormap(colors)
                scatter = plt.scatter(vis_df['x'], vis_df['y'], c=vis_df['regime'], 
                                     cmap=custom_cmap, s=50, alpha=0.7)
            else:
                scatter = plt.scatter(vis_df['x'], vis_df['y'], c=vis_df['regime'], 
                                     cmap='viridis', s=50, alpha=0.7)
                
            # Add legend
            plt.legend(handles=legend_elements, title="Regimes", loc='upper left', 
                      bbox_to_anchor=(1.01, 1), borderaxespad=0)
        else:
            scatter = plt.scatter(vis_df['x'], vis_df['y'], c=vis_df['regime'], 
                                 cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(scatter, label='Regime')
        
        plt.title(f'Market Regimes in Feature Space ({model_name} with {method.upper()})')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        # Add timestamp annotations for a subset of points
        if 'time' in vis_df.columns:
            step = max(1, len(vis_df) // 20)  # Show up to 20 time labels
            for i in range(0, len(vis_df), step):
                plt.annotate(
                    vis_df['time'].iloc[i].strftime('%H:%M:%S'),
                    (vis_df['x'].iloc[i], vis_df['y'].iloc[i]),
                    fontsize=8
                )
        
        plt.tight_layout()
        file_path = self.output_dir / f'regime_distribution_{model_name}_{method}.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved regime distribution plot to {file_path}")
        return file_path
    
    def visualize_feature_distribution(self, feature_name, model_name=None):
        """
        Visualize the distribution of a feature across different regimes
        
        Args:
            feature_name: Name of the feature to visualize
            model_name: Name of the model to use for regimes (None for the first model)
            
        Returns:
            Path to the saved visualization
        """
        if feature_name not in self.features.columns:
            raise ValueError(f"Feature {feature_name} not found in features")
        
        if model_name is None:
            model_name = next(iter(self.labels.keys()))
        
        if model_name not in self.labels:
            raise ValueError(f"Model {model_name} not found in labels")
        
        labels = self.labels[model_name]
        
        # Create a DataFrame with the feature and labels
        data = pd.DataFrame({
            'feature': self.features[feature_name],
            'regime': labels
        })
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        # Use regime names if available
        if model_name in self.regime_names:
            regime_mapping = {r: f"{r}: {name}" for r, name in self.regime_names[model_name].items()}
            data['regime_name'] = data['regime'].map(regime_mapping)
            sns.violinplot(x='regime_name', y='feature', data=data)
            plt.xticks(rotation=45, ha='right')
        else:
            sns.violinplot(x='regime', y='feature', data=data)
        
        plt.title(f'Distribution of {feature_name} Across Regimes ({model_name})')
        plt.xlabel('Regime')
        plt.ylabel(feature_name)
        plt.tight_layout()
        
        file_path = self.output_dir / f'feature_distribution_{feature_name}_{model_name}.png'
        plt.savefig(file_path)
        plt.close()
        
        print(f"Saved feature distribution plot to {file_path}")
        return file_path
    
    def visualize_key_features(self, model_name=None, top_n=5):
        """
        Visualize the distribution of the top N most distinctive features across regimes
        
        Args:
            model_name: Name of the model to use for regimes (None for the first model)
            top_n: Number of top features to visualize
            
        Returns:
            List of paths to the saved visualizations
        """
        if model_name is None:
            model_name = next(iter(self.labels.keys()))
        
        if model_name not in self.labels:
            raise ValueError(f"Model {model_name} not found in labels")
            
        if model_name not in self.regime_characteristics:
            self.analyze_regimes(model_name)
        
        # Find the most distinctive features
        distinctive_features = self._find_distinctive_features(model_name, top_n)
        
        # Visualize each feature
        paths = []
        for feature in distinctive_features:
            path = self.visualize_feature_distribution(feature, model_name)
            paths.append(path)
        
        return paths
    
    def _find_distinctive_features(self, model_name, top_n=5):
        """
        Find the top N features that best distinguish between regimes
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            List of feature names
        """
        # Calculate feature distinctiveness using F-statistic (ANOVA)
        from scipy import stats
        
        labels = self.labels[model_name]
        
        # Create groups for each feature
        feature_f_values = {}
        for col in self.features.columns:
            groups = [self.features[col][labels == label] for label in np.unique(labels) if label != -1]
            if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                try:
                    f_val, p_val = stats.f_oneway(*groups)
                    feature_f_values[col] = f_val
                except:
                    pass
        
        # Sort features by F-value (higher means more distinctive)
        sorted_features = sorted(feature_f_values.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N feature names
        return [f[0] for f in sorted_features[:top_n]]
    
    def generate_summary_report(self, model_name=None):
        """
        Generate a summary report of the regime analysis
        
        Args:
            model_name: Name of the model to analyze (None for the first model)
            
        Returns:
            Dictionary with report data
        """
        if model_name is None:
            model_name = next(iter(self.labels.keys()))
        
        if model_name not in self.regime_characteristics:
            self.analyze_regimes(model_name)
            
        if model_name not in self.transition_matrices:
            self.analyze_transitions(model_name)
        
        # Get regime characteristics
        characteristics = self.regime_characteristics[model_name]
        
        # Calculate regime statistics
        labels = self.labels[model_name]
        unique_labels = np.unique(labels)
        regime_counts = {int(label): int(np.sum(labels == label)) for label in unique_labels}
        total_count = len(labels)
        regime_percentages = {int(label): float(count / total_count * 100) for label, count in regime_counts.items()}
        
        # Identify dominant regimes
        dominant_regimes = sorted(regime_percentages.items(), key=lambda x: x[1], reverse=True)[:3]
        # Convert NumPy types to native Python types
        dominant_regimes = [(int(label), float(pct)) for label, pct in dominant_regimes]
        
        # Identify most likely transitions
        transition_matrix = self.transition_matrices[model_name]
        highest_transitions = []
        for i in range(len(transition_matrix)):
            row = transition_matrix.iloc[i]
            if row.max() > 0.3:  # Only include significant transitions
                from_regime = transition_matrix.index[i]
                to_regime = row.idxmax()
                highest_transitions.append((from_regime, to_regime, float(row.max())))
        
        highest_transitions.sort(key=lambda x: x[2], reverse=True)
        highest_transitions = highest_transitions[:5]  # Top 5 transitions
        
        # Generate report data
        report = {
            'model_name': model_name,
            'total_samples': int(total_count),
            'num_regimes': int(len(unique_labels) - (1 if -1 in unique_labels else 0)),
            'dominant_regimes': dominant_regimes,
            'highest_transitions': highest_transitions
        }
        
        # Convert regime characteristics to serializable format
        serializable_chars = {}
        for label, stats in characteristics.items():
            # Convert NumPy types to Python native types
            serializable_stats = {}
            for k, v in stats.items():
                if k == 'avg_features' or k == 'std_features':
                    # Handle nested dictionaries
                    serializable_stats[k] = {
                        feat: float(val) if isinstance(val, (np.number, np.ndarray)) else val
                        for feat, val in v.items()
                    }
                elif isinstance(v, np.ndarray):
                    serializable_stats[k] = v.tolist()
                elif isinstance(v, np.number):
                    serializable_stats[k] = float(v) if isinstance(v, np.floating) else int(v)
                else:
                    serializable_stats[k] = v
            serializable_chars[int(label) if isinstance(label, np.integer) else label] = serializable_stats
        
        report['regime_characteristics'] = serializable_chars
        
        # Save report as JSON
        import json
        file_path = self.output_dir / f'regime_report_{model_name}.json'
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
            
        print(f"Saved summary report to {file_path}")
        
        return report