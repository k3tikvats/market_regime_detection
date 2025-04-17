# run_regime_detection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from pathlib import Path
import argparse
import logging

# Import project components
from backend.data_pipeline.feature_engineering.normalizer import Normalizer
from backend.clustering_executor import ClusteringExecutor
from backend.regime_analyzer import RegimeAnalyzer
from backend.ml_service.regime_detector import MarketRegimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('market_regime_detection')

def generate_synthetic_features(n_samples=1000, n_features=20):
    """
    Generate synthetic market data features for testing
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        
    Returns:
        DataFrame with synthetic features
    """
    logger.info(f"Generating {n_samples} synthetic samples with {n_features} features")
    
    # Create a datetime index
    start_date = datetime.now() - timedelta(days=5)
    minutes = pd.date_range(start=start_date, periods=n_samples, freq='1min')
    
    # Generate base price series with trends and reversals
    np.random.seed(42)
    
    # Create different market regimes
    regime_lengths = [100, 200, 150, 200, 350]
    regimes = []
    
    for length in regime_lengths:
        if len(regimes) >= n_samples:
            break
        regimes.extend([len(regimes) // 100] * min(length, n_samples - len(regimes)))
    
    regimes = regimes[:n_samples]
    
    # Generate price series with different behavior in each regime
    price = 100.0
    prices = []
    returns = []
    volatilities = []
    
    for i in range(n_samples):
        regime = regimes[i]
        
        # Different volatility per regime
        if regime == 0:  # Low volatility, slight uptrend
            volatility = 0.0005
            drift = 0.0002
        elif regime == 1:  # High volatility, strong uptrend
            volatility = 0.002
            drift = 0.001
        elif regime == 2:  # Medium volatility, mean-reverting
            volatility = 0.001
            drift = -0.0003 if price > 105 else 0.0003
        elif regime == 3:  # High volatility, downtrend
            volatility = 0.0025
            drift = -0.001
        else:  # Low volatility, sideways
            volatility = 0.0004
            drift = 0.0
        
        # Generate return
        ret = np.random.normal(drift, volatility)
        price *= (1 + ret)
        
        prices.append(price)
        returns.append(ret)
        volatilities.append(volatility)
    
    # Create base features DataFrame
    features = {
        'price': prices,
        'returns': returns,
        'volatility': volatilities,
        'spread': [0.01 + 0.03 * (0.8 if r == 1 or r == 3 else 0.3) + 0.01 * np.random.rand() for r in regimes],
        'depth': [1000 * (0.5 if r == 1 or r == 3 else 1.5) + 200 * np.random.rand() for r in regimes],
        'volume': [100 * (2.0 if r == 1 or r == 3 else 0.8) + 50 * np.random.rand() for r in regimes],
        'volume_imbalance': [0.2 * np.random.randn() + (0.3 if r == 1 else -0.3 if r == 3 else 0.0) for r in regimes],
        'mid_price': prices.copy(),
    }
    
    # Add bid/ask prices and quantities
    features['bid_price_0'] = [p - s/2 for p, s in zip(prices, features['spread'])]
    features['ask_price_0'] = [p + s/2 for p, s in zip(prices, features['spread'])]
    features['bid_qty_0'] = [d * (0.8 + 0.4 * np.random.rand()) for d in features['depth']]
    features['ask_qty_0'] = [d * (0.8 + 0.4 * np.random.rand()) for d in features['depth']]
    
    # Add more synthetic features
    features['price_zscore_10'] = np.random.randn(n_samples)  # Synthetic Z-score
    features['price_zscore_30'] = np.random.randn(n_samples)  # Synthetic Z-score
    features['volatility_10'] = [v * 10 * (0.8 + 0.4 * np.random.rand()) for v in volatilities]
    features['volatility_30'] = [v * 30 * (0.8 + 0.4 * np.random.rand()) for v in volatilities]
    features['volatility_60'] = [v * 60 * (0.8 + 0.4 * np.random.rand()) for v in volatilities]
    
    # Add momentum features - ensuring they have the correct length
    features['momentum_10'] = features['returns'].copy()  # Placeholder with correct length
    for i in range(n_samples):
        start_idx = max(0, i-10)
        features['momentum_10'][i] = sum(features['returns'][start_idx:i+1]) / min(10, i+1)
        
    features['momentum_30'] = features['returns'].copy()  # Placeholder with correct length
    for i in range(n_samples):
        start_idx = max(0, i-30)
        features['momentum_30'][i] = sum(features['returns'][start_idx:i+1]) / min(30, i+1)
    
    # Add liquidity features
    features['cum_bid_qty'] = [qty * 5 for qty in features['bid_qty_0']]  # Synthetic cumulative depth
    features['cum_ask_qty'] = [qty * 5 for qty in features['ask_qty_0']]  # Synthetic cumulative depth
    
    # Create DataFrame with datetime index
    df = pd.DataFrame(features, index=minutes)
    
    # Verify all columns have the same length
    for col in df.columns:
        if len(df[col]) != n_samples:
            logger.error(f"Column {col} has incorrect length: {len(df[col])} vs expected {n_samples}")
    
    # Add noise to make features more realistic
    for col in df.columns:
        if col != 'price' and col != 'mid_price':
            noise_factor = 0.05
            df[col] = df[col] * (1 + noise_factor * np.random.randn(n_samples))
    
    logger.info(f"Generated synthetic data with shape {df.shape}")
    return df

def run_regime_detection(features_df=None, output_dir="./results"):
    """
    Run the complete market regime detection pipeline
    
    Args:
        features_df: DataFrame containing features (or None to generate synthetic data)
        output_dir: Directory to save results
        
    Returns:
        Dictionary with results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate synthetic data if needed
    if features_df is None:
        logger.info("No feature data provided, generating synthetic data")
        features_df = generate_synthetic_features(n_samples=1000, n_features=20)
        
        # Save generated features
        features_path = output_path / "synthetic_features.csv"
        features_df.to_csv(features_path)
        logger.info(f"Saved synthetic features to {features_path}")
    
    # 1. Normalize features
    logger.info("Normalizing features")
    normalizer = Normalizer(method="standard")
    normalized_features = normalizer.fit_transform(features_df)
    
    # 2. Run clustering
    logger.info("Running clustering models")
    cluster_executor = ClusteringExecutor(features_df)
    
    # Apply additional normalization if needed (e.g., PCA)
    normalized_features = cluster_executor.normalize_features(
        method="standard", 
        pca_components=None  # Set to a number to use PCA
    )
    
    # Setup and run clustering models
    cluster_executor.setup_models()
    labels_dict = cluster_executor.run_clustering()
    
    # Evaluate models
    evaluation_results = cluster_executor.evaluate_models()
    best_model = cluster_executor.get_best_model(metric='silhouette_score')
    logger.info(f"Best clustering model: {best_model}")
    
    # Visualize clusters
    logger.info("Creating cluster visualizations")
    cluster_executor.visualize_clusters(method='umap', model_name=best_model)
    cluster_executor.visualize_clusters(method='tsne', model_name=best_model)
    
    # 3. Analyze regimes
    logger.info("Analyzing market regimes")
    analyzer = RegimeAnalyzer(features_df, labels_dict, output_dir=output_dir)
    regime_characteristics = analyzer.analyze_regimes()
    
    # Analyze regime transitions
    logger.info("Analyzing regime transitions")
    transition_matrix = analyzer.analyze_transitions(model_name=best_model)
    analyzer.visualize_transitions(model_name=best_model)
    
    # Visualize regime evolution
    logger.info("Visualizing regime evolution")
    analyzer.visualize_regime_evolution(model_name=best_model, overlay_price=True)
    analyzer.visualize_regime_distribution(model_name=best_model, method='umap')
    
    # Visualize key feature distributions
    logger.info("Visualizing feature distributions")
    feature_plot_paths = analyzer.visualize_key_features(model_name=best_model, top_n=5)
    
    # 4. Generate comprehensive report
    logger.info("Generating analysis report")
    summary_report = analyzer.generate_summary_report(model_name=best_model)
    
    # 5. Create readable markdown report
    logger.info("Creating markdown report")
    report_path = generate_markdown_report(
        features_df, 
        labels_dict,
        evaluation_results, 
        regime_characteristics, 
        transition_matrix,
        best_model,
        output_dir
    )
    
    return {
        "features": features_df,
        "normalized_features": normalized_features,
        "labels": labels_dict,
        "best_model": best_model,
        "evaluation_results": evaluation_results,
        "regime_characteristics": regime_characteristics,
        "transition_matrix": transition_matrix,
        "report_path": report_path
    }

def generate_markdown_report(features, labels, evaluation_results, regime_characteristics, 
                           transition_matrix, best_model, output_dir):
    """
    Generate a comprehensive markdown report
    
    Args:
        features: DataFrame with features
        labels: Dictionary of model labels
        evaluation_results: DataFrame with model evaluation results
        regime_characteristics: Dictionary of regime characteristics
        transition_matrix: DataFrame with transition matrix
        best_model: Name of the best model
        output_dir: Directory to save the report
        
    Returns:
        Path to the saved report
    """
    output_path = Path(output_dir)
    report_path = output_path / "market_regime_analysis.md"
    
    with open(report_path, "w") as f:
        # Title
        f.write("# Market Regime Detection Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report presents the results of unsupervised market regime detection using ")
        f.write(f"clustering algorithms on market data. The analysis identified ")
        
        # Count regimes in the best model (excluding noise if present)
        best_labels = labels[best_model]
        unique_labels = np.unique(best_labels)
        n_regimes = len(unique_labels)
        if -1 in unique_labels:  # HDBSCAN noise
            n_regimes -= 1
            
        f.write(f"**{n_regimes} distinct market regimes** with ")
        f.write("clear differences in volatility, trend behavior, and liquidity characteristics.\n\n")
        
        # Add information about the dominant regime
        if best_model in regime_characteristics:
            chars = regime_characteristics[best_model]
            
            # Find the most common regime
            regime_counts = {label: np.sum(best_labels == label) for label in unique_labels if label != -1}
            dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
            dominant_pct = regime_counts[dominant_regime] / len(best_labels) * 100
            
            if dominant_regime in chars:
                dominant_name = chars[dominant_regime].get('name', f'Regime {dominant_regime}')
                f.write(f"The dominant regime is **{dominant_name}**, occurring in ")
                f.write(f"**{dominant_pct:.1f}%** of the analyzed time period.\n\n")
                
                # Add some characteristics of the dominant regime
                dom_chars = chars[dominant_regime]
                f.write("Key characteristics of the dominant regime:\n")
                
                if 'volatility_type' in dom_chars:
                    f.write(f"- **{dom_chars['volatility_type']} volatility**\n")
                if 'price_behavior' in dom_chars:
                    f.write(f"- **{dom_chars['price_behavior']} price behavior**\n")
                if 'liquidity' in dom_chars:
                    f.write(f"- **{dom_chars['liquidity']} liquidity**\n")
                if 'volume_type' in dom_chars:
                    f.write(f"- **{dom_chars['volume_type']} volume**\n")
                
                f.write("\n")
        
        # 1. Data Overview
        f.write("## 1. Data Overview\n\n")
        
        f.write(f"- **Number of samples**: {len(features)}\n")
        f.write(f"- **Number of features**: {len(features.columns)}\n")
        f.write(f"- **Time period**: {features.index[0]} to {features.index[-1]}\n\n")
        
        # Feature categories
        f.write("### Feature Categories\n\n")
        feature_groups = {
            "Price": [col for col in features.columns if any(x in col.lower() for x in ['price', 'return', 'momentum'])],
            "Volatility": [col for col in features.columns if 'volatility' in col.lower() or 'std' in col.lower()],
            "Liquidity": [col for col in features.columns if any(x in col.lower() for x in ['spread', 'depth', 'qty'])],
            "Volume": [col for col in features.columns if 'volume' in col.lower()]
        }
        
        for group, cols in feature_groups.items():
            if cols:
                f.write(f"**{group} Features** ({len(cols)}):\n")
                for col in sorted(cols)[:5]:  # Show first 5 features
                    f.write(f"- `{col}`\n")
                if len(cols) > 5:
                    f.write(f"- ... and {len(cols) - 5} more\n")
                f.write("\n")
        
        # 2. Clustering Model Evaluation
        f.write("## 2. Clustering Model Evaluation\n\n")
        
        f.write(f"The best performing clustering model is **{best_model}** ")
        if not evaluation_results.empty:
            best_score = evaluation_results.iloc[0]['silhouette_score']
            f.write(f"with a silhouette score of **{best_score:.4f}**.\n\n")
        else:
            f.write(".\n\n")
            
        # Show top models table
        f.write("### Model Performance Comparison\n\n")
        f.write("| Model | # Clusters | Silhouette Score ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |\n")
        f.write("|-------|------------|-------------------|-----------------|---------------------|\n")
        
        top_models = min(5, len(evaluation_results))
        for i in range(top_models):
            row = evaluation_results.iloc[i]
            f.write(f"| {row['model']} | {row['n_clusters']} | {row['silhouette_score']:.4f} | ")
            f.write(f"{row['davies_bouldin_score']:.4f} | {row['calinski_harabasz_score']:.4f} |\n")
        
        f.write("\n")
        
        # 3. Regime Characteristics
        f.write("## 3. Regime Characteristics\n\n")
        
        if best_model in regime_characteristics:
            chars = regime_characteristics[best_model]
            
            f.write("### Regime Summary\n\n")
            f.write("| Regime | Name | Size | Volatility | Price Behavior | Liquidity | Volume |\n")
            f.write("|--------|------|------|------------|----------------|-----------|--------|\n")
            
            for label in sorted(chars.keys()):
                stats = chars[label]
                regime_name = stats.get('name', f'Regime {label}')
                size = stats.get('size', 'N/A')
                size_str = f"{size:.1%}" if isinstance(size, (int, float)) else size
                volatility = stats.get('volatility_type', 'N/A')
                price_behavior = stats.get('price_behavior', 'N/A')
                liquidity = stats.get('liquidity', 'N/A')
                volume = stats.get('volume_type', 'N/A')
                
                f.write(f"| {label} | {regime_name} | {size_str} | {volatility} | ")
                f.write(f"{price_behavior} | {liquidity} | {volume} |\n")
            
            f.write("\n")
            
            # Detailed regime descriptions
            f.write("### Detailed Regime Descriptions\n\n")
            
            for label in sorted(chars.keys()):
                stats = chars[label]
                regime_name = stats.get('name', f'Regime {label}')
                
                f.write(f"#### Regime {label}: {regime_name}\n\n")
                
                size = stats.get('size', 'N/A')
                size_str = f"{size:.1%}" if isinstance(size, (int, float)) else size
                f.write(f"- **Size**: {size_str} of the analyzed period\n")
                
                # Add specific characteristics
                for key in ['volatility', 'autocorrelation', 'spread', 'depth', 'volume']:
                    if key in stats and not isinstance(stats[key], dict):
                        f.write(f"- **{key.capitalize()}**: {stats[key]:.4f}\n")
                
                # Add descriptive characteristics
                for key in ['price_behavior', 'volatility_type', 'liquidity', 'direction_type', 'volume_type']:
                    if key in stats:
                        f.write(f"- **{key.replace('_', ' ').title()}**: {stats[key]}\n")
                
                f.write("\n")
                
                # If available, add average feature values
                if 'avg_features' in stats and stats['avg_features']:
                    f.write("Selected average feature values:\n")
                    
                    # Select a subset of interesting features
                    interesting_features = {}
                    for feat, val in stats['avg_features'].items():
                        if any(keyword in feat for keyword in ['volatility', 'spread', 'volume', 'momentum', 'zscore']):
                            interesting_features[feat] = val
                    
                    # Display top 5 interesting features
                    for i, (feat, val) in enumerate(list(interesting_features.items())[:5]):
                        f.write(f"- `{feat}`: {val:.4f}\n")
                    
                    f.write("\n")
        
        # 4. Regime Transitions
        f.write("## 4. Regime Transitions\n\n")
        
        if isinstance(transition_matrix, pd.DataFrame) and not transition_matrix.empty:
            f.write("### Transition Probability Matrix\n\n")
            
            # Only display a compact matrix if it's large
            if len(transition_matrix) > 5:
                f.write("*Note: Showing major transitions only.*\n\n")
                
                # Find significant transitions
                significant = []
                for i in range(len(transition_matrix)):
                    for j in range(len(transition_matrix.columns)):
                        if i != j and transition_matrix.iloc[i, j] >= 0.2:
                            significant.append({
                                'from': transition_matrix.index[i],
                                'to': transition_matrix.columns[j],
                                'probability': transition_matrix.iloc[i, j]
                            })
                
                if significant:
                    significant.sort(key=lambda x: x['probability'], reverse=True)
                    
                    f.write("| From | To | Probability |\n")
                    f.write("|------|----|--------------|\n")
                    
                    for s in significant[:10]:  # Show top 10
                        f.write(f"| {s['from']} | {s['to']} | {s['probability']:.2f} |\n")
                    
                    f.write("\n")
                else:
                    f.write("*No significant transitions found.*\n\n")
            else:
                # Display full matrix for small number of regimes
                f.write("| From / To | " + " | ".join(transition_matrix.columns) + " |\n")
                f.write("|" + "|".join(["---"] * (len(transition_matrix.columns) + 1)) + "|\n")
                
                for idx, row in transition_matrix.iterrows():
                    f.write(f"| {idx} | " + " | ".join([f"{val:.2f}" for val in row.values]) + " |\n")
                
                f.write("\n")
        
            # Add Markov analysis insight
            f.write("### Markov Chain Analysis\n\n")
            f.write("The market regime transitions exhibit Markovian properties, where the ")
            f.write("probability of transitioning to a new regime depends primarily on the ")
            f.write("current regime. Based on the transition matrix, we can observe:\n\n")
            
            # Find self-transitions (diagonal)
            diag_vals = np.diag(transition_matrix.values)
            avg_self_transition = np.mean(diag_vals)
            f.write(f"1. On average, regimes have a **{avg_self_transition:.2f} probability** of ")
            f.write("persisting from one time step to the next, indicating a degree of regime stability.\n\n")
            
            # Find the highest non-diagonal value
            mask = np.ones(transition_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            if np.any(mask):
                highest_trans = np.max(transition_matrix.values * mask)
                if highest_trans > 0.2:
                    i, j = np.where((transition_matrix.values * mask) == highest_trans)
                    if len(i) > 0 and len(j) > 0:
                        from_regime = transition_matrix.index[i[0]]
                        to_regime = transition_matrix.columns[j[0]]
                        f.write(f"2. The most likely regime transition is from **{from_regime}** to ")
                        f.write(f"**{to_regime}** with a probability of **{highest_trans:.2f}**.\n\n")
        
        # 5. Visualizations
        f.write("## 5. Visualizations\n\n")
        
        f.write("The following visualizations are available in the results directory:\n\n")
        
        # Reference visualization files
        f.write("### Regime Evolution\n\n")
        f.write(f"![Regime Evolution](regime_evolution_{best_model}.png)\n\n")
        
        f.write("### Regime Distribution (UMAP)\n\n")
        f.write(f"![Regime Distribution](regime_distribution_{best_model}_umap.png)\n\n")
        
        f.write("### Transition Heatmap\n\n")
        f.write(f"![Transition Heatmap](transition_heatmap_{best_model}.png)\n\n")
        
        f.write("### Feature Distributions\n\n")
        f.write("Multiple feature distribution plots are available showing how key features vary across regimes.\n\n")
        
        # 6. Trading Implications
        f.write("## 6. Trading Implications\n\n")
        
        f.write("Different market regimes require different trading strategies. Based on the identified regimes:\n\n")
        
        if best_model in regime_characteristics:
            chars = regime_characteristics[best_model]
            
            # Generate trading implications for each regime
            for label, stats in chars.items():
                regime_name = stats.get('name', f'Regime {label}')
                f.write(f"### {regime_name}\n\n")
                
                # Generate specific recommendations based on regime characteristics
                implications = []
                
                # Volatility-based recommendations
                if 'volatility_type' in stats:
                    if stats['volatility_type'] == 'High':
                        implications.append("- **Wider stop-losses** may be needed due to high volatility")
                        implications.append("- Consider **volatility-based position sizing** to manage risk")
                    elif stats['volatility_type'] == 'Low':
                        implications.append("- Tighter stop-losses can be used due to lower volatility")
                        implications.append("- May require **larger position sizes** to achieve target returns")
                
                # Price behavior recommendations
                if 'price_behavior' in stats:
                    if stats['price_behavior'] == 'Mean-reverting':
                        implications.append("- **Mean-reversion strategies** like range trading or oscillator-based approaches")
                        implications.append("- Consider **fading extreme moves** expecting reversion to the mean")
                    elif stats['price_behavior'] == 'Trending':
                        implications.append("- **Trend-following strategies** using moving averages or breakout systems")
                        implications.append("- **Trailing stops** to capture extended moves in the trend direction")
                
                # Liquidity recommendations
                if 'liquidity' in stats:
                    if stats['liquidity'] == 'Low':
                        implications.append("- **Reduce position sizes** due to higher transaction costs")
                        implications.append("- Use **limit orders** instead of market orders to minimize slippage")
                    elif stats['liquidity'] == 'High':
                        implications.append("- Can execute larger orders with minimal market impact")
                        implications.append("- More suitable for **high-frequency strategies** due to lower transaction costs")
                
                # Add the trading implications
                if implications:
                    for imp in implications:
                        f.write(f"{imp}\n")
                else:
                    f.write("- Trading implications need to be determined based on specific strategy requirements\n")
                
                f.write("\n")
        
        # 7. Conclusion
        f.write("## 7. Conclusion and Next Steps\n\n")
        
        f.write("This analysis has successfully identified distinct market regimes with ")
        f.write("characteristic behaviors in terms of volatility, trend, and liquidity. ")
        f.write("These regimes provide a foundation for developing adaptive trading strategies.\n\n")
        
        f.write("### Recommended Next Steps\n\n")
        
        f.write("1. **Regime Prediction Model**: Develop a model to predict regime shifts in advance\n")
        f.write("2. **Strategy Optimization**: Fine-tune trading strategies for each specific regime\n")
        f.write("3. **Extended Validation**: Validate regime detection across different market conditions and timeframes\n")
        f.write("4. **Feature Importance Analysis**: Identify which features are most significant in each regime\n")
        f.write("5. **Integrate with Execution Framework**: Deploy regime detection in a live trading environment\n\n")
        
        f.write("---\n\n")
        f.write("© Market Regime Detection Analysis | Generated on ")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logger.info(f"Markdown report generated at {report_path}")
    return report_path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Market Regime Detection')
    
    parser.add_argument('--features', type=str, default=None,
                       help='Path to features file (CSV or Parquet). If not provided, synthetic data will be generated.')
    
    parser.add_argument('--output', type=str, default='./results',
                       help='Directory to save results')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Load features if provided
    features_df = None
    if args.features:
        if os.path.exists(args.features):
            if args.features.endswith('.csv'):
                features_df = pd.read_csv(args.features, index_col=0, parse_dates=True)
            elif args.features.endswith('.parquet'):
                features_df = pd.read_parquet(args.features)
            else:
                logger.error(f"Unsupported file format: {args.features}. Use CSV or Parquet.")
                sys.exit(1)
                
            logger.info(f"Loaded {len(features_df)} samples from {args.features}")
        else:
            logger.error(f"Features file not found: {args.features}")
            sys.exit(1)
    
    # Run the regime detection pipeline
    results = run_regime_detection(
        features_df=features_df,
        output_dir=args.output
    )
    
    # Print path to the report
    logger.info(f"Analysis complete. Report saved to {results['report_path']}")
    logger.info(f"Visualizations saved to {args.output}")

"""
Market Regime Detection Pipeline

This script demonstrates how to run the market regime detection pipeline 
using both order book (depth20) and trade data (aggTrade).
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from backend.data_pipeline.data_loader import MarketDataLoader
from backend.data_pipeline.feature_engineering.feature_extractor import FeatureExtractor
from backend.ml_service.regime_detector import RegimeDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the market regime detection pipeline.
    """
    # Parameters
    symbol = "BNBFDUSD"  # Use BNBFDUSD as available in the data
    start_date = "20250314"  # First date in dataset
    end_date = "20250317"    # Last date in dataset
    resample_interval = "5min"
    data_directory = "data"
    
    # Initialize components
    data_loader = MarketDataLoader(data_dir=data_directory)
    feature_extractor = FeatureExtractor(data_dir=data_directory)
    
    # List available dates
    available_dates = data_loader.list_available_dates()
    logger.info(f"Available dates: {available_dates}")
    
    # Extract features for all days in range
    features = feature_extractor.extract_multi_day_features(
        symbol=symbol, 
        start_date=start_date,
        end_date=end_date,
        resample_interval=resample_interval,
        window_sizes=[5, 15, 30]
    )
    
    if features.empty:
        logger.error("Feature extraction failed. No features generated.")
        return
    
    logger.info(f"Extracted features shape: {features.shape}")
    logger.info(f"Feature columns: {features.columns.tolist()}")
    
    # Select features for clustering
    feature_cols = [col for col in features.columns if col != 'timestamp']
    
    # Initialize regime detector with GMM model (you can change to KMeans or HDBSCAN)
    regime_detector = RegimeDetector(model_type='gmm', n_clusters=4)
    
    # Train the model
    logger.info("Training regime detector...")
    regime_detector.fit(features[feature_cols])
    
    # Predict regimes
    logger.info("Predicting market regimes...")
    features['regime'] = regime_detector.predict(features[feature_cols])
    
    # Get regime probabilities
    if hasattr(regime_detector.model, 'predict_proba'):
        probs = regime_detector.model.predict_proba(features[feature_cols])
        for i in range(probs.shape[1]):
            features[f'regime_prob_{i}'] = probs[:, i]
    
    # Plot results
    logger.info("Creating visualization...")
    plot_regimes(features, symbol)
    
    # Save results
    output_file = f"regime_results_{symbol}_{start_date}_to_{end_date}.csv"
    features.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
def plot_regimes(features, symbol):
    """
    Plot price and detected regimes.
    
    Args:
        features (pd.DataFrame): DataFrame with features and regime labels
        symbol (str): Trading symbol
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Price with regimes as color
    plt.subplot(2, 1, 1)
    for regime in sorted(features['regime'].unique()):
        mask = features['regime'] == regime
        plt.scatter(
            features.loc[mask, 'timestamp'], 
            features.loc[mask, 'price'],
            label=f'Regime {regime}',
            alpha=0.7,
            s=30
        )
    
    plt.title(f'Market Regimes for {symbol}')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Regime distribution over time
    plt.subplot(2, 1, 2)
    
    # If we have regime probabilities
    prob_cols = [col for col in features.columns if col.startswith('regime_prob_')]
    if prob_cols:
        regime_data = features[prob_cols].values
        plt.stackplot(
            features['timestamp'], 
            [features[col] for col in prob_cols],
            labels=[f'Regime {i}' for i in range(len(prob_cols))],
            alpha=0.7
        )
        plt.title('Regime Probability Distribution')
    else:
        # Simple regime count over time
        for regime in sorted(features['regime'].unique()):
            regime_count = features['regime'].rolling(20).apply(
                lambda x: (x == regime).sum() / len(x)
            )
            plt.plot(features['timestamp'], regime_count, label=f'Regime {regime}')
        plt.title('Regime Distribution (20-period moving window)')
    
    plt.ylabel('Probability / Distribution')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'regimes_{symbol}.png')
    logger.info(f"Plot saved as regimes_{symbol}.png")
    plt.close()

if __name__ == "__main__":
    main()