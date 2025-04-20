#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Market Regime Detection Backtesting

This script sets up the necessary directory structure and runs the backtesting engine
on the provided trade and orderbook data files.
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from datetime import datetime

# Import project modules
from data_pipeline.feature_engineering.normalizer import Normalizer
from data_pipeline.feature_engineering.feature_normalization_validation import validate_normalization_pipeline
from clustering_executor import ClusteringExecutor
from regime_analyzer import RegimeAnalyzer
from ml_service.regime_detector import RegimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('market_regime_detection')

def setup_environment():
    """
    Set up the directory structure and prepare the environment for running backtests.
    """
    # Create necessary directories
    directories = [
        './data',
        './results',
        './reports',
        './models',
        './data_pipeline/feature_engineering',
        './backtesting/engine'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create empty __init__.py files for proper imports
    init_paths = [
        './__init__.py',
        './models/__init__.py',
        './data_pipeline/__init__.py',
        './data_pipeline/feature_engineering/__init__.py',
        './backtesting/__init__.py',
        './backtesting/engine/__init__.py'
    ]
    
    for init_path in init_paths:
        with open(init_path, 'w') as f:
            f.write("# Package initialization file")
    
    print("Environment set up successfully.")

def check_files_exist():
    """
    Check if necessary data files exist.
    """
    required_files = [
        'aggtrade.txt',
        'depth20.txt',
        'backtesting_engine.py'
    ]
    
    missing_files = [file for file in required_files if not os.path.exists(file)]
    
    if missing_files:
        print(f"Error: The following required files are missing: {', '.join(missing_files)}")
        print("Please make sure all required files are in the current directory.")
        return False
    
    return True

def load_features(file_path):
    """
    Load feature data from CSV or Parquet file
    
    Args:
        file_path: Path to the feature file
        
    Returns:
        DataFrame containing features
    """
    logger.info(f"Loading features from {file_path}")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")
    
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df

def run_pipeline(features_path, output_dir='./results', normalization_method='standard', 
                use_pca=False, pca_components=10, generate_report=True):
    """
    Run the entire market regime detection pipeline
    
    Args:
        features_path: Path to the feature file
        output_dir: Directory to save results
        normalization_method: Method for feature normalization
        use_pca: Whether to use PCA for dimensionality reduction
        pca_components: Number of PCA components if using PCA
        generate_report: Whether to generate a summary report
        
    Returns:
        Dictionary containing all results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Start timer
    start_time = datetime.now()
    
    # 1. Load features
    features_df = load_features(features_path)
    
    # 2. Validate normalization pipeline
    logger.info("Validating normalization pipeline...")
    validation_results = validate_normalization_pipeline()
    
    # 3. Run clustering
    logger.info("Running clustering pipeline...")
    cluster_executor = ClusteringExecutor(features_df)
    
    # Normalize features
    pca_components_arg = pca_components if use_pca else None
    normalized_features = cluster_executor.normalize_features(
        method=normalization_method, 
        pca_components=pca_components_arg
    )
    
    # Setup and run clustering models
    cluster_executor.setup_models()
    labels_dict = cluster_executor.run_clustering()
    
    # Evaluate models
    evaluation_results = cluster_executor.evaluate_models()
    
    # Visualize clusters for the best model
    best_model = cluster_executor.get_best_model(metric='silhouette_score')
    cluster_executor.visualize_clusters(method='umap', model_name=best_model)
    cluster_executor.visualize_clusters(method='tsne', model_name=best_model)
    
    # Save labels
    cluster_executor.save_labels(model_name=best_model)
    
    # 4. Analyze regimes
    logger.info("Analyzing market regimes...")
    analyzer = RegimeAnalyzer(features_df, labels_dict, output_dir=output_dir)
    
    # Analyze regimes for all models
    regime_characteristics = analyzer.analyze_regimes()
    
    # Analyze transitions for the best model
    transition_matrix = analyzer.analyze_transitions(model_name=best_model)
    analyzer.visualize_transitions(model_name=best_model)
    
    # Visualize regime evolution
    analyzer.visualize_regime_evolution(model_name=best_model, overlay_price=True)
    
    # Visualize regime distribution
    analyzer.visualize_regime_distribution(model_name=best_model, method='umap')
    
    # Visualize key features
    analyzer.visualize_key_features(model_name=best_model, top_n=5)
    
    # Generate summary report
    if generate_report:
        logger.info("Generating summary report...")
        summary = analyzer.generate_summary_report(model_name=best_model)
        
        # Create a formatted text report
        report_path = create_report(
            features_df, 
            labels_dict, 
            evaluation_results, 
            regime_characteristics,
            transition_matrix, 
            output_dir
        )
    
    # Calculate run time
    run_time = datetime.now() - start_time
    logger.info(f"Pipeline completed in {run_time}")
    
    return {
        "features": features_df,
        "normalized_features": normalized_features,
        "labels": labels_dict,
        "evaluation_results": evaluation_results,
        "best_model": best_model,
        "regime_characteristics": regime_characteristics,
        "transition_matrix": transition_matrix,
        "run_time": run_time
    }

def create_report(features, labels, evaluation_results, regime_characteristics, 
                 transition_matrix, output_dir):
    """
    Create a concise report summarizing the regime detection analysis
    
    Args:
        features: DataFrame of features
        labels: Dictionary of model names and cluster labels
        evaluation_results: DataFrame of model evaluation results
        regime_characteristics: Dictionary of regime characteristics
        transition_matrix: DataFrame of regime transition probabilities
        output_dir: Directory to save the report
        
    Returns:
        Path to the saved report
    """
    output_path = Path(output_dir)
    report_path = output_path / "regime_detection_report.md"
    
    with open(report_path, "w") as f:
        # Title
        f.write("# Market Regime Detection Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. Data Summary
        f.write("## 1. Data Summary\n\n")
        f.write(f"- **Number of samples**: {len(features)}\n")
        f.write(f"- **Features used**: {len(features.columns)}\n")
        f.write("- **Time period**: "
               f"{features.index.min().strftime('%Y-%m-%d %H:%M:%S')} to "
               f"{features.index.max().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # List important features
        f.write("### Key Features\n\n")
        feature_groups = {
            "Liquidity": [col for col in features.columns if 'spread' in col or 'depth' in col],
            "Volatility": [col for col in features.columns if 'volatility' in col or 'std' in col],
            "Volume": [col for col in features.columns if 'volume' in col],
            "Price": [col for col in features.columns if 'price' in col or 'returns' in col]
        }
        
        for group, cols in feature_groups.items():
            if cols:
                f.write(f"**{group} Features**:\n")
                for col in cols[:5]:  # Show the first 5 features in each group
                    f.write(f"- {col}\n")
                if len(cols) > 5:
                    f.write(f"- ... and {len(cols) - 5} more\n")
                f.write("\n")
        
        # 2. Clustering Model Evaluation
        f.write("## 2. Clustering Model Evaluation\n\n")
        
        # Show best model
        best_model = evaluation_results.iloc[0]['model']
        f.write(f"**Best model**: {best_model} with "
               f"silhouette score = {evaluation_results.iloc[0]['silhouette_score']:.4f}\n\n")
        
        # Show top 5 models in a markdown table
        f.write("### Top 5 Models\n\n")
        f.write("| Model | Clusters | Silhouette Score ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |\n")
        f.write("|-------|----------|-------------------|-----------------|---------------------|\n")
        
        for i in range(min(5, len(evaluation_results))):
            row = evaluation_results.iloc[i]
            f.write(f"| {row['model']} | {row['n_clusters']} | {row['silhouette_score']:.4f} | "
                   f"{row['davies_bouldin_score']:.4f} | {row['calinski_harabasz_score']:.4f} |\n")
        f.write("\n")
        
        # 3. Regime Characteristics
        f.write("## 3. Regime Characteristics\n\n")
        
        # Get characteristics of the best model
        if best_model in regime_characteristics:
            characteristics = regime_characteristics[best_model]
            
            f.write("| Regime | Name | Size | Volatility | Price Behavior | Liquidity | Volume |\n")
            f.write("|--------|------|------|------------|----------------|-----------|--------|\n")
            
            for label, stats in characteristics.items():
                regime_name = stats.get('name', f'Regime {label}')
                size = stats.get('size', 'N/A')
                volatility = stats.get('volatility_type', 'N/A')
                price_behavior = stats.get('price_behavior', 'N/A')
                liquidity = stats.get('liquidity', 'N/A')
                volume = stats.get('volume_type', 'N/A')
                
                f.write(f"| {label} | {regime_name} | {size:.2%} | {volatility} | "
                       f"{price_behavior} | {liquidity} | {volume} |\n")
            
            f.write("\n")
        
        # 4. Regime Transitions
        f.write("## 4. Regime Transitions\n\n")
        f.write("### Transition Probability Matrix\n\n")
        
        if isinstance(transition_matrix, pd.DataFrame):
            # Format the transition matrix as a markdown table
            f.write("| From / To | " + " | ".join(transition_matrix.columns) + " |\n")
            f.write("|" + "|".join(["---"] * (len(transition_matrix.columns) + 1)) + "|\n")
            
            for idx, row in transition_matrix.iterrows():
                f.write(f"| {idx} | " + " | ".join([f"{val:.2f}" for val in row.values]) + " |\n")
            
            f.write("\n")
            
            # Find most common transitions
            flattened = []
            for i in range(len(transition_matrix)):
                for j in range(len(transition_matrix.columns)):
                    if i != j and transition_matrix.iloc[i, j] > 0.2:  # Only significant transitions
                        flattened.append({
                            'from': transition_matrix.index[i],
                            'to': transition_matrix.columns[j],
                            'probability': transition_matrix.iloc[i, j]
                        })
            
            if flattened:
                sorted_transitions = sorted(flattened, key=lambda x: x['probability'], reverse=True)
                
                f.write("### Common Regime Transitions\n\n")
                f.write("| From | To | Probability |\n")
                f.write("|------|----|--------------|\n")
                
                for t in sorted_transitions[:5]:  # Show top 5 transitions
                    f.write(f"| {t['from']} | {t['to']} | {t['probability']:.2f} |\n")
                
                f.write("\n")
        
        # 5. Key Insights
        f.write("## 5. Key Insights\n\n")
        
        # Add placeholder insights that should be filled in manually or generated programmatically
        f.write("1. **Regime Distribution**: The most common market regime is [FILL_IN], occurring [FILL_IN]% of the time.\n\n")
        f.write("2. **Volatility Patterns**: [FILL_IN] regimes show the highest volatility, characterized by [FILL_IN].\n\n")
        f.write("3. **Transitional Behavior**: The market tends to transition from [FILL_IN] to [FILL_IN] with high probability.\n\n")
        f.write("4. **Liquidity Conditions**: [FILL_IN] regimes exhibit the lowest liquidity, which correlates with [FILL_IN].\n\n")
        f.write("5. **Trading Implications**: Different regimes require different trading strategies. For example, in [FILL_IN] regime, [FILL_IN] strategies may be more effective.\n\n")
        
        # 6. Visualizations
        f.write("## 6. Visualization Links\n\n")
        f.write("- [Regime Evolution Plot](regime_evolution_{}.png)\n".format(best_model))
        f.write("- [Regime Distribution Plot](regime_distribution_{}_umap.png)\n".format(best_model))
        f.write("- [Transition Heatmap](transition_heatmap_{}.png)\n".format(best_model))
        f.write("- [Feature Distribution Plots](feature_distribution_*.png)\n")
        
        # 7. Conclusion
        f.write("## 7. Conclusion\n\n")
        f.write("This analysis identified {} distinct market regimes with clear ".format(len(characteristics) if best_model in regime_characteristics else "multiple"))
        f.write("characteristics in terms of volatility, trend behavior, and liquidity. ")
        f.write("These regimes provide a foundation for developing regime-specific trading strategies ")
        f.write("that can adapt to changing market conditions.\n\n")
        f.write("Further analysis could focus on developing predictive models for regime transitions ")
        f.write("and optimizing trading strategies for each regime type.\n")
    
    logger.info(f"Report saved to {report_path}")
    return report_path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Market Regime Detection')
    
    parser.add_argument('--features', type=str, required=True,
                       help='Path to the features file (CSV or Parquet)')
    
    parser.add_argument('--output', type=str, default='./results',
                       help='Directory to save results')
    
    parser.add_argument('--norm', type=str, default='standard',
                       choices=['standard', 'minmax', 'robust'],
                       help='Normalization method')
    
    parser.add_argument('--use-pca', action='store_true',
                       help='Use PCA for dimensionality reduction')
    
    parser.add_argument('--pca-components', type=int, default=10,
                       help='Number of PCA components if using PCA')
    
    parser.add_argument('--no-report', action='store_true',
                       help='Skip report generation')
    
    return parser.parse_args()

def main():
    """
    Main function to set up and run the backtesting.
    """
    print("Setting up environment for market regime detection backtesting...")
    setup_environment()
    
    if not check_files_exist():
        return
    
    # Move data files to data directory
    data_files = ['aggtrade.txt', 'depth20.txt']
    for file in data_files:
        if os.path.exists(file):
            shutil.copy(file, f"./data/{file}")
    
    # Copy backtesting script to proper location
    shutil.copy('backtesting_script.py', './backtesting_engine_run.py')
    
    print("\nEverything is set up. Running backtesting script...\n")
    
    # Run the backtesting script
    os.system('python backtesting_engine_run.py')
    
    print("\nBacktesting completed. Check the results and reports directories for outputs.")

if __name__ == "__main__":
    args = parse_arguments()
    
    results = run_pipeline(
        features_path=args.features,
        output_dir=args.output,
        normalization_method=args.norm,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        generate_report=not args.no_report
    )
    
    logger.info("Pipeline execution completed successfully")