# Market Regime Detection

A sophisticated system for identifying and analyzing market regimes using unsupervised machine learning on high-frequency financial data.

## Overview

This repository contains a framework for detecting distinct market states (regimes) from order book and trade data. The system extracts relevant features, applies clustering algorithms, and provides tools for visualizing and analyzing the detected regimes.

Market regimes can be broadly understood as distinct states of market behavior characterized by specific patterns in volatility, trend, liquidity, and other market dynamics. This system helps traders and researchers identify these regimes automatically, enabling strategy adaptation and risk management.

## Features

- **Comprehensive Feature Engineering**: 
  - Order book metrics (imbalance, depth, spread)
  - Price-based features (volatility, momentum, z-scores)
  - Volume analysis
  - Multi-timeframe calculations
  - Advanced volatility estimators (Parkinson, directional)
  
- **Multiple Clustering Algorithms**:
  - K-Means
  - Gaussian Mixture Models (GMM)
  - HDBSCAN
  
- **Robust Evaluation Framework**:
  - Silhouette score
  - Davies-Bouldin index
  - Calinski-Harabasz index
  
- **Visualization Tools**:
  - 2D cluster visualization (UMAP, t-SNE, PCA)
  - Regime evolution over time
  - Cluster quality comparisons
  
- **Regime Analysis**:
  - Characteristic profiling
  - Transition probability analysis
  - Statistical summaries

## Project Structure

```
Directory structure:
└── k3tikvats-market_regime_detection/
    ├── clustering_executor.py # Execution pipeline for clustering
    ├── regime_analyzer.py
    ├── run_regime_detection.py
    ├── data_pipeline/  # Data acquisition and processing
    │   ├── data_loader.py  # Loading raw market data
    │   └── feature_engineering/ # Feature extraction and normalization
    │       ├── feature_extractor.py  # Extract features from raw data
    │       ├── feature_normalization_validation.py
    │       └── normalizer.py    # Feature normalization and reduction
    ├── ml_service/ # Machine learning components
    │   ├── backtesting_engine.py
    │   ├── regime_detector.py # Main market regime detection service
    │   └── models/ # Clustering models
    │       ├── _init_.py
    │       ├── base_model.py
    │       ├── gmm.py  # Gaussian mixture model
    │       ├── hdbscan.py   # HDBSCAN clustering
    │       ├── kmeans.py # K-means clustering
    │       └── __pycache__/
    └── results/ # Output directory for results and visualizations
        ├── market_regime_analysis.md
        ├── model_evaluation.csv
        ├── regime_characteristics_gmm_3_diag.csv
        ├── regime_characteristics_gmm_3_full.csv
        ├── regime_characteristics_gmm_5_diag.csv
        ├── regime_characteristics_gmm_5_full.csv
        ├── regime_characteristics_hdbscan_10.csv
        ├── regime_characteristics_hdbscan_15.csv
        ├── regime_characteristics_hdbscan_5.csv
        ├── regime_characteristics_kmeans_3.csv
        ├── regime_characteristics_kmeans_5.csv
        ├── regime_characteristics_kmeans_7.csv
        ├── regime_report_hdbscan_10.json
        ├── synthetic_features.csv
        └── transition_matrix_hdbscan_10.csv
                     # Output directory for results and visualizations
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/market-regime-detection.git
cd market-regime-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from ml_service.regime_detector import MarketRegimeDetector
from data_pipeline.feature_engineering.feature_extractor import FeatureExtractor

# Extract features
extractor = FeatureExtractor(data_dir='path/to/data')
features = extractor.extract_multi_day_features(
    symbol='BNBFDUSD',
    start_date='20240101',
    end_date='20240131',
    resample_interval='5min',
    window_sizes=[5, 15, 30]
)

# Detect regimes
detector = MarketRegimeDetector(config={
    'normalization_method': 'standard',
    'pca_components': 20,
    'selected_model': 'kmeans'
})

detector.fit(features)
regime_labels = detector.predict(features)

# Get regime characteristics
regime_info = detector.analyze_regimes()
print(regime_info)

# Visualize regimes
detector.visualize_regimes(method='tsne')
```

### Advanced Usage with Clustering Executor

```python
from clustering_executor import ClusteringExecutor

# Initialize with preprocessed features
executor = ClusteringExecutor(features_df)

# Normalize features
normalized_features = executor.normalize_features(
    method='standard',
    pca_components=15
)

# Setup and run all clustering models
executor.setup_models()
labels = executor.run_clustering()

# Evaluate models
evaluation = executor.evaluate_models()
print(evaluation)

# Find best model and visualize
best_model = executor.get_best_model(metric='silhouette_score')
executor.visualize_clusters(method='umap', model_name=best_model)
```

## Example Results

The market regime detection system typically identifies several distinct market regimes, such as:

1. **Trending & High Volatility**: Strong directional movement with large price swings
2. **Mean-reverting & Low Liquidity**: Oscillating price action with thin order books
3. **High Liquidity & Low Volatility**: Stable price action with deep order books
4. **Transitional**: Brief periods between major regimes

These regimes can be visualized and analyzed to inform trading decisions and risk management.

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- UMAP-learn
- HDBSCAN


<!-- 
## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request -->

## Citation

If you use this code in your research or project, please cite:

```
@software{market_regime_detection,
  author = {Your Name},
  title = {Market Regime Detection},
  year = {2025},
  url = {https://github.com/yourusername/market-regime-detection}
}
```