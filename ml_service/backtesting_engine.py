#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Market Regime Detection Backtesting Script

This script demonstrates how to use the BacktestingEngine to evaluate
different regime detection models on the provided trade and orderbook data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Import the models (assuming these are in the correct directory structure)
from models.base_model import BaseModel
from ml_service.models.kmeans import KMeansModel
from ml_service.models.gmm import GMMModel
from models.hdbscan import HDBSCANModel
from data_pipeline.feature_engineering.feature_extractor import FeatureExtractor
from data_pipeline.feature_engineering.normalizer import Normalizer
from backtesting_engine import BacktestingEngine

def load_data(trades_file, orderbook_file):
    """
    Load and preprocess the trade and orderbook data from the provided files.
    
    Args:
        trades_file (str): Path to the trades data file
        orderbook_file (str): Path to the orderbook data file
        
    Returns:
        tuple: Preprocessed trade and orderbook DataFrames
    """
    print(f"Loading data from {trades_file} and {orderbook_file}")
    
    # Load trade data
    trades_df = pd.read_csv(trades_file, delimiter=',')
    
    # Extract column names from the first row if needed
    if 'Time' not in trades_df.columns and 'Time' in trades_df.iloc[0].values:
        header = trades_df.iloc[0].values
        trades_df = pd.read_csv(trades_file, delimiter=',', names=header, skiprows=1)
    
    # Convert time to datetime
    trades_df['Time'] = pd.to_datetime(trades_df['Time'])
    
    # Load orderbook data - it's more complex due to the many columns
    orderbook_df = pd.read_csv(orderbook_file, delimiter=',')
    
    # Extract column names from the first row if needed
    if len(orderbook_df.columns) < 10 and len(orderbook_df.iloc[0].values) > 10:
        header_row = orderbook_df.iloc[0].values
        # Generate column names for all the bid/ask levels
        header = []
        for col in header_row:
            if col not in [None, '', 'nan']:
                header.append(col.strip())
        
        orderbook_df = pd.read_csv(orderbook_file, delimiter=',', names=header, skiprows=1)
    
    # Convert time to datetime
    orderbook_df['Time'] = pd.to_datetime(orderbook_df['Time'])
    
    # Set Time as index
    trades_df.set_index('Time', inplace=True)
    orderbook_df.set_index('Time', inplace=True)
    
    return trades_df, orderbook_df

class CustomFeatureExtractor(FeatureExtractor):
    """
    Custom feature extractor for market regime detection.
    """
    
    def extract_features(self, trades_df, orderbook_df):
        """
        Extract features from the trade and orderbook data.
        
        Args:
            trades_df: DataFrame containing trade data
            orderbook_df: DataFrame containing orderbook data
            
        Returns:
            DataFrame containing extracted features
        """
        print("Extracting features...")
        
        # Create a result dataframe with all timestamps from both dataframes
        all_times = trades_df.index.union(orderbook_df.index).sort_values()
        features_df = pd.DataFrame(index=all_times)
        
        # Extract basic price and volume features from trades
        if 'Price' in trades_df.columns:
            # Forward fill to handle missing values
            price_series = trades_df['Price'].reindex(all_times).ffill()
            features_df['price'] = price_series
            
            # Calculate returns
            features_df['returns'] = price_series.pct_change()
            
            # Calculate volume features
            if 'Quantity' in trades_df.columns:
                qty_series = trades_df['Quantity'].reindex(all_times).fillna(0)
                features_df['volume'] = qty_series
                
                # Compute rolling volume
                features_df['rolling_volume_10s'] = qty_series.rolling('10s').sum()
                features_df['rolling_volume_30s'] = qty_series.rolling('30s').sum()
        
        # Extract orderbook features
        if len(orderbook_df) > 0:
            # Find the bid/ask column names
            bid_cols = [col for col in orderbook_df.columns if 'BidPrice' in col]
            ask_cols = [col for col in orderbook_df.columns if 'AskPrice' in col]
            bid_qty_cols = [col for col in orderbook_df.columns if 'BidQty' in col]
            ask_qty_cols = [col for col in orderbook_df.columns if 'AskQty' in col]
            
            if bid_cols and ask_cols:
                # Get best bid/ask
                best_bid_col = sorted(bid_cols, key=lambda x: int(x.replace('BidPriceL', '')))[-1]
                best_ask_col = sorted(ask_cols, key=lambda x: int(x.replace('AskPriceL', '')))[-1]
                
                # Forward fill to handle missing values
                best_bid = orderbook_df[best_bid_col].reindex(all_times).ffill()
                best_ask = orderbook_df[best_ask_col].reindex(all_times).ffill()
                
                # Calculate spread
                features_df['spread'] = best_ask - best_bid
                features_df['mid_price'] = (best_ask + best_bid) / 2
                features_df['mid_price_returns'] = features_df['mid_price'].pct_change()
                
                # Calculate rolling volatility
                features_df['volatility_10s'] = features_df['mid_price_returns'].rolling('10s').std()
                features_df['volatility_30s'] = features_df['mid_price_returns'].rolling('30s').std()
                
                # Calculate directional features
                features_df['price_direction'] = features_df['mid_price_returns'].rolling('30s').mean()
                
                # Compute imbalance features if quantities are available
                if bid_qty_cols and ask_qty_cols:
                    best_bid_qty_col = f"BidQtyL{best_bid_col.replace('BidPriceL', '')}"
                    best_ask_qty_col = f"AskQtyL{best_ask_col.replace('AskPriceL', '')}"
                    
                    if best_bid_qty_col in orderbook_df.columns and best_ask_qty_col in orderbook_df.columns:
                        best_bid_qty = orderbook_df[best_bid_qty_col].reindex(all_times).ffill()
                        best_ask_qty = orderbook_df[best_ask_qty_col].reindex(all_times).ffill()
                        
                        # Calculate order book imbalance
                        sum_qty = best_bid_qty + best_ask_qty
                        features_df['imbalance_lvl1'] = np.where(
                            sum_qty > 0, (best_bid_qty - best_ask_qty) / sum_qty, 0
                        )
                        
                        # Calculate microprice
                        features_df['microprice'] = np.where(
                            sum_qty > 0, 
                            (best_bid * best_ask_qty + best_ask * best_bid_qty) / sum_qty,
                            features_df['mid_price']
                        )
                        
                        # Calculate liquidity features
                        features_df['liquidity'] = best_bid_qty + best_ask_qty
                
                # Calculate cumulative depth if possible
                if bid_qty_cols and ask_qty_cols:
                    cum_bid_qty = sum(orderbook_df[col].reindex(all_times).ffill().fillna(0) for col in bid_qty_cols)
                    cum_ask_qty = sum(orderbook_df[col].reindex(all_times).ffill().fillna(0) for col in ask_qty_cols)
                    
                    features_df['cum_bid_qty'] = cum_bid_qty
                    features_df['cum_ask_qty'] = cum_ask_qty
                    features_df['depth_imbalance'] = np.where(
                        (cum_bid_qty + cum_ask_qty) > 0,
                        (cum_bid_qty - cum_ask_qty) / (cum_bid_qty + cum_ask_qty),
                        0
                    )
        
        # Drop rows with NaN values
        features_df = features_df.dropna()
        
        print(f"Extracted {features_df.shape[1]} features")
        return features_df

def main():
    """
    Main function to run the backtesting engine.
    """
    # Data files
    trades_file = "aggtrade.txt"
    orderbook_file = "depth20.txt"
    
    # Check if files exist
    if not os.path.exists(trades_file) or not os.path.exists(orderbook_file):
        print(f"Error: Data files not found. Please ensure {trades_file} and {orderbook_file} exist.")
        return
    
    # Create feature extractor and normalizer
    feature_extractor = CustomFeatureExtractor()
    normalizer = Normalizer()
    
    # Create models to evaluate
    models = [
        KMeansModel(n_clusters=3, name="KMeans-3"),
        KMeansModel(n_clusters=5, name="KMeans-5"),
        GMMModel(n_components=3, name="GMM-3"),
        GMMModel(n_components=5, name="GMM-5"),
        HDBSCANModel(min_cluster_size=5, name="HDBSCAN-5")
    ]
    
    # Configuration for the backtesting engine
    config = {
        'backtest_params': {
            'train_split': 0.7,
            'evaluation_metrics': ['silhouette', 'davies_bouldin', 'calinski_harabasz'],
            'rolling_window': '1D',
            'feature_set': 'full'
        }
    }
    
    # Create backtesting engine
    engine = BacktestingEngine(
        config=config,
        data_path="./data",
        models=models,
        feature_extractor=feature_extractor,
        normalizer=normalizer
    )
    
    # Load data
    trades_df, orderbook_df = load_data(trades_file, orderbook_file)
    
    # Prepare features
    features_df = engine.prepare_features(trades_df, orderbook_df)
    
    # Run backtest
    results = engine.run_backtest(trades_file, orderbook_file)
    
    # Save results
    results_file = engine.save_results("./results")
    print(f"Results saved to {results_file}")
    
    # Generate report
    report_file = engine.generate_report("./reports")
    print(f"Report saved to {report_file}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # 1. Plot regime transitions
    fig1 = engine.plot_regime_transitions()
    plt.savefig("./results/regime_transitions.png")
    
    # 2. Plot regime characteristics
    fig2 = engine.plot_regime_characteristics()
    plt.savefig("./results/regime_characteristics.png")
    
    # 3. Plot transition matrix
    fig3 = engine.plot_transition_matrix()
    plt.savefig("./results/transition_matrix.png")
    
    # 4. Compare models
    fig4 = engine.compare_models(metric='silhouette_score')
    plt.savefig("./results/model_comparison.png")
    
    # Print best model
    best_model = engine.get_best_model(metric='silhouette_score')
    print(f"Best model based on silhouette score: {best_model}")
    
    # Print regime characteristics for the best model
    if best_model and best_model in engine.results['models']:
        print("\nRegime Characteristics for the best model:")
        regime_chars = engine.results['models'][best_model]['regime_characteristics']
        for regime, chars in regime_chars.items():
            print(f"Regime {regime} ({chars.get('description', 'Unknown')}):")
            print(f"  - Size: {chars['size']:.2%}")
            print("  - Average Features:")
            for feat, val in chars['avg_features'].items():
                if any(keyword in feat for keyword in ['volatility', 'direction', 'imbalance', 'spread', 'liquidity']):
                    print(f"    {feat}: {val:.4f}")
            print()
    
    print("Backtesting completed successfully.")

if __name__ == "__main__":
    main()