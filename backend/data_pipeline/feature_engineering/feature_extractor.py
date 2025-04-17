"""
Feature extractor for market regime detection.
Extracts features from order book and trade data.
"""
import os
import pandas as pd
import numpy as np
from ..data_loader import MarketDataLoader
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extracts and computes features from market data for regime detection.
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the feature extractor.
        
        Args:
            data_dir (str): Directory containing the data
        """
        self.data_loader = MarketDataLoader(data_dir=data_dir)
        
    def extract_features(self, symbol, date_str, resample_interval='5min', window_sizes=[5, 15, 30]):
        """
        Extract features from market data for a given symbol and date.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BNBFDUSD')
            date_str (str): Date string in format YYYYMMDD
            resample_interval (str): Interval for resampling time series
            window_sizes (list): List of window sizes for rolling calculations
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        logger.info(f"Extracting features for {symbol} on {date_str}")
        
        # Load combined data
        raw_data = self.data_loader.load_combined_data(symbol, date_str, resample_interval)
        
        if raw_data.empty:
            logger.warning(f"No data found for {symbol} on {date_str}")
            return pd.DataFrame()
        
        # Create feature dataframe
        features = pd.DataFrame()
        features['timestamp'] = raw_data['timestamp']
        
        # Add basic price features
        if 'mid_price' in raw_data.columns:
            features['price'] = raw_data['mid_price']
        elif 'avg_price' in raw_data.columns:
            features['price'] = raw_data['avg_price']
        else:
            logger.warning("No price column found in data")
            return pd.DataFrame()
            
        # Add basic volume features
        if 'volume' in raw_data.columns:
            features['volume'] = raw_data['volume']
        
        # Add order book imbalance if available
        if 'vol_imbalance' in raw_data.columns:
            features['ob_imbalance'] = raw_data['vol_imbalance']
        
        # Add spread if available
        if 'spread' in raw_data.columns:
            features['spread'] = raw_data['spread']
            features['rel_spread'] = raw_data['spread'] / raw_data['mid_price']
        
        # Calculate price-based features for different window sizes
        for window in window_sizes:
            window_str = f'{window}'
            
            # Calculate returns
            features[f'return_{window_str}'] = features['price'].pct_change(window).fillna(0)
            
            # Calculate volatility
            features[f'volatility_{window_str}'] = features['price'].rolling(window).std().fillna(0)
            features[f'rel_volatility_{window_str}'] = features[f'volatility_{window_str}'] / features['price']
            
            # Calculate price acceleration
            features[f'price_accel_{window_str}'] = features[f'return_{window_str}'].diff(window).fillna(0)
            
            # Calculate volume features
            if 'volume' in features.columns:
                features[f'volume_ma_{window_str}'] = features['volume'].rolling(window).mean().fillna(0)
                features[f'rel_volume_{window_str}'] = features['volume'] / features[f'volume_ma_{window_str}'].replace(0, np.nan).fillna(1)
            
            # Calculate spread features
            if 'spread' in features.columns:
                features[f'spread_ma_{window_str}'] = features['spread'].rolling(window).mean().fillna(0)
                features[f'rel_spread_{window_str}'] = features['spread'] / features[f'spread_ma_{window_str}'].replace(0, np.nan).fillna(1)
            
            # Order book imbalance features
            if 'ob_imbalance' in features.columns:
                features[f'ob_imbalance_ma_{window_str}'] = features['ob_imbalance'].rolling(window).mean().fillna(0)
                features[f'ob_imbalance_std_{window_str}'] = features['ob_imbalance'].rolling(window).std().fillna(0)
        
        # Add trade-specific features if available
        if 'buy_ratio' in raw_data.columns:
            features['buy_ratio'] = raw_data['buy_ratio']
            
            for window in window_sizes:
                window_str = f'{window}'
                features[f'buy_ratio_ma_{window_str}'] = features['buy_ratio'].rolling(window).mean().fillna(0.5)
        
        # Drop NaN values
        features = features.fillna(0)
        
        logger.info(f"Extracted {len(features.columns)} features for {symbol}")
        
        return features
    
    def extract_multi_day_features(self, symbol, start_date, end_date, resample_interval='5min', window_sizes=[5, 15, 30]):
        """
        Extract features for multiple days and concatenate them.
        
        Args:
            symbol (str): Trading symbol
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            resample_interval (str): Interval for resampling
            window_sizes (list): Window sizes for feature calculation
            
        Returns:
            pd.DataFrame: Combined features for all days
        """
        # Get all available dates
        all_dates = self.data_loader.list_available_dates()
        
        # Filter dates in range
        dates_in_range = [date for date in all_dates if start_date <= date <= end_date]
        
        if not dates_in_range:
            logger.warning(f"No data available between {start_date} and {end_date}")
            return pd.DataFrame()
        
        # Extract features for each day
        all_features = []
        for date_str in dates_in_range:
            logger.info(f"Processing date {date_str}")
            daily_features = self.extract_features(symbol, date_str, resample_interval, window_sizes)
            if not daily_features.empty:
                all_features.append(daily_features)
        
        if not all_features:
            logger.warning("No features extracted for any date")
            return pd.DataFrame()
        
        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        
        return combined_features
