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
        elif 'Price' in raw_data.columns:
            features['price'] = raw_data['Price']
        else:
            logger.warning("No price column found in data")
            return pd.DataFrame()
            
        # Add basic volume features
        if 'volume' in raw_data.columns:
            features['volume'] = raw_data['volume'] + 1  # Add small offset to avoid zeros
        elif 'Quantity' in raw_data.columns:
            features['volume'] = raw_data['Quantity'] + 1
        
        # Add order book imbalance if available
        if 'vol_imbalance' in raw_data.columns:
            features['ob_imbalance'] = raw_data['vol_imbalance']
        elif all(col in raw_data.columns for col in ['BidQtyL1', 'AskQtyL1']):
            # Calculate order book imbalance if not present but bid/ask data is available
            features['ob_imbalance'] = (raw_data['BidQtyL1'] - raw_data['AskQtyL1']) / (raw_data['BidQtyL1'] + raw_data['AskQtyL1'] + 1e-10)
        
        # Add spread if available
        if 'spread' in raw_data.columns:
            features['spread'] = raw_data['spread']
            features['rel_spread'] = raw_data['spread'] / (raw_data['mid_price'] + 1e-10)
        elif all(col in raw_data.columns for col in ['BidPriceL1', 'AskPriceL1']):
            # Calculate spread if not present but L1 data is available
            features['spread'] = raw_data['AskPriceL1'] - raw_data['BidPriceL1']
            features['rel_spread'] = features['spread'] / (features['price'] + 1e-10)
        
        # Calculate depth features if order book data is available
        depth_levels = min(5, 20)  # Use up to 5 levels, or whatever is available up to 20
        if all(f'BidQtyL{i}' in raw_data.columns for i in range(1, depth_levels+1)):
            # Calculate cumulative depth
            bid_depths = [f'BidQtyL{i}' for i in range(1, depth_levels+1)]
            ask_depths = [f'AskQtyL{i}' for i in range(1, depth_levels+1)]
            
            features['cum_bid_qty'] = raw_data[bid_depths].sum(axis=1)
            features['cum_ask_qty'] = raw_data[ask_depths].sum(axis=1)
            features['depth_imbalance'] = (features['cum_bid_qty'] - features['cum_ask_qty']) / (features['cum_bid_qty'] + features['cum_ask_qty'] + 1e-10)
            
            # Calculate microprice if available
            if all(col in raw_data.columns for col in ['BidPriceL1', 'AskPriceL1', 'BidQtyL1', 'AskQtyL1']):
                bid_qty = raw_data['BidQtyL1']
                ask_qty = raw_data['AskQtyL1']
                bid_price = raw_data['BidPriceL1']
                ask_price = raw_data['AskPriceL1']
                
                features['microprice'] = (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty + 1e-10)
                features['microprice_diff'] = (features['microprice'] - features['price']) / (features['price'] + 1e-10)
        
        # Calculate price-based features for different window sizes
        for window in window_sizes:
            window_str = f'{window}'
            
            # Calculate returns (add small noise to ensure variance)
            features[f'return_{window_str}'] = features['price'].pct_change(window).fillna(0)
            
            # Calculate volatility (ensure positive values)
            features[f'volatility_{window_str}'] = features['price'].rolling(window).std().fillna(0) + 1e-6
            features[f'rel_volatility_{window_str}'] = features[f'volatility_{window_str}'] / (features['price'] + 1e-10)
            
            # Calculate momentum
            features[f'momentum_{window_str}'] = (features['price'] / features['price'].shift(window) - 1).fillna(0)
            
            # Z-score of price
            rolling_mean = features['price'].rolling(window).mean()
            rolling_std = features['price'].rolling(window).std()
            features[f'price_zscore_{window_str}'] = ((features['price'] - rolling_mean) / (rolling_std + 1e-10)).fillna(0)
            
            # Calculate price acceleration
            features[f'price_accel_{window_str}'] = features[f'return_{window_str}'].diff(window).fillna(0)
            
            # Calculate volume features
            if 'volume' in features.columns:
                features[f'volume_ma_{window_str}'] = features['volume'].rolling(window).mean().fillna(1)  # Avoid division by zero
                features[f'rel_volume_{window_str}'] = features['volume'] / features[f'volume_ma_{window_str}']
                
                # Calculate volume volatility
                features[f'volume_vol_{window_str}'] = features['volume'].rolling(window).std().fillna(0) / (features[f'volume_ma_{window_str}'] + 1e-10)
            
            # Calculate spread features
            if 'spread' in features.columns:
                features[f'spread_ma_{window_str}'] = features['spread'].rolling(window).mean().fillna(features['spread'].mean())
                features[f'rel_spread_{window_str}'] = features['spread'] / (features[f'spread_ma_{window_str}'] + 1e-10)
                
                # Spread volatility
                features[f'spread_vol_{window_str}'] = features['spread'].rolling(window).std().fillna(0)
            
            # Order book imbalance features
            if 'ob_imbalance' in features.columns:
                features[f'ob_imbalance_ma_{window_str}'] = features['ob_imbalance'].rolling(window).mean().fillna(0)
                features[f'ob_imbalance_std_{window_str}'] = features['ob_imbalance'].rolling(window).std().fillna(0.01)  # Add small value to ensure variance
        
        # Add autocorrelation features
        for lag in [1, 5, 10]:
            features[f'autocorr_{lag}'] = features['price'].pct_change().fillna(0).rolling(window=10).corr(features['price'].pct_change().fillna(0).shift(lag))
        
        # Add trade-specific features if available
        if 'buy_ratio' in raw_data.columns:
            features['buy_ratio'] = raw_data['buy_ratio']
            
            for window in window_sizes:
                window_str = f'{window}'
                features[f'buy_ratio_ma_{window_str}'] = features['buy_ratio'].rolling(window).mean().fillna(0.5)
                features[f'buy_ratio_std_{window_str}'] = features['buy_ratio'].rolling(window).std().fillna(0.01)
        
        # Trend strength indicators
        for window in window_sizes:
            # ADX-like trend strength 
            up_move = features['price'].diff(1).fillna(0)
            down_move = -features['price'].diff(1).fillna(0)
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            features[f'trend_strength_{window}'] = pd.Series(plus_dm).rolling(window).sum().fillna(0) - pd.Series(minus_dm).rolling(window).sum().fillna(0)
            features[f'trend_strength_{window}'] = features[f'trend_strength_{window}'] / (features['price'] + 1e-10)
        
        # Advanced volatility measures
        if len(features) > 15:  # Need enough data points
            # Parkinson volatility estimator (uses high-low range)
            price_max = features['price'].rolling(10).max()
            price_min = features['price'].rolling(10).min()
            features['parkinson_vol'] = np.log(price_max / price_min) / (4 * np.log(2))
            
            # Directional volatility (separate upside and downside)
            returns = features['price'].pct_change().fillna(0)
            features['upside_vol'] = returns.rolling(10).apply(lambda x: np.sqrt(np.mean(np.square(x[x > 0])))).fillna(0)
            features['downside_vol'] = returns.rolling(10).apply(lambda x: np.sqrt(np.mean(np.square(x[x < 0])))).fillna(0)
        
        # Add some noise to all features to ensure variance
        for col in features.columns:
            if col != 'timestamp' and features[col].nunique() <= 1:
                logger.warning(f"Feature {col} has zero variance. Adding small noise.")
                features[col] = features[col] + np.random.normal(0, 0.0001, len(features))
        
        # Replace inf and -inf
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column means instead of zeros
        for col in features.columns:
            if col != 'timestamp':
                if features[col].isna().any():
                    mean_val = features[col].mean()
                    if pd.isna(mean_val):
                        mean_val = 0
                    features[col] = features[col].fillna(mean_val)
        
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
        combined_features = pd.concat(all_features, ignore_index=False)
        
        # Set timestamp as index if it exists
        if 'timestamp' in combined_features.columns:
            combined_features.set_index('timestamp', inplace=True)
        
        # Check for and report low-variance features
        feature_std = combined_features.std()
        low_var_features = feature_std[feature_std < 1e-5].index.tolist()
        if low_var_features:
            logger.warning(f"Found {len(low_var_features)} features with low variance: {low_var_features[:5]}")
        
        # Log overall feature statistics
        logger.info(f"Final feature set: {combined_features.shape[0]} samples, {combined_features.shape[1]} features")
        
        return combined_features
