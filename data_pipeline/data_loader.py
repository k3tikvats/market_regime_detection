"""
Data loader for processing market data files.
Handles both order book data (depth20) and trade data (aggTrade).
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataLoader:
    """
    Loads and processes market data from files.
    Supports both order book data (depth20) and trade data (aggTrade).
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Base directory containing the data folders
        """
        self.data_dir = data_dir
        self.depth_dir = os.path.join(data_dir, 'depth20_1000ms')
        self.trade_dir = os.path.join(data_dir, 'aggTrade')
        
    def load_depth_data(self, symbol, date_str):
        """
        Load order book depth data for a given symbol and date
        
        Args:
            symbol (str): Trading symbol
            date_str (str): Date string in format YYYYMMDD
            
        Returns:
            DataFrame with depth data
        """
        depth_path = os.path.join(self.data_dir, 'depth20_1000ms', f'{symbol}_{date_str}.txt')
        
        if not os.path.exists(depth_path):
            logger.warning(f"No depth data found for {symbol} on {date_str}")
            return pd.DataFrame()
            
        try:
            # Load the file
            logger.info(f"Loading order book data from {depth_path}")
            df = pd.read_csv(depth_path, sep=',')
            
            # Log column names to help with debugging
            logger.info(f"Depth data columns: {df.columns.tolist()}")
            
            # Check if 'Time' column exists and rename it to 'timestamp' for consistency
            if 'Time' in df.columns:
                df = df.rename(columns={'Time': 'timestamp'})
            elif 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})
                
            # If still no timestamp column, create one
            if 'timestamp' not in df.columns:
                logger.warning("No timestamp column found in depth data. Creating sequential timestamps.")
                df['timestamp'] = pd.date_range(
                    start=datetime.strptime(date_str, '%Y%m%d'),
                    periods=len(df),
                    freq='1s'
                )
                return df
            
            # Convert timestamps - handle complex timestamp format
            try:
                # Parse the timestamp, handling IST timezone if present
                if isinstance(df['timestamp'].iloc[0], str) and '+0530 IST' in df['timestamp'].iloc[0]:
                    # Custom parsing for the specific format
                    df['timestamp'] = df['timestamp'].apply(
                        lambda x: pd.Timestamp(x.split('+0530')[0].strip())
                    )
                else:
                    # Standard parsing
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                return df
                
            except Exception as e:
                logger.error(f"Error converting timestamps: {e}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading depth data: {e}")
            return pd.DataFrame()
    
    def load_trade_data(self, symbol, date_str):
        """
        Load trade data for a given symbol and date
        
        Args:
            symbol (str): Trading symbol
            date_str (str): Date string in format YYYYMMDD
            
        Returns:
            DataFrame with trade data
        """
        trade_path = os.path.join(self.data_dir, 'aggTrade', f'{symbol}_{date_str}.txt')
        
        if not os.path.exists(trade_path):
            logger.warning(f"No trade data found for {symbol} on {date_str}")
            return pd.DataFrame()
            
        try:
            logger.info(f"Loading trade data from {trade_path}")
            df = pd.read_csv(trade_path, sep=',')
            
            # Log column names to help with debugging
            logger.info(f"Trade data columns: {df.columns.tolist()}")
            
            # Check if we have alternative time column names
            if 'Time' in df.columns:
                df = df.rename(columns={'Time': 'timestamp'})
            elif 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})
            
            # If still no timestamp column, create one
            if 'timestamp' not in df.columns:
                logger.warning("No timestamp column found in trade data. Creating sequential timestamps.")
                df['timestamp'] = pd.date_range(
                    start=datetime.strptime(date_str, '%Y%m%d'),
                    periods=len(df),
                    freq='1s'
                )
                return df
                
            # Convert timestamps
            try:
                # Parse the timestamp, handling IST timezone if present
                if isinstance(df['timestamp'].iloc[0], str) and '+0530 IST' in df['timestamp'].iloc[0]:
                    # Custom parsing for the specific format
                    df['timestamp'] = df['timestamp'].apply(
                        lambda x: pd.Timestamp(x.split('+0530')[0].strip())
                    )
                else:
                    # Standard parsing
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                return df
                
            except Exception as e:
                logger.error(f"Error converting trade timestamps: {e}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading trade data: {e}")
            return pd.DataFrame()
    
    def load_combined_data(self, symbol, date_str, resample_interval='1min'):
        """
        Load and combine both order book and trade data, resampling to a common time interval.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BNBFDUSD')
            date_str (str): Date string in format YYYYMMDD
            resample_interval (str): Interval for resampling time series (e.g., '1min', '5min')
            
        Returns:
            pd.DataFrame: Combined and resampled data
        """
        depth_data = self.load_depth_data(symbol, date_str)
        trade_data = self.load_trade_data(symbol, date_str)
        
        # Process depth data to extract key metrics
        depth_features = self._extract_depth_features(depth_data)
        
        # Process trade data to extract key metrics
        trade_features = self._extract_trade_features(trade_data)
        
        # Resample both to common time interval
        depth_resampled = depth_features.set_index('timestamp').resample(resample_interval).last().ffill()
        trade_resampled = trade_features.set_index('timestamp').resample(resample_interval).agg({
            'volume': 'sum',
            'trade_count': 'sum',
            'avg_price': 'mean',
            'buy_ratio': 'mean',
            'price_volatility': 'max'
        })
        
        # Combine datasets
        combined = pd.concat([depth_resampled, trade_resampled], axis=1)
        combined = combined.ffill().reset_index()
        
        return combined
    
    def _extract_depth_features(self, depth_df):
        """
        Extract features from order book depth data.
        
        Args:
            depth_df (pd.DataFrame): Raw order book data
            
        Returns:
            pd.DataFrame: Dataframe with extracted features
        """
        features = pd.DataFrame()
        features['timestamp'] = depth_df['timestamp']
        
        try:
            # Calculate mid price
            if all(col in depth_df.columns for col in ['bid_price_0', 'ask_price_0']):
                features['mid_price'] = (depth_df['bid_price_0'] + depth_df['ask_price_0']) / 2
                features['spread'] = depth_df['ask_price_0'] - depth_df['bid_price_0']
                
                # Calculate bid and ask volumes (for top 5 levels)
                bid_vol, ask_vol = 0, 0
                for i in range(5):
                    if f'bid_qty_{i}' in depth_df.columns:
                        bid_vol += depth_df[f'bid_qty_{i}']
                    if f'ask_qty_{i}' in depth_df.columns:
                        ask_vol += depth_df[f'ask_qty_{i}']
                
                features['bid_volume'] = bid_vol
                features['ask_volume'] = ask_vol
                features['vol_imbalance'] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
                
            else:
                # If column structure is different, try to adapt
                logger.warning("Expected order book column structure not found, using fallback processing")
                # Simplified processing with generic column names
                numeric_cols = depth_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 2:
                    # Create basic features from available numeric columns
                    features['depth_feature_1'] = depth_df[numeric_cols[1]]
                    features['depth_feature_2'] = depth_df[numeric_cols[2]]
                
        except Exception as e:
            logger.error(f"Error extracting depth features: {e}")
            
        return features
    
    def _extract_trade_features(self, trade_df):
        """
        Extract features from trade data.
        
        Args:
            trade_df (pd.DataFrame): Raw trade data
            
        Returns:
            pd.DataFrame: Dataframe with extracted features
        """
        features = pd.DataFrame()
        features['timestamp'] = trade_df['timestamp']
        
        try:
            if 'quantity' in trade_df.columns:
                features['volume'] = trade_df['quantity']
            elif 'col_2' in trade_df.columns:  # Fallback to generic column
                features['volume'] = trade_df['col_2']
            else:
                features['volume'] = 0
                
            # Trade count (each row is a trade)
            features['trade_count'] = 1
            
            # Average price
            if 'price' in trade_df.columns:
                features['avg_price'] = trade_df['price']
            elif 'col_1' in trade_df.columns:  # Fallback to generic column
                features['avg_price'] = trade_df['col_1']
            else:
                features['avg_price'] = 0
            
            # Buy/sell ratio
            if 'is_buyer_maker' in trade_df.columns:
                features['buy_ratio'] = 1 - trade_df['is_buyer_maker']
            elif 'col_3' in trade_df.columns:
                features['buy_ratio'] = 1 - trade_df['col_3']
            else:
                features['buy_ratio'] = 0.5
                
            # Calculate price volatility (rolling std)
            if 'price' in trade_df.columns:
                features['price_volatility'] = trade_df['price'].rolling(20).std().fillna(0)
            elif 'col_1' in trade_df.columns:
                features['price_volatility'] = trade_df['col_1'].rolling(20).std().fillna(0)
            else:
                features['price_volatility'] = 0
                
        except Exception as e:
            logger.error(f"Error extracting trade features: {e}")
            
        return features
    
    def list_available_dates(self, data_type='both'):
        """
        List available dates in the dataset.
        
        Args:
            data_type (str): 'depth', 'trade', or 'both'
            
        Returns:
            list: Available dates as strings
        """
        dates = set()
        
        if data_type in ['depth', 'both']:
            for file in os.listdir(self.depth_dir):
                if file.endswith('.txt'):
                    parts = file.split('_')
                    if len(parts) >= 2:
                        dates.add(parts[-1].replace('.txt', ''))
        
        if data_type in ['trade', 'both']:
            for file in os.listdir(self.trade_dir):
                if file.endswith('.txt'):
                    parts = file.split('_')
                    if len(parts) >= 2:
                        dates.add(parts[-1].replace('.txt', ''))
        
        return sorted(list(dates))