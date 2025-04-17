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
        Load order book depth data for a specific symbol and date.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BNBFDUSD')
            date_str (str): Date string in format YYYYMMDD
            
        Returns:
            pd.DataFrame: Processed order book data
        """
        filename = f"{symbol}_{date_str}.txt"
        filepath = os.path.join(self.depth_dir, filename)
        
        logger.info(f"Loading order book data from {filepath}")
        
        try:
            # First, check the file format by reading first few lines
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                first_row = next(reader, None)
            
            # Detect if file has headers
            has_header = False
            timestamp_col_idx = 0  # Default timestamp column index
            
            # Check if header contains typical column names
            if header:
                header_str = ' '.join([str(h).lower() for h in header if h])
                has_header = 'time' in header_str or 'timestamp' in header_str
            
            # Using chunking to handle large files
            chunks = []
            # Skip header row if it exists
            skip_rows = 1 if has_header else 0
            
            for chunk in pd.read_csv(filepath, chunksize=100000, delimiter=',', header=None, skiprows=skip_rows):
                chunks.append(chunk)
            
            df = pd.concat(chunks)
            
            # Generate column names
            if has_header:
                # Use the header we detected
                df.columns = header
            else:
                # Generate standard column names
                columns = ['timestamp']
                
                # Add bid price and quantity columns
                for i in range(20):
                    columns.extend([f'bid_price_{i}', f'bid_qty_{i}'])
                    
                # Add ask price and quantity columns
                for i in range(20):
                    columns.extend([f'ask_price_{i}', f'ask_qty_{i}'])
                
                # Check if column count matches
                if len(df.columns) >= len(columns):
                    # Use our standard column names for the expected columns
                    rename_dict = {i: columns[i] for i in range(len(columns))}
                    df = df.rename(columns=rename_dict)
                else:
                    logger.warning(f"Column count mismatch: expected {len(columns)}, got {len(df.columns)}")
                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
                    # Assume first column is timestamp
                    if len(df.columns) > 0:
                        df = df.rename(columns={'col_0': 'timestamp'})
            
            # Find timestamp column
            timestamp_col = None
            for col in df.columns:
                col_name = str(col).lower()
                if 'time' in col_name or 'timestamp' in col_name:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                # If no column has 'time' in name, use first column
                timestamp_col = df.columns[0]
                df = df.rename(columns={timestamp_col: 'timestamp'})
                timestamp_col = 'timestamp'
            
            # Convert timestamp to datetime
            try:
                # Try different timestamp formats based on the first few values
                sample = df[timestamp_col].iloc[:5].astype(str)
                
                # Check if timestamps are numeric (epoch)
                if all(s.isdigit() for s in sample):
                    # Convert from milliseconds to datetime
                    df['timestamp'] = pd.to_datetime(df[timestamp_col].astype(float), unit='ms')
                else:
                    # Try common timestamp formats
                    for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y%m%d %H:%M:%S.%f', '%Y%m%d %H:%M:%S']:
                        try:
                            df['timestamp'] = pd.to_datetime(df[timestamp_col], format=fmt, errors='raise')
                            break
                        except:
                            continue
                    
                    # If specific formats fail, try the generic parser with errors='coerce'
                    if pd.isna(df['timestamp']).all():
                        df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
                
                # Check if conversion was successful
                if pd.isna(df['timestamp']).all():
                    # Last resort - try to parse strings from epoch
                    try:
                        df['timestamp'] = pd.to_datetime(df[timestamp_col].astype(float) / 1000, unit='s')
                    except:
                        pass
                
                if pd.isna(df['timestamp']).all():
                    raise ValueError("Could not convert timestamp column to datetime")
                
                # Drop original timestamp column if it's not named 'timestamp'
                if timestamp_col != 'timestamp':
                    df = df.drop(columns=[timestamp_col])
                
            except Exception as e:
                logger.error(f"Error converting timestamps: {e}")
                raise
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading depth data: {e}")
            raise
    
    def load_trade_data(self, symbol, date_str):
        """
        Load aggregated trade data for a specific symbol and date.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BNBFDUSD')
            date_str (str): Date string in format YYYYMMDD
            
        Returns:
            pd.DataFrame: Processed trade data
        """
        filename = f"{symbol}_{date_str}.txt"
        filepath = os.path.join(self.trade_dir, filename)
        
        logger.info(f"Loading trade data from {filepath}")
        
        try:
            # First, check the file format by reading first few lines
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                first_row = next(reader, None)
            
            # Detect if file has headers
            has_header = False
            timestamp_col_idx = 0  # Default timestamp column index
            
            # Check if header contains typical column names
            if header:
                header_str = ' '.join([str(h).lower() for h in header if h])
                has_header = 'time' in header_str or 'timestamp' in header_str or 'price' in header_str
            
            # Using chunking to handle large files
            chunks = []
            # Skip header row if it exists
            skip_rows = 1 if has_header else 0
            
            for chunk in pd.read_csv(filepath, chunksize=100000, delimiter=',', header=None, skiprows=skip_rows):
                chunks.append(chunk)
            
            df = pd.concat(chunks)
            
            # Generate column names
            if has_header:
                # Use the header we detected
                df.columns = header
            else:
                # Use standard column names for trade data
                expected_cols = ['timestamp', 'price', 'quantity', 'is_buyer_maker']
                
                # Check if column count matches
                if len(df.columns) >= len(expected_cols):
                    # Use our standard column names for the expected columns
                    rename_dict = {i: expected_cols[i] for i in range(len(expected_cols))}
                    df = df.rename(columns=rename_dict)
                else:
                    logger.warning(f"Column count mismatch: expected {len(expected_cols)}, got {len(df.columns)}")
                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
                    # Assume first column is timestamp
                    if len(df.columns) > 0:
                        df = df.rename(columns={'col_0': 'timestamp'})
            
            # Find timestamp column
            timestamp_col = None
            for col in df.columns:
                col_name = str(col).lower()
                if 'time' in col_name or 'timestamp' in col_name:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                # If no column has 'time' in name, use first column
                timestamp_col = df.columns[0]
                df = df.rename(columns={timestamp_col: 'timestamp'})
                timestamp_col = 'timestamp'
            
            # Convert timestamp to datetime
            try:
                # Try different timestamp formats based on the first few values
                sample = df[timestamp_col].iloc[:5].astype(str)
                
                # Check if timestamps are numeric (epoch)
                if all(s.isdigit() for s in sample):
                    # Convert from milliseconds to datetime
                    df['timestamp'] = pd.to_datetime(df[timestamp_col].astype(float), unit='ms')
                else:
                    # Try common timestamp formats
                    for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y%m%d %H:%M:%S.%f', '%Y%m%d %H:%M:%S']:
                        try:
                            df['timestamp'] = pd.to_datetime(df[timestamp_col], format=fmt, errors='raise')
                            break
                        except:
                            continue
                    
                    # If specific formats fail, try the generic parser with errors='coerce'
                    if pd.isna(df['timestamp']).all():
                        df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
                
                # Check if conversion was successful
                if pd.isna(df['timestamp']).all():
                    # Last resort - try to parse strings from epoch
                    try:
                        df['timestamp'] = pd.to_datetime(df[timestamp_col].astype(float) / 1000, unit='s')
                    except:
                        pass
                
                if pd.isna(df['timestamp']).all():
                    raise ValueError("Could not convert timestamp column to datetime")
                
                # Drop original timestamp column if it's not named 'timestamp'
                if timestamp_col != 'timestamp':
                    df = df.drop(columns=[timestamp_col])
                
            except Exception as e:
                logger.error(f"Error converting trade timestamps: {e}")
                raise
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading trade data: {e}")
            raise
    
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
        depth_resampled = depth_features.set_index('timestamp').resample(resample_interval).last().fillna(method='ffill')
        trade_resampled = trade_features.set_index('timestamp').resample(resample_interval).agg({
            'volume': 'sum',
            'trade_count': 'sum',
            'avg_price': 'mean',
            'buy_ratio': 'mean',
            'price_volatility': 'max'
        })
        
        # Combine datasets
        combined = pd.concat([depth_resampled, trade_resampled], axis=1)
        combined = combined.fillna(method='ffill').reset_index()
        
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