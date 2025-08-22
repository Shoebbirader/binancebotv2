import os
import json
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
import threading
import logging

load_dotenv('config/secrets.env')

class DataCollector:
    def __init__(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
        self.symbols = config['symbols']
        self.interval = config['interval']
        self.client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
        self._lock = threading.Lock()
        self._cache = {}
        self._cache_max_size = 100  # Limit cache to 100 entries

    def fetch_historical(self, symbol, limit=1000):
        """Fetch historical data with caching and optimized processing"""
        try:
            limit = min(limit, 1000)
            
            # Simple caching check (in-memory)
            cache_key = f"{symbol}_{self.interval}_{limit}"
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            with self._lock:
                klines = self.client.futures_klines(symbol=symbol, interval=self.interval, limit=limit)
            
            if not klines or len(klines) == 0:
                logging.warning(f"No data for {symbol}, using fallback")
                return self._create_fallback_data(limit)
                
            # Fast DataFrame creation
            df = pd.DataFrame(klines, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_asset_volume","num_trades",
                "taker_buy_base","taker_buy_quote","ignore"
            ])
            
            # Vectorized type conversion
            numeric_cols = ["open","high","low","close","volume"]
            df[numeric_cols] = df[numeric_cols].astype(float)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            
            # Cache the result with size limit
            if len(self._cache) >= self._cache_max_size:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = df
            
            logging.info(f"Fetched {len(df)} points for {symbol}")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching {symbol}: {e}")
            return self._create_fallback_data(limit)
    
    def _create_fallback_data(self, limit):
        """Fast fallback data creation"""
        return pd.DataFrame({
            'open_time': pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='5T'),
            'open': [100.0]*limit,
            'high': [101.0]*limit,
            'low': [99.0]*limit,
            'close': [100.5]*limit,
            'volume': [10.0]*limit
        }).set_index('open_time')

    def fetch_extended_history(self, symbol, target_samples=3000):
        """
        Fetch extended history by making multiple API calls if needed.
        """
        try:
            # Start with 1000 samples
            df = self.fetch_historical(symbol, limit=1000)
            
            if df is None:
                return None
                
            # If we need more data, make additional calls
            if len(df) < target_samples:
                print(f"Fetching additional data for {symbol}...")
                
                # Get the earliest timestamp from current data
                earliest_time = df.index[0]
                
                # Fetch more data before this timestamp
                with self._lock:
                    additional_klines = self.client.futures_klines(
                        symbol=symbol, 
                        interval=self.interval, 
                        limit=1000,
                        endTime=int(earliest_time.timestamp() * 1000)
                    )
                
                if additional_klines:
                    # Convert additional data
                    additional_df = pd.DataFrame(additional_klines, columns=[
                        "open_time","open","high","low","close","volume",
                        "close_time","quote_asset_volume","num_trades",
                        "taker_buy_base","taker_buy_quote","ignore"
                    ])
                    
                    numeric_cols = ["open","high","low","close","volume"]
                    additional_df[numeric_cols] = additional_df[numeric_cols].astype(float)
                    additional_df['open_time'] = pd.to_datetime(additional_df['open_time'], unit='ms')
                    additional_df.set_index('open_time', inplace=True)
                    
                    # Combine dataframes
                    df = pd.concat([additional_df, df])
                    df = df.sort_index()
                    
                    print(f"Extended history: {len(df)} total data points for {symbol}")
                
            return df
            
        except Exception as e:
            print(f"Error fetching extended history for {symbol}: {e}")
            return None
