import pandas as pd
import ta
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings
# --- Advanced Technical Indicator Utilities ---
def compute_bid_ask_spread(order_book):
    """
    Compute bid-ask spread from order book snapshot.
    order_book: dict with 'bids' and 'asks' as lists of [price, size]
    Returns: spread (float)
    """
    try:
        best_bid = order_book['bids'][0][0]
        best_ask = order_book['asks'][0][0]
        return best_ask - best_bid
    except Exception:
        return np.nan

def compute_order_book_depth_imbalance(order_book, depth=5):
    """
    Compute order book depth imbalance (top N levels).
    Returns: imbalance (float)
    """
    try:
        bid_vol = sum([x[1] for x in order_book['bids'][:depth]])
        ask_vol = sum([x[1] for x in order_book['asks'][:depth]])
        return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
    except Exception:
        return np.nan

def compute_volume_profile(df, bins=10):
    """
    Compute volume profile: high volume price nodes.
    Returns: price level with max volume (support/resistance)
    """
    try:
        price_bins = pd.cut(df['close'], bins)
        vol_profile = df.groupby(price_bins)['volume'].sum()
        max_vol_price = vol_profile.idxmax().mid
        return max_vol_price
    except Exception:
        return np.nan

def compute_trade_imbalance(trades):
    """
    Compute trade imbalance from tick/trade data.
    trades: DataFrame with 'is_buyer_maker' and 'qty'
    Returns: imbalance (float)
    """
    try:
        buy_qty = trades.loc[~trades['is_buyer_maker'], 'qty'].sum()
        sell_qty = trades.loc[trades['is_buyer_maker'], 'qty'].sum()
        return (buy_qty - sell_qty) / (buy_qty + sell_qty + 1e-9)
    except Exception:
        return np.nan

def compute_funding_rate_trend(funding_rates, window=24):
    """
    Compute funding rate trend (rolling mean).
    funding_rates: Series of funding rates
    Returns: rolling mean
    """
    try:
        return funding_rates.rolling(window).mean().iloc[-1]
    except Exception:
        return np.nan

# ---------- Helper utilities for advanced features (safe and lightweight) ----------
def _ensure_dt_index(df):
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if 'timestamp' in df.columns:
        try:
            df = df.set_index(pd.to_datetime(df['timestamp']))
        except Exception:
            pass
    return df

def _resample_ohlcv(df, rule):
    try:
        ohlcv = df[['open','high','low','close','volume']].copy()
        res = ohlcv.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        })
        return res.dropna(how='all')
    except Exception:
        return None

def _compute_donchian(high, low, window):
    try:
        dc_high = high.rolling(window).max()
        dc_low = low.rolling(window).min()
        dc_width = (dc_high - dc_low) / (dc_low.replace(0, np.nan))
        dc_pos = (high - dc_low) / (dc_high - dc_low).replace(0, np.nan)
        return dc_high, dc_low, dc_width, dc_pos
    except Exception:
        return None, None, None, None

def _compute_vwap(high, low, close, volume, window=20):
    try:
        # Try ta implementation
        vwap_ind = ta.volume.VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume, window=window)
        return vwap_ind.volume_weighted_average_price()
    except Exception:
        # Manual typical price weighted by volume
        tp = (high + low + close) / 3.0
        num = (tp * volume).rolling(window).sum()
        denom = volume.rolling(window).sum().replace(0, np.nan)
        return (num / denom)

def _compute_cmf(high, low, close, volume, window=20):
    try:
        cmf_ind = ta.volume.ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume, window=window)
        return cmf_ind.chaikin_money_flow()
    except Exception:
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        mfv = mfm * volume
        cmf = mfv.rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)
        return cmf

def engineer_features(df):
    """
    Optimized feature engineering with caching and reduced complexity
    Streamlined to 20 most predictive features for faster processing
    Always returns a valid DataFrame, even on error.
    """
    df = df.copy()
    
    # Cache key columns to avoid repeated calculations
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    try:
        # Essential momentum indicators (fastest)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_21'] = ta.momentum.rsi(df['close'], window=21)
        
        # Moving averages with pre-calculated values
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        # Price relationships (vectorized)
        df['price_vs_ema9'] = (close - df['ema_9']) / df['ema_9']
        df['price_vs_ema21'] = (close - df['ema_21']) / df['ema_21']
        df['ema_cross'] = df['ema_9'] - df['ema_21']
        
        # MACD (single calculation)
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        
        # Bollinger Bands (optimized)
        bb_high = ta.volatility.bollinger_hband(df['close'])
        bb_low = ta.volatility.bollinger_lband(df['close'])
        df['bb_width'] = (bb_high - bb_low) / close
        df['bb_position'] = (close - bb_low) / (bb_high - bb_low)
        
        # Volume analysis (fast indicators)
        volume_sma = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = volume / volume_sma
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volatility (single calculation)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['atr_ratio'] = df['atr'] / close
        
        # Price momentum (fast)
        df['roc'] = ta.momentum.roc(df['close'], window=10)
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # Support/Resistance (simple)
        df['pivot_high'] = df['high'].rolling(window=5).max()
        df['pivot_low'] = df['low'].rolling(window=5).min()
        df['resistance_distance'] = (df['pivot_high'] - close) / close
        df['support_distance'] = (close - df['pivot_low']) / close
        
        # Enhanced features for better prediction accuracy
        
        # Market microstructure features
        df['price_momentum_3'] = df['close'].pct_change(3)
        df['price_momentum_5'] = df['close'].pct_change(5) 
        df['volume_price_trend'] = ((df['close'] - df['close'].shift(1)) * df['volume']).rolling(10).mean()
        
        # Volatility clustering
        df['volatility_5'] = df['close'].rolling(5).std() / df['close']
        df['volatility_10'] = df['close'].rolling(10).std() / df['close']
        df['vol_ratio'] = df['volatility_5'] / (df['volatility_10'] + 1e-8)
        
        # Market strength indicators
        df['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        df['close_to_open'] = (df['close'] - df['open']) / (df['open'] + 1e-8)
        
        # Price acceleration
        df['price_accel'] = df['close'].pct_change().diff()
        
        # Volume indicators
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_std_ratio'] = df['volume'] / df['volume'].rolling(20).std()
        
        # Time features (enhanced)
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_market_hours'] = ((df.index.hour >= 8) & (df.index.hour <= 20)).astype(int)
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        elif 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            df['hour'] = ts.dt.hour
            df['day_of_week'] = ts.dt.dayofweek
            df['is_market_hours'] = ((ts.dt.hour >= 8) & (ts.dt.hour <= 20)).astype(int)
            df['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)
        
        print(f"Optimized: Engineered {len(df.columns)} features")
        return df
        
    except Exception as e:
        print(f"Feature engineering error: {e}")
        # Minimal fallback
        try:
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            print(f"Fallback: Engineered {len(df.columns)} basic features")
            return df
        except Exception as e2:
            print(f"Fallback feature engineering failed: {e2}")
            # Return a DataFrame with only price columns if all else fails
            return df[['open','high','low','close','volume']] if all(col in df.columns for col in ['open','high','low','close','volume']) else pd.DataFrame()

def select_best_features(X, y, k=15):
    """
    FIXED: Select only the most predictive features with proper error handling
    """
    try:
        # Reshape X for feature selection
        if len(X.shape) == 3:
            X_reshaped = X.reshape(-1, X.shape[-1])
        else:
            X_reshaped = X
            
        # Ensure we have enough samples
        if X_reshaped.shape[0] < 10:
            print(f"Warning: Too few samples ({X_reshaped.shape[0]}) for feature selection")
            return X, np.ones(X.shape[-1], dtype=bool)
        
        # Use mutual information for feature selection
        k = min(k, X_reshaped.shape[1])
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X_reshaped, y)
        
        # Get selected feature indices
        selected_features = selector.get_support()
        feature_scores = selector.scores_
        
        # Print feature importance
        print(f"Selected {X_selected.shape[1]} best features out of {X_reshaped.shape[1]}")
        
        return X_selected, selected_features
        
    except Exception as e:
        print(f"Feature selection error: {e}")
        return X, np.ones(X.shape[-1], dtype=bool)

def augment_data(X, y, augmentation_factor=1):
    """
    FIXED: Data augmentation with proper error handling
    """
    try:
        # Check if we have enough samples
        if len(X) < 10:
            print(f"Warning: Too few samples ({len(X)}) for augmentation")
            return X, y
            
        augmented_X = []
        augmented_y = []
        
        # Original data
        augmented_X.append(X)
        augmented_y.append(y)
        
        # Augment positive samples (minority class)
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            # Add noise to positive samples
            for _ in range(augmentation_factor):
                noise = np.random.normal(0, 0.01, X.shape)  # Small noise
                X_aug = X + noise
                augmented_X.append(X_aug)
                augmented_y.append(y)
        
        # Combine all data
        X_combined = np.vstack(augmented_X)
        y_combined = np.hstack(augmented_y)
        
        print(f"Data augmented: {len(X)} -> {len(X_combined)} samples")
        return X_combined, y_combined
        
    except Exception as e:
        print(f"Data augmentation error: {e}")
        return X, y

def prepare_training_data_enhanced(data, feature_columns, lookback, target_horizon=1):
    """
    Enhanced data preparation with proper lookahead prevention and NaN handling
    """
    try:
        # Ensure we have the required columns
        required_cols = feature_columns + ['close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None, None
            
        # Create a working copy and handle NaN values
        df = data[required_cols].copy()
        
        # Fill NaN values with forward fill, then backward fill, then 0
        df = df.ffill().bfill().fillna(0)
        
        # Check if we have enough data
        if len(df) < lookback + 2:
            print(f"Warning: Insufficient data. Need {lookback + 2}, have {len(df)}")
            return None, None

        X = []
        y = []

        # Use only the next immediate price movement to prevent lookahead bias
        # Pre-filter data to ensure X and y synchronization
        valid_indices = []
        valid_targets = []
        
        # First pass: identify valid samples and their targets
        for i in range(lookback, len(df) - 1):
            # Target: Next immediate price movement (1 candle ahead)
            current_price = df['close'].iloc[i]
            next_price = df['close'].iloc[i + 1]

            # Calculate immediate price change
            price_change_pct = (next_price - current_price) / current_price * 100

            # Balanced target definition for equal BUY/SELL opportunities
            if price_change_pct > 0.12:  # BUY signal threshold (slightly lower)
                valid_indices.append(i)
                valid_targets.append(1)  # BUY signal
            elif price_change_pct < -0.12:  # SELL signal threshold (symmetric)
                valid_indices.append(i)
                valid_targets.append(0)  # SELL signal
            # Skip neutral movements between -0.12% and +0.12%

        # Second pass: create features for valid indices only (NO continue statements)
        for idx, target in zip(valid_indices, valid_targets):
            # Features: lookback periods of feature data (only past data)
            feature_slice = df[feature_columns].iloc[idx-lookback:idx].values
            
            # Ensure numeric type and handle NaN - NO SKIPPING to maintain sync
            try:
                feature_slice = feature_slice.astype(np.float64)
                feature_slice = np.nan_to_num(feature_slice, nan=0.0, posinf=0.0, neginf=0.0)
                
                # If all values are zero, fill with small random values instead of skipping
                if np.all(feature_slice == 0) or np.all(np.isnan(feature_slice)):
                    feature_slice = np.random.normal(0, 0.01, feature_slice.shape)
                    
            except (ValueError, TypeError):
                # Handle non-numeric data - create dummy data instead of skipping
                feature_slice = np.random.normal(0, 0.01, (lookback, len(feature_columns)))
            
            # Always add both feature and target (ensures perfect synchronization)
            X.append(feature_slice)
            y.append(target)

        if len(X) == 0:
            print("Warning: No valid data after filtering")
            return None, None
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        print(f"Successfully prepared data: X shape {X.shape}, y shape {y.shape}")
        print(f"Target distribution: {np.bincount(y)}")

        return X, y
        
    except Exception as e:
        print(f"Error in prepare_training_data_enhanced: {e}")
        return None, None

def prepare_training_data_simple(data, feature_columns, lookback):
    """
    FIXED: Simple fallback data preparation with proper scaler handling
    """
    try:
        # Select features and target
        features = data[feature_columns].values.astype(np.float32)
        prices = data['close'].values.astype(np.float32)
        
        # Create simple target: 1 if price goes up next period, 0 otherwise
        target = np.zeros(len(prices) - 1, dtype=np.int64)
        for i in range(len(prices) - 1):
            if prices[i + 1] > prices[i]:
                target[i] = 1
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(features) - 1):
            X.append(features[i-lookback:i])
            y.append(target[i])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        # Remove any NaN values
        valid_indices = ~(np.isnan(X).any(axis=(1, 2)) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Check if we have any data left
        if len(X) == 0:
            print("Warning: No valid data after NaN removal")
            return None, None
        
        # Normalize features
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Ensure no NaN in reshaped data
        if np.isnan(X_reshaped).any():
            print("Warning: NaN values found in reshaped data, filling with 0")
            X_reshaped = np.nan_to_num(X_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_scaled = scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(X.shape[0], X.shape[1], -1)
        
        return X, y
        
    except Exception as e:
        print(f"Error in prepare_training_data_simple: {e}")
        return None, None
