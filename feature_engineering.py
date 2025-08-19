import pandas as pd
import ta
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings

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
    Enhanced feature engineering with 40+ predictive features
    Fixed to use correct ta library functions
    """
    df = df.copy()
    # Ensure a timestamp column exists for time-based features
    try:
        if 'timestamp' not in df.columns:
            # If index is datetime-like, use it
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
            else:
                # As a fallback, try to parse any existing time columns
                for candidate in ['open_time', 'time', 'date']:
                    if candidate in df.columns:
                        df['timestamp'] = pd.to_datetime(df[candidate])
                        break
                if 'timestamp' not in df.columns:
                    # Last resort: create a monotonic timestamp index
                    df['timestamp'] = pd.to_datetime(pd.RangeIndex(start=0, stop=len(df), step=1), unit='s')
    except Exception:
        pass
    
    try:
        # Price-based features (most predictive)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_21'] = ta.momentum.rsi(df['close'], window=21)
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
        
        # Moving averages with price relationships
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['price_vs_ema9'] = (df['close'] - df['ema_9']) / df['ema_9']
        df['price_vs_ema21'] = (df['close'] - df['ema_21']) / df['ema_21']
        df['ema_cross'] = df['ema_9'] - df['ema_21']
        
        # MACD with signal line
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_histogram'] = ta.trend.macd_diff(df['close'])
        
        # Bollinger Bands
        df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # Volume analysis - FIXED: Use correct volume functions
        df['volume_sma'] = df['volume'].rolling(window=20).mean()  # Manual volume SMA
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['volume_ema'] = ta.trend.ema_indicator(df['volume'], window=20)
        df['volume_std'] = df['volume'].rolling(window=20).std()
        # PVO: try library first, otherwise compute manually
        try:
            pvo_ind = ta.volume.PVOIndicator(volume=df['volume'], window_slow=26, window_fast=12, window_sign=9)
            df['pvo'] = pvo_ind.pvo()
            df['pvo_signal'] = pvo_ind.pvo_signal()
            df['pvo_hist'] = pvo_ind.pvo_hist()
        except Exception:
            vol_ema_fast = df['volume'].ewm(span=12, adjust=False).mean()
            vol_ema_slow = df['volume'].ewm(span=26, adjust=False).mean()
            pvo = (vol_ema_fast - vol_ema_slow) / vol_ema_slow.replace(0, np.nan) * 100.0
            df['pvo'] = pvo.fillna(0)
            df['pvo_signal'] = df['pvo'].ewm(span=9, adjust=False).mean()
            df['pvo_hist'] = df['pvo'] - df['pvo_signal']
        
        # Volatility indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['atr_ratio'] = df['atr'] / df['close']
        df['bb_percent'] = ta.volatility.bollinger_pband(df['close'])
        df['bb_bandwidth'] = ta.volatility.bollinger_wband(df['close'])
        
        # Price momentum
        df['roc'] = ta.momentum.roc(df['close'], window=10)
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        df['stoch_rsi'] = ta.momentum.stochrsi(df['close'])
        df['tsi'] = ta.momentum.tsi(df['close'])
        
        # Support/Resistance levels - FIXED: Use correct pivot functions
        df['pivot_high'] = df['high'].rolling(window=5).max()  # Manual pivot high
        df['pivot_low'] = df['low'].rolling(window=5).min()    # Manual pivot low
        df['resistance_distance'] = (df['pivot_high'] - df['close']) / df['close']
        df['support_distance'] = (df['close'] - df['pivot_low']) / df['close']
        
        # Advanced momentum
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        try:
            aroon = ta.trend.AroonIndicator(high=df['high'], low=df['low'], window=25)
            df['aroon_up'] = aroon.aroon_up()
            df['aroon_down'] = aroon.aroon_down()
        except Exception:
            window = 25
            pos_of_max = df['high'].rolling(window).apply(lambda x: np.argmax(x) + 1, raw=True)
            pos_of_min = df['low'].rolling(window).apply(lambda x: np.argmin(x) + 1, raw=True)
            df['aroon_up'] = (pos_of_max / window) * 100.0
            df['aroon_down'] = (pos_of_min / window) * 100.0
        df['aroon_indicator'] = df['aroon_up'] - df['aroon_down']
        
        # Price patterns
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['inside_bar'] = ((df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))).astype(int)
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Feature interactions
        df['rsi_macd'] = df['rsi'] * df['macd']
        df['volume_price'] = df['volume_ratio'] * df['price_vs_ema9']
        df['momentum_volume'] = df['roc'] * df['volume_ratio']
        
        # Additional technical indicators
        df['kama'] = ta.momentum.kama(df['close'])
        # PPO: try function API first, then class-based fallback if needed
        try:
            df['ppo'] = ta.momentum.ppo(df['close'])
        except Exception:
            try:
                ppo_ind = ta.momentum.PPOIndicator(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
                df['ppo'] = ppo_ind.ppo()
            except Exception:
                pass
        # PVO already computed above via PVOIndicator or manual fallback; avoid non-existent ta.volume.pvo
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        
        # Price action features
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        df['doji'] = (abs(df['close'] - df['open']) < (df['high'] - df['low']) * 0.1).astype(int)

        # -------- Advanced single-timeframe features --------
        # Donchian channels
        dc_h, dc_l, dc_w, dc_pos = _compute_donchian(df['high'], df['low'], window=20)
        if dc_h is not None:
            df['donchian_width_20'] = dc_w
            df['donchian_pos_20'] = dc_pos
        dc_h2, dc_l2, dc_w2, dc_pos2 = _compute_donchian(df['high'], df['low'], window=55)
        if dc_h2 is not None:
            df['donchian_width_55'] = dc_w2
            df['donchian_pos_55'] = dc_pos2

        # SuperTrend (approx): use ATR bands and price relation
        try:
            atr14 = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            ema = ta.trend.ema_indicator(df['close'], window=21)
            mult = 3.0
            upper_band = ema + mult * atr14
            lower_band = ema - mult * atr14
            df['supertrend_upper'] = upper_band
            df['supertrend_lower'] = lower_band
            df['supertrend_trend'] = np.where(df['close'] > ema, 1, -1)
        except Exception:
            pass

        # Keltner Channels and squeeze
        try:
            ema20 = ta.trend.ema_indicator(df['close'], window=20)
            atr20 = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=20)
            kc_mult = 2.0
            kc_upper = ema20 + kc_mult * atr20
            kc_lower = ema20 - kc_mult * atr20
            df['kc_upper'] = kc_upper
            df['kc_lower'] = kc_lower
            # squeeze: BB width vs KC width
            df['squeeze_on'] = ((df['bb_high'] < kc_upper) & (df['bb_low'] > kc_lower)).astype(int)
        except Exception:
            pass

        # VWAP and anchored VWAP deviation (session/day anchor best-effort)
        vwap20 = _compute_vwap(df['high'], df['low'], df['close'], df['volume'], window=20)
        df['vwap20'] = vwap20
        df['price_vs_vwap20'] = (df['close'] - vwap20) / vwap20.replace(0, np.nan)
        try:
            # Daily anchored: groupby date
            ts = pd.to_datetime(df['timestamp'])
            day = ts.dt.floor('D')
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            num = (tp * df['volume']).groupby(day).cumsum()
            denom = df['volume'].groupby(day).cumsum().replace(0, np.nan)
            avwap_day = num / denom
            df['avwap_day'] = avwap_day
            df['price_vs_avwap_day'] = (df['close'] - avwap_day) / avwap_day.replace(0, np.nan)
        except Exception:
            pass

        # Chaikin Money Flow
        df['cmf_20'] = _compute_cmf(df['high'], df['low'], df['close'], df['volume'], window=20)

        # Realized volatility and return z-scores
        ret1 = df['close'].pct_change()
        df['rv_20'] = (ret1.rolling(20).std() * np.sqrt(288))  # 5m ~ 288 bars/day
        df['ret_z_5'] = (ret1 - ret1.rolling(5).mean()) / ret1.rolling(5).std().replace(0, np.nan)
        df['ret_z_20'] = (ret1 - ret1.rolling(20).mean()) / ret1.rolling(20).std().replace(0, np.nan)

        # ADX regime buckets
        try:
            di_pos = ta.trend.plus_di(df['high'], df['low'], df['close'])
            di_neg = ta.trend.minus_di(df['high'], df['low'], df['close'])
            df['adx_regime'] = pd.cut(df['adx'], bins=[-np.inf, 15, 25, np.inf], labels=[0,1,2]).astype(int)
            df['di_spread'] = di_pos - di_neg
        except Exception:
            pass

        # -------- Multi-timeframe features (15m and 1h) --------
        try:
            df_idx = _ensure_dt_index(df)
            df_15m = _resample_ohlcv(df_idx, '15T')
            df_1h = _resample_ohlcv(df_idx, '1H')

            def _suffix_join(base_df, higher_df, suffix):
                if higher_df is None or higher_df.empty:
                    return base_df
                # Compute a few key indicators on higher timeframe
                hi = higher_df
                hi_feat = pd.DataFrame(index=hi.index)
                try:
                    hi_feat[f'rsi{suffix}'] = ta.momentum.rsi(hi['close'], window=14)
                    hi_feat[f'adx{suffix}'] = ta.trend.adx(hi['high'], hi['low'], hi['close'])
                    hi_feat[f'ema21{suffix}'] = ta.trend.ema_indicator(hi['close'], window=21)
                    bb_u = ta.volatility.bollinger_hband(hi['close'])
                    bb_l = ta.volatility.bollinger_lband(hi['close'])
                    hi_feat[f'bb_width{suffix}'] = (bb_u - bb_l) / hi['close']
                except Exception:
                    pass
                # align and forward-fill to base timeframe
                hi_feat = hi_feat.reindex(base_df.index, method='ffill')
                return base_df.join(hi_feat)

            df = _suffix_join(df, df_15m, '_15m')
            df = _suffix_join(df, df_1h, '_1h')
        except Exception:
            pass
        
        print(f"Successfully engineered {len(df.columns)} features")
        
    except Exception as e:
        print(f"Feature engineering error: {e}")
        # Fallback to basic features
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
        print(f"Fallback: Engineered {len(df.columns)} basic features")
    
    return df

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

def prepare_training_data_enhanced(data, feature_columns, lookback, target_horizon=5):
    """
    Enhanced data preparation with better target creation
    """
    data = data.dropna()
    
    X = []
    y = []
    
    for i in range(lookback, len(data) - target_horizon):
        # Features: lookback periods of feature data
        X.append(data[feature_columns].iloc[i-lookback:i].values)
        
        # Enhanced target: Consider price movement magnitude
        current_price = data['close'].iloc[i]
        future_price = data['close'].iloc[i + target_horizon]
        
        # Calculate price change percentage
        price_change_pct = (future_price - current_price) / current_price * 100
        
        # More sophisticated target: 1 for significant upward movement, 0 otherwise
        if price_change_pct > 1.0:  # 1% or more increase
            y.append(1)
        elif price_change_pct < -1.0:  # 1% or more decrease
            y.append(0)
        else:
            # For small movements, use trend direction
            y.append(1 if price_change_pct > 0 else 0)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    return X, y

def prepare_training_data_simple(data, feature_columns, lookback):
    """
    FIXED: Simple fallback data preparation
    """
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
    
    # Normalize features
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(X.shape[0], X.shape[1], -1)
    
    return X, y
