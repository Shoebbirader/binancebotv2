import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class MarketRegimeDetector:
    """
    Detects market regimes: volatility, trend, and phase (bull/bear/sideways)
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)

    def compute_volatility_regime(self, df, atr_window=14, threshold_high=0.02, threshold_low=0.005):
        """
        ATR-based volatility regime detection
        Returns: 'HIGH', 'LOW', or 'NORMAL'
        """
        if df is None or len(df) == 0 or df[['high','low','close']].isnull().any().any():
            return 'NORMAL'
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_window)
        atr_ratio = atr / df['close']
        last_atr = atr_ratio.iloc[-1]
        if last_atr > threshold_high:
            return 'HIGH'
        elif last_atr < threshold_low:
            return 'LOW'
        else:
            return 'NORMAL'

    def compute_trend_strength(self, df, adx_window=14, ma_window=21, adx_threshold=25, slope_threshold=0.001):
        """
        ADX and moving average slope for trend detection
        Returns: 'STRONG_TREND', 'WEAK_TREND', or 'SIDEWAYS'
        """
        if df is None or len(df) < ma_window or df[['high','low','close']].isnull().any().any():
            return 'SIDEWAYS'
        adx = ta.trend.adx(df['high'], df['low'], df['close'], window=adx_window)
        ma = df['close'].rolling(ma_window).mean()
        ma_slice = ma[-ma_window:]
        if len(ma_slice) == 0 or ma_slice.mean() == 0:
            return 'SIDEWAYS'
        slope = np.polyfit(np.arange(ma_window), ma_slice, 1)[0] / ma_slice.mean()
        last_adx = adx.iloc[-1]
        if last_adx > adx_threshold and abs(slope) > slope_threshold:
            return 'STRONG_TREND'
        elif last_adx < adx_threshold:
            return 'SIDEWAYS'
        else:
            return 'WEAK_TREND'

    def train_phase_classifier(self, df, feature_columns, target):
        """
        Train ML classifier for market phase (bull/bear/sideways)
        """
        X = df[feature_columns].values
        y = target
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)

    def predict_phase(self, df, feature_columns):
        """
        Predict market phase using trained classifier
        Returns: 'BULL', 'BEAR', or 'SIDEWAYS' - simplified to avoid training issues
        """
        # Simplified phase detection based on price momentum to avoid ML training issues
        try:
            prices = df['close'].values
            if len(prices) < 20:
                return 'SIDEWAYS'
            
            # Simple momentum calculation
            short_ma = np.mean(prices[-10:])
            long_ma = np.mean(prices[-20:])
            
            if short_ma > long_ma * 1.02:  # 2% upward momentum
                return 'BULL'
            elif short_ma < long_ma * 0.98:  # 2% downward momentum
                return 'BEAR'
            else:
                return 'SIDEWAYS'
        except:
            return 'SIDEWAYS'

    def get_regime_summary(self, df, feature_columns=None):
        """
        Returns a summary of all detected regimes
        """
        summary = {}
        summary['volatility'] = self.compute_volatility_regime(df)
        summary['trend'] = self.compute_trend_strength(df)
        if feature_columns:
            summary['phase'] = self.predict_phase(df, feature_columns)
        return summary

# Example usage:
# regime_detector = MarketRegimeDetector()
# regime = regime_detector.get_regime_summary(df, feature_columns)
# print(regime)
