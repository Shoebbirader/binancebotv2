import logging
import pandas as pd
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_error(message):
    """Log error messages."""
    logging.error(message)

def log_info(message):
    """Log info messages."""
    logging.info(message)

def log_warning(message):
    """Log warning messages."""
    logging.warning(message)

def log_trade(symbol, action, price, amount):
    """Log basic trade information."""
    logging.info(f"TRADE: {symbol} {action} at {price} for {amount}")

def log_prediction(symbol, prediction, confidence, action):
    """Log prediction information."""
    logging.info(f"PREDICTION: {symbol} - {action} (confidence: {confidence:.2f})")

# Global trades.csv file path
TRADES_FILE = 'trades.csv'

def ensure_trades_csv():
    """Ensure trades.csv exists with proper headers."""
    if not os.path.exists(TRADES_FILE):
        headers = [
            'timestamp', 'symbol', 'action', 'price', 'quantity', 'total_value',
            'balance_before', 'balance_after', 'position_size', 'leverage',
            'stop_loss', 'take_profit', 'prediction', 'confidence', 'regime',
            'pnl', 'pnl_pct', 'trade_type', 'status', 'order_id'
        ]
        df = pd.DataFrame(columns=headers)
        df.to_csv(TRADES_FILE, index=False)
        log_info(f"Created {TRADES_FILE} with headers")

def log_paper_trade(**kwargs):
    """Log comprehensive paper trading data to trades.csv."""
    ensure_trades_csv()
    
    # Create trade record
    trade_record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': kwargs.get('symbol', ''),
        'action': kwargs.get('action', ''),
        'price': kwargs.get('price', 0.0),
        'quantity': kwargs.get('quantity', 0.0),
        'total_value': kwargs.get('total_value', 0.0),
        'balance_before': kwargs.get('balance_before', 0.0),
        'balance_after': kwargs.get('balance_after', 0.0),
        'position_size': kwargs.get('position_size', 0.0),
        'leverage': kwargs.get('leverage', 1.0),
        'stop_loss': kwargs.get('stop_loss', 0.0),
        'take_profit': kwargs.get('take_profit', 0.0),
        'prediction': kwargs.get('prediction', 0.0),
        'confidence': kwargs.get('confidence', 0.0),
        'regime': kwargs.get('regime', ''),
        'pnl': kwargs.get('pnl', 0.0),
        'pnl_pct': kwargs.get('pnl_pct', 0.0),
        'trade_type': kwargs.get('trade_type', 'paper'),
        'status': kwargs.get('status', 'executed'),
        'order_id': kwargs.get('order_id', '')
    }
    
    # Append to CSV
    try:
        df = pd.DataFrame([trade_record])
        df.to_csv(TRADES_FILE, mode='a', header=False, index=False)
        log_info(f"Logged {trade_record['trade_type']} trade to {TRADES_FILE}")
    except Exception as e:
        log_error(f"Error logging trade to CSV: {e}")

def get_trade_history():
    """Get trade history from trades.csv."""
    if os.path.exists(TRADES_FILE):
        return pd.read_csv(TRADES_FILE)
    return pd.DataFrame()

def get_paper_trades():
    """Get only paper trading trades."""
    df = get_trade_history()
    if not df.empty:
        return df[df['trade_type'] == 'paper']
    return pd.DataFrame()
