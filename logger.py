import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_error(message):
    """Log error messages"""
    logging.error(message)

def log_info(message):
    """Log info messages"""
    logging.info(message)

def log_warning(message):
    """Log warning messages"""
    logging.warning(message)

def log_trade(symbol, action, price, amount, timestamp=None):
    """Log trade information"""
    if timestamp is None:
        timestamp = datetime.now()
    logging.info(f"TRADE: {symbol} {action} {amount} @ {price} | {timestamp}")

def log_prediction(symbol, prediction, confidence, action):
    """Log prediction information"""
    logging.info(f"PREDICTION: {symbol} | {prediction:.4f} | Confidence: {confidence:.2f} | Action: {action}")
