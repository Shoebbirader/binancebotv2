"""Configuration validation module"""
import logging
from typing import Dict, Any, List, Tuple

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate trading configuration parameters
    Returns: (is_valid, list_of_errors)
    """
    errors = []
    
    # Required parameters
    required_params = [
        'symbols', 'interval', 'prediction_threshold', 'position_size_pct',
        'max_loss_per_trade_pct', 'min_leverage', 'max_leverage',
        'stop_loss_percent', 'take_profit_percent', 'max_positions'
    ]
    
    for param in required_params:
        if param not in config:
            errors.append(f"Missing required parameter: {param}")
    
    # Validate ranges
    if 'prediction_threshold' in config:
        val = config['prediction_threshold']
        if not (0.5 <= val <= 0.9):
            errors.append(f"prediction_threshold must be between 0.5 and 0.9, got {val}")
    
    if 'position_size_pct' in config:
        val = config['position_size_pct']
        if not (1 <= val <= 50):
            errors.append(f"position_size_pct must be between 1 and 50, got {val}")
    
    if 'max_loss_per_trade_pct' in config:
        val = config['max_loss_per_trade_pct']
        if not (0.5 <= val <= 10):
            errors.append(f"max_loss_per_trade_pct must be between 0.5 and 10, got {val}")
    
    if 'stop_loss_percent' in config:
        val = config['stop_loss_percent']
        if not (0.5 <= val <= 10):
            errors.append(f"stop_loss_percent must be between 0.5 and 10, got {val}")
    
    if 'take_profit_percent' in config:
        val = config['take_profit_percent']
        if not (1 <= val <= 20):
            errors.append(f"take_profit_percent must be between 1 and 20, got {val}")
    
    if 'min_leverage' in config and 'max_leverage' in config:
        min_lev = config['min_leverage']
        max_lev = config['max_leverage']
        if not (1 <= min_lev <= max_lev <= 20):
            errors.append(f"Invalid leverage range: min={min_lev}, max={max_lev}")
    
    if 'max_positions' in config:
        val = config['max_positions']
        if not (1 <= val <= 20):
            errors.append(f"max_positions must be between 1 and 20, got {val}")
    
    # Validate symbols list
    if 'symbols' in config:
        symbols = config['symbols']
        if not isinstance(symbols, list) or len(symbols) == 0:
            errors.append("symbols must be a non-empty list")
        elif len(symbols) > 50:
            errors.append(f"Too many symbols ({len(symbols)}), maximum is 50")
    
    # Validate interval
    if 'interval' in config:
        valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
        if config['interval'] not in valid_intervals:
            errors.append(f"Invalid interval: {config['interval']}")
    
    is_valid = len(errors) == 0
    
    if not is_valid:
        logging.error(f"Configuration validation failed: {errors}")
    else:
        logging.info("Configuration validation passed")
    
    return is_valid, errors

def validate_environment_vars() -> Tuple[bool, List[str]]:
    """Validate required environment variables"""
    import os
    errors = []
    
    required_vars = ['BINANCE_API_KEY', 'BINANCE_API_SECRET']
    
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing environment variable: {var}")
        elif len(os.getenv(var)) < 10:
            errors.append(f"Invalid {var}: too short")
    
    return len(errors) == 0, errors
