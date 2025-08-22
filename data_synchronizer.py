"""Data synchronization utilities to prevent X/y mismatches"""
import numpy as np
import logging

def validate_training_data(X, y, symbol):
    """Comprehensive validation of training data synchronization"""
    errors = []
    
    # Basic null checks
    if X is None or y is None:
        errors.append("X or y is None")
        return False, errors
    
    # Size validation
    if len(X) == 0 or len(y) == 0:
        errors.append("Empty X or y arrays")
        return False, errors
    
    # Length synchronization
    if len(X) != len(y):
        errors.append(f"X and y length mismatch: X={len(X)}, y={len(y)}")
        return False, errors
    
    # Data type validation
    try:
        X = np.array(X)
        y = np.array(y)
    except Exception as e:
        errors.append(f"Array conversion failed: {e}")
        return False, errors
    
    # Shape validation
    if len(X.shape) != 3:
        errors.append(f"X should be 3D array, got shape: {X.shape}")
        return False, errors
    
    if len(y.shape) != 1:
        errors.append(f"y should be 1D array, got shape: {y.shape}")
        return False, errors
    
    # NaN validation
    if np.isnan(X).any():
        errors.append("X contains NaN values")
        return False, errors
    
    if np.isnan(y).any():
        errors.append("y contains NaN values") 
        return False, errors
    
    # Class balance validation
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        errors.append(f"Only one class present: {unique}")
        return False, errors
    
    min_class_size = min(counts)
    if min_class_size < 10:
        errors.append(f"Insufficient samples per class: {dict(zip(unique, counts))}")
        return False, errors
    
    # Log success
    logging.info(f"Data validation passed for {symbol}: X{X.shape}, y{y.shape}, classes: {dict(zip(unique, counts))}")
    return True, []

def synchronize_arrays(X, y, symbol):
    """Force synchronization of X and y arrays"""
    try:
        if X is None or y is None:
            return None, None
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Force same length
        min_len = min(len(X), len(y))
        if min_len == 0:
            return None, None
        
        X_sync = X[:min_len]
        y_sync = y[:min_len]
        
        # Clean data
        X_sync = np.nan_to_num(X_sync, nan=0.0, posinf=0.0, neginf=0.0)
        y_sync = np.nan_to_num(y_sync, nan=0, posinf=0, neginf=0)
        
        # Ensure proper types
        X_sync = X_sync.astype(np.float32)
        y_sync = y_sync.astype(np.int64)
        
        logging.info(f"Arrays synchronized for {symbol}: X{X_sync.shape}, y{y_sync.shape}")
        return X_sync, y_sync
        
    except Exception as e:
        logging.error(f"Array synchronization failed for {symbol}: {e}")
        return None, None
