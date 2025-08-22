"""Model Performance Tracking and Auto-Enhancement"""
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional

class ModelPerformanceTracker:
    def __init__(self, performance_file='models/performance_history.json'):
        self.performance_file = performance_file
        self.performance_data = self._load_performance_data()
        
    def _load_performance_data(self):
        """Load historical performance data"""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading performance data: {e}")
            return {}
    
    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving performance data: {e}")
    
    def record_prediction(self, symbol: str, model_type: str, prediction: float, 
                         confidence: float, actual_outcome: int = None):
        """Record a model prediction for performance tracking"""
        timestamp = datetime.now().isoformat()
        
        if symbol not in self.performance_data:
            self.performance_data[symbol] = {}
        if model_type not in self.performance_data[symbol]:
            self.performance_data[symbol][model_type] = []
        
        record = {
            'timestamp': timestamp,
            'prediction': prediction,
            'confidence': confidence,
            'actual_outcome': actual_outcome
        }
        
        self.performance_data[symbol][model_type].append(record)
        
        # Keep only last 1000 records per model
        if len(self.performance_data[symbol][model_type]) > 1000:
            self.performance_data[symbol][model_type] = self.performance_data[symbol][model_type][-500:]
        
        self._save_performance_data()
    
    def get_model_accuracy(self, symbol: str, model_type: str, days: int = 7) -> float:
        """Calculate model accuracy over the last N days"""
        try:
            if symbol not in self.performance_data or model_type not in self.performance_data[symbol]:
                return 0.5  # Default accuracy
            
            records = self.performance_data[symbol][model_type]
            cutoff_date = datetime.now() - timedelta(days=days)
            
            recent_records = [
                r for r in records 
                if r.get('actual_outcome') is not None and 
                datetime.fromisoformat(r['timestamp']) > cutoff_date
            ]
            
            if len(recent_records) < 5:
                return 0.5  # Not enough data
            
            correct = 0
            for record in recent_records:
                prediction = record['prediction']
                actual = record['actual_outcome']
                
                # Check if prediction was correct
                if (prediction > 0.5 and actual == 1) or (prediction <= 0.5 and actual == 0):
                    correct += 1
            
            accuracy = correct / len(recent_records)
            return accuracy
            
        except Exception as e:
            logging.error(f"Error calculating model accuracy: {e}")
            return 0.5
    
    def should_retrain_model(self, symbol: str, model_type: str, accuracy_threshold: float = 0.45) -> bool:
        """Determine if a model should be retrained based on performance"""
        recent_accuracy = self.get_model_accuracy(symbol, model_type, days=3)
        
        if recent_accuracy < accuracy_threshold:
            logging.info(f"Model {model_type} for {symbol} needs retraining: accuracy {recent_accuracy:.3f} < {accuracy_threshold}")
            return True
        
        return False
    
    def get_best_model_for_symbol(self, symbol: str) -> str:
        """Get the best performing model for a symbol"""
        model_types = ['lstm', 'random_forest', 'xgboost', 'lightgbm']
        best_model = 'lstm'  # Default
        best_accuracy = 0.0
        
        for model_type in model_types:
            accuracy = self.get_model_accuracy(symbol, model_type)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_type
        
        return best_model
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary"""
        summary = {
            'total_symbols': len(self.performance_data),
            'model_accuracies': {},
            'avg_confidence': 0.0,
            'best_performers': {}
        }
        
        all_accuracies = []
        all_confidences = []
        
        for symbol, models in self.performance_data.items():
            summary['best_performers'][symbol] = self.get_best_model_for_symbol(symbol)
            
            for model_type in models:
                accuracy = self.get_model_accuracy(symbol, model_type)
                model_key = f"{symbol}_{model_type}"
                summary['model_accuracies'][model_key] = accuracy
                all_accuracies.append(accuracy)
                
                # Get recent confidences
                recent_records = models[model_type][-50:]  # Last 50 predictions
                confidences = [r.get('confidence', 0.5) for r in recent_records]
                all_confidences.extend(confidences)
        
        if all_accuracies:
            summary['avg_accuracy'] = np.mean(all_accuracies)
        if all_confidences:
            summary['avg_confidence'] = np.mean(all_confidences)
        
        return summary

# Global performance tracker
performance_tracker = ModelPerformanceTracker()
