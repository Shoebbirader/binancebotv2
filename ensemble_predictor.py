"""Enhanced ensemble prediction with confidence calibration"""
import numpy as np
import logging
from typing import List, Dict, Tuple

class EnhancedEnsemble:
    def __init__(self):
        self.model_weights = {
            'lstm': 0.35,      # Deep learning gets higher weight
            'random_forest': 0.25,
            'xgboost': 0.25, 
            'lightgbm': 0.15
        }
        self.performance_history = {}
    
    def weighted_ensemble_predict(self, predictions: Dict[str, float]) -> Tuple[float, float]:
        """
        Enhanced ensemble prediction with weighted voting and confidence scoring
        Returns: (ensemble_prediction, confidence_score)
        """
        if not predictions:
            return 0.5, 0.0
        
        # Filter valid predictions
        valid_preds = {k: v for k, v in predictions.items() 
                      if v is not None and not np.isnan(v) and 0 <= v <= 1}
        
        if len(valid_preds) < 2:
            logging.warning(f"Insufficient valid predictions: {len(valid_preds)}")
            return 0.5, 0.0
        
        # Calculate weighted ensemble
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_type, prediction in valid_preds.items():
            weight = self.model_weights.get(model_type, 0.25)
            weighted_sum += prediction * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5, 0.0
            
        ensemble_pred = weighted_sum / total_weight
        
        # Calculate confidence score based on model agreement
        pred_values = list(valid_preds.values())
        std_dev = np.std(pred_values)
        mean_pred = np.mean(pred_values)
        
        # Confidence inversely related to standard deviation
        confidence = max(0.0, 1.0 - (std_dev * 2))  # Higher agreement = higher confidence
        
        # Boost confidence for extreme predictions
        if ensemble_pred > 0.7 or ensemble_pred < 0.3:
            confidence *= 1.2
        
        # Penalize confidence for predictions near 0.5 (uncertain)
        if 0.4 <= ensemble_pred <= 0.6:
            confidence *= 0.7
            
        confidence = min(1.0, max(0.0, confidence))
        
        return ensemble_pred, confidence
    
    def update_model_performance(self, model_type: str, accuracy: float):
        """Update model weights based on recent performance"""
        if model_type not in self.performance_history:
            self.performance_history[model_type] = []
        
        self.performance_history[model_type].append(accuracy)
        
        # Keep last 50 performance scores
        if len(self.performance_history[model_type]) > 50:
            self.performance_history[model_type] = self.performance_history[model_type][-50:]
        
        # Update weights based on recent performance
        if len(self.performance_history[model_type]) >= 10:
            recent_performance = np.mean(self.performance_history[model_type][-10:])
            
            # Adjust weight based on performance (simple linear scaling)
            base_weight = 0.25
            performance_multiplier = recent_performance / 0.6  # Normalize to 60% baseline
            new_weight = base_weight * performance_multiplier
            
            # Ensure weights stay within reasonable bounds
            self.model_weights[model_type] = max(0.1, min(0.5, new_weight))
            
            logging.info(f"Updated {model_type} weight to {self.model_weights[model_type]:.3f} based on recent performance: {recent_performance:.3f}")
    
    def get_trading_signal(self, ensemble_pred: float, confidence: float, 
                          buy_threshold: float = 0.6, sell_threshold: float = 0.4, 
                          min_confidence: float = 0.3) -> Tuple[str, bool]:
        """
        Generate trading signal with enhanced confidence requirements
        Returns: (action, should_execute)
        """
        if confidence < min_confidence:
            return "HOLD", False
        
        # Dynamic thresholds based on confidence
        adjusted_buy_threshold = buy_threshold - (confidence * 0.1)  # Lower threshold for high confidence
        adjusted_sell_threshold = sell_threshold + (confidence * 0.1)  # Higher threshold for high confidence
        
        if ensemble_pred >= adjusted_buy_threshold:
            return "BUY", True
        elif ensemble_pred <= adjusted_sell_threshold:
            return "SELL", True
        else:
            return "HOLD", False
    
    def calibrate_confidence(self, predictions: List[float], actual_outcomes: List[int]) -> float:
        """
        Calibrate confidence scores based on historical accuracy
        Returns: calibration_factor
        """
        if len(predictions) < 10 or len(actual_outcomes) < 10:
            return 1.0
        
        # Simple reliability score
        correct_predictions = 0
        for pred, actual in zip(predictions, actual_outcomes):
            if (pred > 0.5 and actual == 1) or (pred <= 0.5 and actual == 0):
                correct_predictions += 1
        
        accuracy = correct_predictions / len(predictions)
        return accuracy / 0.6  # Normalize to 60% baseline

# Global ensemble instance
ensemble_predictor = EnhancedEnsemble()
