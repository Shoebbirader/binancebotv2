import os
import torch
import pickle
import json
from datetime import datetime

class ModelManager:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.ensure_models_dir()
    
    def ensure_models_dir(self):
        """Create models directory if it doesn't exist"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def save_model(self, model, symbol, model_type, metadata=None):
        """Save a trained model to disk"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{model_type}_{timestamp}.pth"
        filepath = os.path.join(self.models_dir, filename)
        
        # Save PyTorch models
        if isinstance(model, torch.nn.Module):
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
                'symbol': symbol,
                'timestamp': timestamp,
                'metadata': metadata or {}
            }, filepath)
        else:
            # Save sklearn models with pickle
            filename = f"{symbol}_{model_type}_{timestamp}.pkl"
            filepath = os.path.join(self.models_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'model_type': model_type,
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'metadata': metadata or {}
                }, f)
        
        return filepath
    
    def load_latest_model(self, symbol, model_type):
        """Load the most recent model for a symbol and type"""
        model_files = [f for f in os.listdir(self.models_dir) 
                      if f.startswith(f"{symbol}_{model_type}")]
        
        if not model_files:
            return None
        
        # Get the latest model file
        latest_file = sorted(model_files)[-1]
        filepath = os.path.join(self.models_dir, latest_file)
        
        # Load PyTorch models
        if latest_file.endswith('.pth'):
            checkpoint = torch.load(filepath, map_location='cpu')
            return checkpoint
        else:
            # Load sklearn models
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                return data
    
    def list_models(self, symbol=None):
        """List all saved models"""
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for filename in os.listdir(self.models_dir):
            if symbol and not filename.startswith(symbol):
                continue
            
            filepath = os.path.join(self.models_dir, filename)
            
            try:
                if filename.endswith('.pth'):
                    checkpoint = torch.load(filepath, map_location='cpu')
                    models.append({
                        'filename': filename,
                        'symbol': checkpoint['symbol'],
                        'model_type': checkpoint['model_type'],
                        'timestamp': checkpoint['timestamp'],
                        'metadata': checkpoint.get('metadata', {})
                    })
                else:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        models.append({
                            'filename': filename,
                            'symbol': data['symbol'],
                            'model_type': data['model_type'],
                            'timestamp': data['timestamp'],
                            'metadata': data.get('metadata', {})
                        })
            except Exception as e:
                print(f"Error loading model {filename}: {e}")
        
        return models
    
    def cleanup_old_models(self, symbol=None, keep_latest=3):
        """Clean up old model files, keeping only the latest N"""
        models = self.list_models(symbol)
        
        # Group by symbol and model type
        grouped = {}
        for model in models:
            key = (model['symbol'], model['model_type'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(model)
        
        # Keep only latest N for each group
        for key, model_list in grouped.items():
            if len(model_list) > keep_latest:
                # Sort by timestamp and remove old ones
                sorted_models = sorted(model_list, key=lambda x: x['timestamp'])
                for old_model in sorted_models[:-keep_latest]:
                    filepath = os.path.join(self.models_dir, old_model['filename'])
                    try:
                        os.remove(filepath)
                        print(f"Removed old model: {old_model['filename']}")
                    except Exception as e:
                        print(f"Error removing old model {old_model['filename']}: {e}")

# Global model manager instance
model_manager = ModelManager()