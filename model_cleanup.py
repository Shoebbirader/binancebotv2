"""Model file cleanup and management"""
import os
import glob
import logging
from datetime import datetime, timedelta
from typing import List, Dict

class ModelFileManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.max_models_per_symbol = 5  # Keep only 5 most recent models per symbol per type
        self.max_age_days = 7  # Delete models older than 7 days
        
    def cleanup_old_models(self):
        """Clean up old model files to prevent storage bloat"""
        try:
            if not os.path.exists(self.models_dir):
                return
            
            print("🧹 Starting model cleanup...")
            
            # Get all model files
            model_files = []
            for ext in ['*.pkl', '*.pth']:
                model_files.extend(glob.glob(os.path.join(self.models_dir, ext)))
            
            if not model_files:
                print("No model files found for cleanup")
                return
            
            # Group models by symbol and type
            models_by_key = {}
            
            for file_path in model_files:
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    symbol = parts[0]
                    model_type = parts[1]
                    timestamp_str = '_'.join(parts[2:]).replace('.pkl', '').replace('.pth', '')
                    
                    try:
                        # Parse timestamp
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        
                        key = f"{symbol}_{model_type}"
                        if key not in models_by_key:
                            models_by_key[key] = []
                        
                        models_by_key[key].append({
                            'path': file_path,
                            'filename': filename,
                            'timestamp': timestamp,
                            'size_mb': os.path.getsize(file_path) / (1024*1024)
                        })
                        
                    except ValueError:
                        # Invalid timestamp format, mark for deletion
                        self._delete_file(file_path, "Invalid timestamp format")
            
            total_deleted = 0
            total_size_freed = 0.0
            
            # Clean up each symbol/model group
            for key, models in models_by_key.items():
                deleted, size_freed = self._cleanup_model_group(key, models)
                total_deleted += deleted
                total_size_freed += size_freed
            
            print(f"🧹 Cleanup complete: Deleted {total_deleted} files, freed {total_size_freed:.1f}MB")
            logging.info(f"Model cleanup: {total_deleted} files deleted, {total_size_freed:.1f}MB freed")
            
        except Exception as e:
            logging.error(f"Error during model cleanup: {e}")
    
    def _cleanup_model_group(self, key: str, models: List[Dict]) -> tuple:
        """Clean up a specific symbol/model type group"""
        deleted_count = 0
        size_freed = 0.0
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Delete old models (keep only recent ones)
        cutoff_date = datetime.now() - timedelta(days=self.max_age_days)
        
        for i, model in enumerate(models):
            should_delete = False
            reason = ""
            
            # Delete if too old
            if model['timestamp'] < cutoff_date:
                should_delete = True
                reason = f"Older than {self.max_age_days} days"
            
            # Delete if exceeds max count (keep newest)
            elif i >= self.max_models_per_symbol:
                should_delete = True
                reason = f"Exceeds max count ({self.max_models_per_symbol})"
            
            if should_delete:
                size_freed += model['size_mb']
                deleted_count += 1
                self._delete_file(model['path'], reason)
        
        remaining = len(models) - deleted_count
        if deleted_count > 0:
            print(f"   📦 {key}: Kept {remaining}, deleted {deleted_count} models")
        
        return deleted_count, size_freed
    
    def _delete_file(self, file_path: str, reason: str):
        """Safely delete a model file"""
        try:
            os.remove(file_path)
            logging.info(f"Deleted model file: {os.path.basename(file_path)} ({reason})")
        except Exception as e:
            logging.error(f"Failed to delete {file_path}: {e}")
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics for model files"""
        try:
            if not os.path.exists(self.models_dir):
                return {'total_files': 0, 'total_size_mb': 0.0}
            
            model_files = []
            for ext in ['*.pkl', '*.pth']:
                model_files.extend(glob.glob(os.path.join(self.models_dir, ext)))
            
            total_size = sum(os.path.getsize(f) for f in model_files)
            
            return {
                'total_files': len(model_files),
                'total_size_mb': total_size / (1024*1024),
                'models_dir': self.models_dir
            }
            
        except Exception as e:
            logging.error(f"Error getting storage stats: {e}")
            return {'total_files': 0, 'total_size_mb': 0.0}
    
    def emergency_cleanup(self, max_files: int = 100):
        """Emergency cleanup if too many model files exist"""
        try:
            stats = self.get_storage_stats()
            
            if stats['total_files'] > max_files:
                print(f"🚨 Emergency cleanup: {stats['total_files']} files exceed limit ({max_files})")
                
                # Get all model files with timestamps
                model_files = []
                for ext in ['*.pkl', '*.pth']:
                    for file_path in glob.glob(os.path.join(self.models_dir, ext)):
                        try:
                            mtime = os.path.getmtime(file_path)
                            model_files.append((file_path, mtime))
                        except:
                            pass
                
                # Sort by modification time (oldest first)
                model_files.sort(key=lambda x: x[1])
                
                # Delete oldest files
                files_to_delete = len(model_files) - max_files
                deleted = 0
                
                for file_path, _ in model_files[:files_to_delete]:
                    try:
                        os.remove(file_path)
                        deleted += 1
                    except Exception as e:
                        logging.error(f"Emergency delete failed for {file_path}: {e}")
                
                print(f"🧹 Emergency cleanup: Deleted {deleted} oldest model files")
                
        except Exception as e:
            logging.error(f"Emergency cleanup failed: {e}")

# Global model file manager
model_file_manager = ModelFileManager()
