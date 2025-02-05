import os
import torch
import json
import pickle
from datetime import datetime

class ModelCheckpointer:
    def __init__(self, base_dir='checkpoints'):
        self.base_dir = base_dir
        self.metadata_file = os.path.join(base_dir, 'model_metadata.json')
        os.makedirs(base_dir, exist_ok=True)
        
    def save_model(self, model, optimizer, epoch, loss, scaler):
        """Save model checkpoint with metadata."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(self.base_dir, f'model_{timestamp}.pt')
        scaler_path = os.path.join(self.base_dir, f'scaler_{timestamp}.pkl')
        
        # Save model state and optimizer state
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save scaler separately using pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Update metadata file
        metadata = self._load_metadata()
        metadata[timestamp] = {
            'model_path': checkpoint_path,
            'scaler_path': scaler_path,
            'epoch': epoch,
            'loss': loss,
            'date': timestamp
        }
        self._save_metadata(metadata)
        
        return checkpoint_path
    
    def load_latest_model(self, model, optimizer=None, scaler=None):
        """Load the most recent model checkpoint."""
        metadata = self._load_metadata()
        if not metadata:
            return None, 0, float('inf')
            
        # Get most recent checkpoint
        latest_timestamp = max(metadata.keys())
        checkpoint_info = metadata[latest_timestamp]
        
        if not os.path.exists(checkpoint_info['model_path']):
            return None, 0, float('inf')
            
        # Load model checkpoint
        checkpoint = torch.load(checkpoint_info['model_path'])
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scaler if provided
        if scaler is not None and os.path.exists(checkpoint_info['scaler_path']):
            with open(checkpoint_info['scaler_path'], 'rb') as f:
                loaded_scaler = pickle.load(f)
                scaler.min_ = loaded_scaler.min_
                scaler.scale_ = loaded_scaler.scale_
                scaler.data_min_ = loaded_scaler.data_min_
                scaler.data_max_ = loaded_scaler.data_max_
                scaler.data_range_ = loaded_scaler.data_range_
            
        return model, checkpoint['epoch'], checkpoint['loss']
    
    def _load_metadata(self):
        """Load checkpoint metadata from JSON file."""
        if not os.path.exists(self.metadata_file):
            return {}
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    
    def _save_metadata(self, metadata):
        """Save checkpoint metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)