import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

class CryptoLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=2):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # FIXED: Simple input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # FIXED: Simplified transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)  # Only 1 layer
        
        # FIXED: Simple output
        self.fc = nn.Linear(d_model, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Project to d_model
        x = self.input_projection(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Simple pooling
        x = x.mean(dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def train_model_balanced(model, X, y, epochs=100, batch_size=32, lr=0.001):
    """
    FIXED: Simplified training with focus on actual learning
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert to tensors
    if isinstance(X, np.ndarray):
        X_tensor = torch.FloatTensor(X).to(device)
    else:
        X_tensor = X.to(device)
        
    if isinstance(y, np.ndarray):
        y_tensor = torch.FloatTensor(y).to(device)
    else:
        y_tensor = y.to(device)
    
    # FIXED: Better balanced sampling - use more data
    pos_mask = (y_tensor == 1)
    neg_mask = (y_tensor == 0)
    
    pos_indices = torch.where(pos_mask)[0]
    neg_indices = torch.where(neg_mask)[0]
    
    # Use more samples for better learning
    min_samples = min(len(pos_indices), len(neg_indices))
    target_samples = min_samples  # Use all available samples
    
    if len(pos_indices) > target_samples:
        pos_indices = pos_indices[torch.randperm(len(pos_indices))[:target_samples]]
    if len(neg_indices) > target_samples:
        neg_indices = neg_indices[torch.randperm(len(neg_indices))[:target_samples]]
    
    balanced_indices = torch.cat([pos_indices, neg_indices])
    balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]
    
    X_balanced = X_tensor[balanced_indices]
    y_balanced = y_tensor[balanced_indices]
    
    # Split into train/val
    split_idx = int(0.8 * len(X_balanced))
    X_train, X_val = X_balanced[:split_idx], X_balanced[split_idx:]
    y_train, y_val = y_balanced[:split_idx], y_balanced[split_idx:]
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # FIXED: Better optimizer and learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # FIXED: Use standard BCE loss for simplicity
    criterion = nn.BCELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"Training with {len(X_train)} samples, {len(X_val)} validation")
    print(f"Positive samples: {(y_train == 1).sum()}, Negative samples: {(y_train == 0).sum()}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            if len(X_batch.shape) == 2:
                X_batch = X_batch.unsqueeze(0)
            
            outputs = model(X_batch)
            
            # Handle target size mismatch
            if outputs.shape != y_batch.shape:
                outputs = outputs.squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
        
        # Validation
        model.eval()
        with torch.no_grad():
            if len(X_val.shape) == 2:
                X_val = X_val.unsqueeze(0)
            val_outputs = model(X_val)
            
            if val_outputs.shape != y_val.shape:
                val_outputs = val_outputs.squeeze()
                if val_outputs.dim() == 0:
                    val_outputs = val_outputs.unsqueeze(0)
            
            val_loss = criterion(val_outputs, y_val)
            val_preds = (val_outputs > 0.5).float()
            
            # Metrics
            val_acc = accuracy_score(y_val.cpu().numpy(), val_preds.cpu().numpy())
            val_precision = precision_score(y_val.cpu().numpy(), val_preds.cpu().numpy(), zero_division=0)
            val_recall = recall_score(y_val.cpu().numpy(), val_preds.cpu().numpy(), zero_division=0)
            val_f1 = f1_score(y_val.cpu().numpy(), val_preds.cpu().numpy(), zero_division=0)
        
        train_acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"          Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}")
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model

def predict_model(model, X):
    """
    FIXED: Better prediction handling
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X).to(device)
            else:
                X_tensor = X.to(device)
            
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(0)
            
            if X_tensor.numel() == 0:
                print("Warning: Empty input tensor")
                return 0.5
            
            output = model(X_tensor)
            prediction = output.squeeze().cpu().numpy()
            
            if isinstance(prediction, np.ndarray):
                if len(prediction.shape) == 0:
                    return float(prediction)
                else:
                    return float(prediction[0])
            else:
                return float(prediction)
                
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.5
