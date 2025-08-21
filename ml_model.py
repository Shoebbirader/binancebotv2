import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from sklearn.ensemble import RandomForestClassifier
import shap
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

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

class EnhancedTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, output_dim=1, dropout=0.2, max_seq_len=100):
        super(EnhancedTransformer, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input projection with residual connection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Multiple transformer encoder layers with residual connections
        encoder_layers = nn.ModuleList([
            EnhancedTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        self.transformer_layers = encoder_layers
        
        # Global attention pooling
        self.attention_pool = AttentionPooling(d_model)
        
        # Classification head with skip connections
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_dim)
        )
        
        # Output activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer layers with residual connections
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global attention pooling
        x = self.attention_pool(x)
        
        # Classification
        x = self.classifier(x)
        x = self.sigmoid(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        pooled = torch.sum(attention_weights * x, dim=1)  # (batch_size, d_model)
        return pooled

class EnhancedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True, activation='gelu'):
        super(EnhancedTransformerEncoderLayer, self).__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        if activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
            
    def forward(self, src):
        # Self-attention with residual connection
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        
        # Feed-forward with residual connection
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

def train_model_balanced(model, X, y, epochs=100, batch_size=32, lr=0.001):
    """
    FIXED: Enhanced training with weighted sampling for class imbalance and stronger architecture
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Robust NaN handling
    if isinstance(X, np.ndarray):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # Normalize features
        X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-8)
        X_tensor = torch.FloatTensor(X).to(device)
    else:
        X_tensor = X.to(device)

    if isinstance(y, np.ndarray):
        y_tensor = torch.FloatTensor(y).to(device)
    else:
        y_tensor = y.to(device)
    
    # Handle class imbalance with weighted sampling instead of data removal
    if isinstance(y, np.ndarray):
        unique, counts = np.unique(y, return_counts=True)
        min_count = min(counts)
        max_count = max(counts)
        
        if max_count / min_count > 2:  # Significant imbalance
            # Use weighted sampling instead of removing data
            class_weights = {cls: len(y) / (len(unique) * count) for cls, count in zip(unique, counts)}
            sample_weights = np.array([class_weights[label] for label in y])
            
            # Sample indices based on weights to balance classes
            from sklearn.utils import resample
            indices = np.arange(len(y))
            balanced_indices = resample(
                indices, 
                stratify=y,
                n_samples=min(len(y), max_count * 2),  # Keep more data
                random_state=42,
                replace=False
            )
            
            X = X[balanced_indices]
            y = y[balanced_indices]
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(X_tensor))
    X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
    
    # Calculate class weights for balanced loss
    pos_count = torch.sum(y_train == 1).item()
    neg_count = torch.sum(y_train == 0).item()
    
    if pos_count > 0 and neg_count > 0:
        pos_weight = neg_count / pos_count
        class_weight = torch.tensor([pos_weight]).to(device)
    else:
        class_weight = torch.tensor([1.0]).to(device)
    
    # Create weighted sampler for balanced batches
    if pos_count > 0 and neg_count > 0:
        sample_weights = torch.ones(len(y_train))
        sample_weights[y_train == 1] = pos_weight
        sample_weights[y_train == 0] = 1.0
        
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, 
            num_samples=len(y_train), 
            replacement=True
        )
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Enhanced optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    best_model_state = None
    
    print(f"Training with {len(X_train)} train, {len(X_val)} validation samples")
    print(f"Positive samples: {pos_count}, Negative samples: {neg_count}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
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
            
            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
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
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
            
            # Metrics
            val_acc = accuracy_score(y_val.cpu().numpy(), val_preds.cpu().numpy())
            val_precision = precision_score(y_val.cpu().numpy(), val_preds.cpu().numpy(), zero_division=0)
            val_recall = recall_score(y_val.cpu().numpy(), val_preds.cpu().numpy(), zero_division=0)
            val_f1 = f1_score(y_val.cpu().numpy(), val_preds.cpu().numpy(), zero_division=0)
        
        train_acc = accuracy_score(all_labels, all_preds)
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.3f}")
            print(f"          Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}")
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
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

def train_random_forest(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_flat = X.reshape(X.shape[0], -1)
    clf.fit(X_flat, y)
    return clf

def train_xgboost(X, y):
    if xgb is None:
        print("XGBoost not installed.")
        return None
    X_flat = X.reshape(X.shape[0], -1)
    clf = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_flat, y)
    return clf

def train_lightgbm(X, y):
    if lgb is None:
        print("LightGBM not installed.")
        return None
    X_flat = X.reshape(X.shape[0], -1)
    clf = lgb.LGBMClassifier(n_estimators=100)
    clf.fit(X_flat, y)
    return clf

def shap_feature_importance(model, X):
    X_flat = X.reshape(X.shape[0], -1)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_flat)
    shap.summary_plot(shap_values, X_flat)

def online_update(model, X_new, y_new):
    if hasattr(model, 'partial_fit'):
        X_flat = X_new.reshape(X_new.shape[0], -1)
        model.partial_fit(X_flat, y_new)
        print("Model updated online.")
    else:
        print("Online learning not supported for this model.")

class FeatureAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def detect_anomalies_autoencoder(X, threshold=0.05):
    input_dim = X.shape[-1]
    model = FeatureAutoencoder(input_dim)
    X_tensor = torch.FloatTensor(X.reshape(-1, input_dim))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, X_tensor)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        recon = model(X_tensor)
        errors = ((recon - X_tensor) ** 2).mean(dim=1)
        anomalies = errors > threshold
    print(f"Detected {anomalies.sum()} anomalies out of {len(anomalies)} samples.")
    return anomalies

def ensemble_predict(models, X):
    X_flat = X.reshape(X.shape[0], -1)
    preds = []
    for model in models:
        if model is not None:
            preds.append(model.predict_proba(X_flat)[:, 1])
    if preds:
        avg_pred = np.mean(preds, axis=0)
        return avg_pred
    else:
        return np.zeros(X.shape[0])
