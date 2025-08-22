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
        # Optimized LSTM for faster training
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        
        # Simplified dense layers for faster training
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Simplified forward pass for faster training
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

def train_model_balanced(model, X, y, epochs=25, batch_size=32, lr=0.002):
    """
    OPTIMIZED: Faster training with reduced epochs, early stopping, and GPU acceleration
    """
    try:
        # Quick data validation
        if X is None or y is None or len(X) < 10:
            print("Warning: Insufficient data for training")
            return None
        
        # Ensure X and y have same length
        if len(X) != len(y):
            print(f"Warning: X and y size mismatch: X={len(X)}, y={len(y)}")
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            print(f"Truncated to {min_len} samples")

        # Simplified class balancing - use weights instead of resampling
        from collections import Counter
        counter = Counter(y)
        if len(counter) < 2:
            print("Warning: Only one class present")
            return None

        # Fast split (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}")

        # GPU acceleration setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Fixed training setup for better convergence
        criterion = nn.BCELoss()  # Changed from BCEWithLogitsLoss since sigmoid is already applied
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # Create fast data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Fast training loop
        best_val_loss = float('inf')
        patience = 5  # Reduced from 25
        patience_counter = 0

        for epoch in range(min(epochs, 20)):  # Reduced to 20 epochs to prevent timeout
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                if len(batch_X.shape) == 2:
                    batch_X = batch_X.unsqueeze(0)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                # Ensure outputs are properly squeezed and in range [0,1]
                outputs = torch.clamp(outputs.squeeze(), 0.001, 0.999)  # Prevent extreme values
                loss = criterion(outputs, batch_y.float())
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    if len(batch_X.shape) == 2:
                        batch_X = batch_X.unsqueeze(0)
                    
                    outputs = model(batch_X)
                    outputs = torch.clamp(outputs.squeeze(), 0.001, 0.999)  # Prevent extreme values
                    loss = criterion(outputs, batch_y.float())
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 3 == 0:  # Less verbose
                print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stop at epoch {epoch+1}")
                break

        return model
        
    except Exception as e:
        print(f"Training error: {e}")
        return None

def predict_model(model, X):
    """
    Enhanced prediction handling with proper output validation
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
            
            # Ensure input is properly normalized
            X_tensor = torch.nan_to_num(X_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            
            output = model(X_tensor)
            
            # Ensure output is in valid range and not extreme
            output = torch.clamp(output, 0.01, 0.99)  # Prevent extreme predictions
            prediction = output.squeeze().cpu().numpy()
            
            if isinstance(prediction, np.ndarray):
                if len(prediction.shape) == 0:
                    pred_value = float(prediction)
                else:
                    pred_value = float(prediction[0])
            else:
                pred_value = float(prediction)
            
            # Final validation
            if np.isnan(pred_value) or np.isinf(pred_value):
                print("Warning: Invalid prediction, using fallback")
                return 0.5
            
            return pred_value
                
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.5

def train_random_forest(X, y):
    try:
        if len(X) != len(y):
            print(f"RandomForest: X and y size mismatch: X={len(X)}, y={len(y)}")
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
        
        clf = RandomForestClassifier(
            n_estimators=100,  # Reduced from 200 for faster training
            max_depth=12,      # Reduced from 15 
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1          # Use all CPU cores for parallel training
        )
        X_flat = X.reshape(X.shape[0], -1)
        clf.fit(X_flat, y)
        return clf
    except Exception as e:
        print(f"RandomForest training failed: {e}")
        return None

def train_xgboost(X, y):
    try:
        if xgb is None:
            print("XGBoost not installed.")
            return None
        
        if len(X) != len(y):
            print(f"XGBoost: X and y size mismatch: X={len(X)}, y={len(y)}")
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
        X_flat = X.reshape(X.shape[0], -1)
        
        # Suppress XGBoost warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            clf = xgb.XGBClassifier(
                n_estimators=100,  # Reduced from 200 for faster training
                max_depth=6,       # Reduced from 8
                learning_rate=0.15, # Increased for faster convergence
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=0.1,
                use_label_encoder=False, 
                eval_metric='logloss',
                verbosity=0,
                n_jobs=-1          # Use all CPU cores
            )
            clf.fit(X_flat, y)
        return clf
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        return None

def train_lightgbm(X, y):
    try:
        if lgb is None:
            print("LightGBM not installed.")
            return None
        
        if len(X) != len(y):
            print(f"LightGBM: X and y size mismatch: X={len(X)}, y={len(y)}")
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
        X_flat = X.reshape(X.shape[0], -1)
        clf = lgb.LGBMClassifier(
            n_estimators=100,     # Reduced from 200 for faster training
            learning_rate=0.1,    # Increased for faster convergence
            max_depth=8,          # Reduced from 10
            num_leaves=64,        # Reduced from 100
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1,
            n_jobs=-1             # Use all CPU cores
        )
        clf.fit(X_flat, y)
        return clf
    except Exception as e:
        print(f"LightGBM training failed: {e}")
        return None

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
