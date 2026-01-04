"""
VibroFlow AI - Deep Learning Models
CNN and LSTM models for vibration signal classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for vibration signal classification.
    
    Architecture:
        Conv1D -> BatchNorm -> ReLU -> MaxPool (x3) -> FC -> Dropout -> Softmax
    """
    
    def __init__(self, input_size: int, num_classes: int, 
                 channels: List[int] = [32, 64, 128]):
        """
        Initialize the CNN.
        
        Args:
            input_size: Length of input signal
            num_classes: Number of output classes
            channels: List of channel sizes for conv layers
        """
        super(CNN1D, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels in channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            ))
            in_channels = out_channels
        
        # Calculate the size after convolutions
        conv_output_size = input_size
        for _ in channels:
            conv_output_size = conv_output_size // 2
        
        self.flatten_size = channels[-1] * conv_output_size
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """Forward pass."""
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        for conv in self.conv_layers:
            x = conv(x)
        
        return self.classifier(x)


class LSTM(nn.Module):
    """
    LSTM network for time-series classification.
    
    Architecture:
        LSTM (bidirectional) -> FC -> Softmax
    """
    
    def __init__(self, input_size: int, num_classes: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 bidirectional: bool = True):
        """
        Initialize the LSTM.
        
        Args:
            input_size: Length of input signal
            num_classes: Number of output classes
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=1,  # Single feature per timestep
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """Forward pass."""
        # Reshape: (batch, signal_len) -> (batch, signal_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(2)
        
        # LSTM output
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output
        if self.bidirectional:
            # Concatenate final hidden states from both directions
            h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_final = h_n[-1]
        
        return self.classifier(h_final)


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model combining spatial and temporal features.
    
    Architecture:
        CNN (feature extraction) -> LSTM (temporal modeling) -> FC
    """
    
    def __init__(self, input_size: int, num_classes: int):
        """
        Initialize the hybrid model.
        
        Args:
            input_size: Length of input signal
            num_classes: Number of output classes
        """
        super(HybridCNNLSTM, self).__init__()
        
        # CNN layers for local feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=64,  # CNN output channels
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # 64*2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """Forward pass."""
        # Add channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # CNN: (batch, 1, signal_len) -> (batch, 64, signal_len/4)
        cnn_out = self.cnn(x)
        
        # Transpose for LSTM: (batch, 64, time) -> (batch, time, 64)
        cnn_out = cnn_out.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)
        
        # Use final hidden state
        h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        
        return self.classifier(h_final)


class DeepLearningTrainer:
    """
    Trainer class for deep learning models.
    """
    
    def __init__(self, model: nn.Module, device: str = None):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
    def create_dataloader(self, X: np.ndarray, y: np.ndarray, 
                          batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Create a PyTorch DataLoader from numpy arrays.
        
        Args:
            X: Features array
            y: Labels array
            batch_size: Batch size
            shuffle: Whether to shuffle
            
        Returns:
            DataLoader object
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32,
              learning_rate: float = 0.001,
              early_stopping: int = 10) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping: Patience for early stopping
            
        Returns:
            Training history
        """
        train_loader = self.create_dataloader(X_train, y_train, batch_size)
        
        if X_val is not None and y_val is not None:
            val_loader = self.create_dataloader(X_val, y_val, batch_size, shuffle=False)
        else:
            val_loader = None
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        preds = torch.argmax(outputs, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(batch_y.cpu().numpy())
                
                val_loss /= len(val_loader)
                val_acc = accuracy_score(all_labels, all_preds)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - "
                          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}")
        
        # Restore best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                 class_names: Optional[list] = None) -> Dict:
        """
        Evaluate the model.
        
        Args:
            X: Test features
            y: True labels
            class_names: Optional class names
            
        Returns:
            Evaluation metrics
        """
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'classification_report': classification_report(
                y, y_pred, target_names=class_names, zero_division=0
            )
        }
    
    def save(self, filepath: str):
        """Save the model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        print(f"✓ Model loaded from {filepath}")
