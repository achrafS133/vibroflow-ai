"""
VibroFlow AI - CWRU Training Script
Specific for Phase 3
"""

import os
import sys
import numpy as np
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import CWRUBearingDataLoader
from data.preprocessor import DatasetPreprocessor
from models.deep_learning import CNN1D, DeepLearningTrainer

def train_cwru():
    print("\n" + "="*50)
    print("PHASE 3: Training CWRU Bearing Fault Classifier")
    print("="*50)
    
    loader = CWRUBearingDataLoader("dataset1")
    cwru_path = Path("dataset1/CWRU_48k_load_1_CNN_data.npz")
    
    if cwru_path.exists():
        print("Loading NPZ data...")
        X, y = loader.load_preprocessed_npz()
        
        # Flatten if data is images/spectrograms (N, H, W) -> (N, H*W)
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
            
        X = X.astype(np.float32)
        X = X / (np.max(np.abs(X)) + 1e-10)
        
        # Convert string labels to integers
        import pandas as pd
        y_cat = pd.Categorical(y)
        y = y_cat.codes.astype(np.int64)
        class_names = list(y_cat.categories)
        
        print(f"Reshaped Data shape: {X.shape}, labels: {y.shape}")
        print(f"Classes: {class_names}")
        
        # Split
        X_train, X_val, y_train, y_val = DatasetPreprocessor().create_train_test_split(X, y)
        
        # Train CNN
        input_size = X.shape[1]
        num_classes = len(np.unique(y))
        
        print(f"Training CNN model (Input: {input_size}, Classes: {num_classes})...")
        model = CNN1D(input_size=input_size, num_classes=num_classes)
        trainer = DeepLearningTrainer(model)
        
        # Train for 10 epochs
        trainer.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
        
        trainer.save("models/cwru_cnn_model.pth")
        
        metrics = trainer.evaluate(X_val, y_val)
        print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    else:
        print("NPZ not found.")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_cwru()
