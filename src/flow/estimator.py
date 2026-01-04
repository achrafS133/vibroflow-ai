"""
VibroFlow AI - Flow Estimation Module
Non-intrusive flow measurement using vibration signatures
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Optional
import joblib
from pathlib import Path


class FlowEstimator:
    """
    Non-intrusive flow estimator using vibration signatures.
    
    Correlates vibration patterns with flow sensor data to estimate
    flow rates without direct measurement.
    """
    
    MODELS = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'rf': RandomForestRegressor,
        'gb': GradientBoostingRegressor,
        'mlp': MLPRegressor,
    }
    
    DEFAULT_PARAMS = {
        'linear': {},
        'ridge': {'alpha': 1.0},
        'lasso': {'alpha': 0.1},
        'rf': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
        'gb': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
        'mlp': {'hidden_layer_sizes': (100, 50), 'max_iter': 500, 'random_state': 42},
    }
    
    def __init__(self, model_type: str = 'rf'):
        """
        Initialize the flow estimator.
        
        Args:
            model_type: Type of regression model to use
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        self.model = self.MODELS[model_type](**self.DEFAULT_PARAMS[model_type])
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        
    def extract_vibration_features(self, vibration_data: np.ndarray) -> np.ndarray:
        """
        Extract features from vibration signal for flow estimation.
        
        Args:
            vibration_data: Vibration sensor data (n_samples, signal_length)
            
        Returns:
            Feature matrix
        """
        n_samples = vibration_data.shape[0]
        features = []
        
        for i in range(n_samples):
            signal = vibration_data[i]
            
            # Time-domain features
            feat = {
                'mean': np.mean(signal),
                'std': np.std(signal),
                'rms': np.sqrt(np.mean(signal ** 2)),
                'max': np.max(signal),
                'min': np.min(signal),
                'peak_to_peak': np.max(signal) - np.min(signal),
                'var': np.var(signal),
                'energy': np.sum(signal ** 2),
            }
            
            # Frequency features (simple)
            fft_vals = np.abs(np.fft.fft(signal))[:len(signal)//2]
            feat['spectral_energy'] = np.sum(fft_vals ** 2)
            feat['dominant_freq_idx'] = np.argmax(fft_vals)
            
            features.append(list(feat.values()))
        
        return np.array(features)
    
    def prepare_flow_targets(self, flow_data: np.ndarray, 
                              aggregation: str = 'mean') -> np.ndarray:
        """
        Prepare flow targets from flow sensor data.
        
        Args:
            flow_data: Flow sensor data (n_samples, signal_length)
            aggregation: How to aggregate ('mean', 'max', 'median')
            
        Returns:
            1D array of flow values
        """
        if aggregation == 'mean':
            return np.mean(flow_data, axis=1)
        elif aggregation == 'max':
            return np.max(flow_data, axis=1)
        elif aggregation == 'median':
            return np.median(flow_data, axis=1)
        else:
            return np.mean(flow_data, axis=1)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FlowEstimator':
        """
        Train the flow estimator.
        
        Args:
            X: Vibration features or raw data
            y: Flow values
            
        Returns:
            self
        """
        # Scale features
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Train model
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict flow values.
        
        Args:
            X: Vibration features
            
        Returns:
            Predicted flow values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.model.predict(X_scaled)
        
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the estimator.
        
        Args:
            X: Test features
            y_true: True flow values
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        }
    
    def train_from_sensors(self, vibration_data: np.ndarray, 
                           flow_data: np.ndarray) -> Dict:
        """
        End-to-end training from sensor data.
        
        Args:
            vibration_data: Raw vibration sensor data
            flow_data: Raw flow sensor data
            
        Returns:
            Training results
        """
        # Extract features
        X = self.extract_vibration_features(vibration_data)
        y = self.prepare_flow_targets(flow_data)
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train
        self.fit(X_train, y_train)
        
        # Evaluate
        train_metrics = self.evaluate(X_train, y_train)
        test_metrics = self.evaluate(X_test, y_test)
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
    
    def save(self, filepath: str):
        """Save the model and scalers."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"✓ Flow estimator saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FlowEstimator':
        """Load a saved model."""
        data = joblib.load(filepath)
        estimator = cls(model_type=data['model_type'])
        estimator.model = data['model']
        estimator.scaler_X = data['scaler_X']
        estimator.scaler_y = data['scaler_y']
        estimator.is_fitted = data['is_fitted']
        return estimator


class MultiSensorFlowEstimator(FlowEstimator):
    """
    Flow estimator using multiple sensor types (vibration, pressure, temperature).
    """
    
    def combine_sensor_features(self, sensor_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine features from multiple sensors.
        
        Args:
            sensor_data: Dictionary of sensor name to data arrays
            
        Returns:
            Combined feature matrix
        """
        all_features = []
        
        for sensor_name, data in sensor_data.items():
            # Extract basic statistics for each sensor
            features = np.column_stack([
                np.mean(data, axis=1),
                np.std(data, axis=1),
                np.max(data, axis=1),
                np.min(data, axis=1),
                np.sqrt(np.mean(data ** 2, axis=1))  # RMS
            ])
            all_features.append(features)
        
        return np.hstack(all_features)
