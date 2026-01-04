"""
VibroFlow AI - Main Training Script
Trains models for both Hydraulic System and CWRU Bearing datasets
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import HydraulicDataLoader, CWRUBearingDataLoader
from data.preprocessor import DatasetPreprocessor
from data.features import FeatureExtractor
from models.baseline import BaselineClassifier, ModelComparison
from models.deep_learning import CNN1D, DeepLearningTrainer
from flow.estimator import FlowEstimator

def train_hydraulic_maintenance():
    print("\n" + "="*50)
    print("PHASE 1: Training Hydraulic Maintenance Models (dataset0)")
    print("="*50)
    
    loader = HydraulicDataLoader("dataset0")
    preprocessor = DatasetPreprocessor()
    
    # 1. Load targets
    targets = loader.load_targets()
    print(f"Loaded {len(targets)} cycles.")
    
    # 2. Load sensors (using VS1, PS1, FS1 for a sample feature set)
    print("Loading sensors...")
    sensors = loader.load_all_sensors(['VS1', 'PS1', 'FS1', 'TS1'])
    
    # 3. Extract Features
    print("Extracting features from sensors...")
    extractor = FeatureExtractor(sampling_freq=100) # 100Hz max for PS1
    
    all_features = []
    for i in range(len(targets)):
        cycle_features = {}
        for sensor, data in sensors.items():
            # Extract basic features for each cycle
            f = extractor.extract_all(data[i], include_wavelet=False)
            for k, v in f.items():
                cycle_features[f"{sensor}_{k}"] = v
        all_features.append(cycle_features)
    
    X = pd.DataFrame(all_features).values
    
    # 4. Train models for each condition
    results = {}
    for target_col in ['cooler', 'valve', 'pump_leakage', 'accumulator']:
        print(f"\nTraining for: {target_col}")
        y = targets[target_col].values
        
        # Split
        X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(X, y)
        
        # Compare base models
        comparison = ModelComparison(models=['rf', 'gb'])
        comp_df = comparison.compare(X_train, y_train, X_test, y_test)
        print(comp_df)
        
        # Save best model
        best_type, best_clf = comparison.get_best_model()
        model_path = f"models/hydraulic_{target_col}_{best_type}.joblib"
        best_clf.save(model_path)
        results[target_col] = best_type

    return results

def train_flow_estimation():
    print("\n" + "="*50)
    print("PHASE 2: Training Flow Estimation Model")
    print("="*50)
    
    loader = HydraulicDataLoader("dataset0")
    
    # Load vibration and flow
    vs1 = loader.get_vibration_data()
    fs1, fs2 = loader.get_flow_data()
    
    estimator = FlowEstimator(model_type='rf')
    print("Training vibration-to-flow estimator...")
    res = estimator.train_from_sensors(vs1, fs1)
    
    print(f"Flow Estimation Metrics: {res['test_metrics']}")
    
    estimator.save("models/flow_estimator_vibration.joblib")
    return res

def train_cwru_vibration():
    print("\n" + "="*50)
    print("PHASE 3: Training CWRU Bearing Fault Classifier")
    print("="*50)
    
    # This might take time, let's check if the preprocessed NPZ exists
    cwru_path = Path("dataset1/CWRU_48k_load_1_CNN_data.npz")
    if cwru_path.exists():
        print("Using preprocessed CNN data...")
        loader = CWRUBearingDataLoader("dataset1")
        X, y = loader.load_preprocessed_npz()
        
        # Normalize
        X = X / np.max(np.abs(X))
        
        # Split
        X_train, X_val, y_train, y_val = DatasetPreprocessor().create_train_test_split(X, y)
        
        # Train CNN
        print("Training CNN model...")
        model = CNN1D(input_size=1024, num_classes=10) # X shape is usually (N, 1024)
        trainer = DeepLearningTrainer(model)
        
        # Small epoch count for demo/initial run
        trainer.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=64)
        
        trainer.save("models/cwru_cnn_model.pth")
    else:
        print("Preprocessed data not found. Use features CSV for baseline training.")
        loader = CWRUBearingDataLoader("dataset1")
        features_df = loader.load_features_csv()
        
        X = features_df.drop('fault', axis=1).values
        y = pd.Categorical(features_df['fault']).codes
        
        X_train, X_test, y_train, y_test = DatasetPreprocessor().create_train_test_split(X, y)
        
        clf = BaselineClassifier('rf')
        clf.fit(X_train, y_train)
        metrics = clf.evaluate(X_test, y_test)
        print(f"CWRU Baseline Accuracy: {metrics['accuracy']:.4f}")
        clf.save("models/cwru_baseline_rf.joblib")

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    try:
        # Start training
        train_hydraulic_maintenance()
        train_flow_estimation()
        train_cwru_vibration()
        print("\n" + "="*50)
        print("TRAINING COMPLETE: All models saved in models/ directory.")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
