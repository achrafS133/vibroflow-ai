"""
VibroFlow AI - Baseline ML Models
Classical machine learning models: SVM, Random Forest, XGBoost
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class BaselineClassifier:
    """
    Wrapper for baseline ML classifiers with unified interface.
    """
    
    MODELS = {
        'svm': SVC,
        'rf': RandomForestClassifier,
        'gb': GradientBoostingClassifier,
        'knn': KNeighborsClassifier,
        'lr': LogisticRegression,
    }
    
    DEFAULT_PARAMS = {
        'svm': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True},
        'rf': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
        'gb': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
        'knn': {'n_neighbors': 5},
        'lr': {'max_iter': 1000, 'random_state': 42},
    }
    
    PARAM_GRIDS = {
        'svm': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None]
        },
        'gb': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance']
        },
        'lr': {
            'C': [0.1, 1, 10],
            'penalty': ['l2']
        }
    }
    
    def __init__(self, model_type: str = 'rf', **kwargs):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model ('svm', 'rf', 'gb', 'knn', 'lr')
            **kwargs: Additional model parameters
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        params = {**self.DEFAULT_PARAMS[model_type], **kwargs}
        self.model = self.MODELS[model_type](**params)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaselineClassifier':
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                 class_names: Optional[list] = None) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: True labels
            class_names: Optional list of class names
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred),
        }
        
        # Detailed classification report
        metrics['classification_report'] = classification_report(
            y, y_pred, target_names=class_names, zero_division=0
        )
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary of CV scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return {
            'cv_mean': np.mean(scores),
            'cv_std': np.std(scores),
            'cv_scores': scores.tolist()
        }
    
    def grid_search(self, X: np.ndarray, y: np.ndarray, 
                    cv: int = 3, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X: Training features
            y: Training labels
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with best parameters and score
        """
        param_grid = self.PARAM_GRIDS.get(self.model_type, {})
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X, y)
        
        # Update model with best estimator
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save(self, filepath: str):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaselineClassifier':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded classifier
        """
        data = joblib.load(filepath)
        classifier = cls(model_type=data['model_type'])
        classifier.model = data['model']
        classifier.is_fitted = data['is_fitted']
        return classifier


class ModelComparison:
    """
    Compare multiple baseline models on the same dataset.
    """
    
    def __init__(self, models: list = ['svm', 'rf', 'gb', 'knn', 'lr']):
        """
        Initialize with list of models to compare.
        
        Args:
            models: List of model type strings
        """
        self.model_types = models
        self.classifiers = {}
        self.results = {}
        
    def compare(self, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                class_names: Optional[list] = None) -> pd.DataFrame:
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            class_names: Optional class names
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_type in self.model_types:
            print(f"\nTraining {model_type.upper()}...")
            
            # Create and train model
            clf = BaselineClassifier(model_type)
            clf.fit(X_train, y_train)
            
            # Evaluate
            metrics = clf.evaluate(X_test, y_test, class_names)
            
            results.append({
                'Model': model_type.upper(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision_macro'],
                'Recall': metrics['recall_macro'],
                'F1-Score': metrics['f1_macro']
            })
            
            self.classifiers[model_type] = clf
            self.results[model_type] = metrics
            
            print(f"✓ {model_type.upper()}: Accuracy = {metrics['accuracy']:.4f}")
        
        return pd.DataFrame(results).sort_values('F1-Score', ascending=False)
    
    def get_best_model(self) -> Tuple[str, BaselineClassifier]:
        """
        Get the best performing model based on F1-score.
        
        Returns:
            Tuple of (model_type, classifier)
        """
        best_type = max(
            self.results.keys(), 
            key=lambda k: self.results[k]['f1_macro']
        )
        return best_type, self.classifiers[best_type]


# XGBoost wrapper (if available)
try:
    from xgboost import XGBClassifier
    
    class XGBoostClassifier(BaselineClassifier):
        """XGBoost classifier wrapper."""
        
        def __init__(self, **kwargs):
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss'
            }
            params = {**default_params, **kwargs}
            self.model = XGBClassifier(**params)
            self.model_type = 'xgboost'
            self.is_fitted = False
            
except ImportError:
    XGBoostClassifier = None
    print("XGBoost not available. Install with: pip install xgboost")
