"""
Preprocessor Module for Iris Classification Project
Handles feature scaling using StandardScaler and train-test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path


class IrisPreprocessor:
    """
    Preprocesses Iris data: scales features and encodes labels.
    
    Attributes:
        scaler (StandardScaler): Scaler for feature normalization
        label_encoder (LabelEncoder): Encoder for target labels
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.test_size = test_size
        self.random_state = random_state
        
        # Store fitted state
        self._is_fitted = False
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Fit scaler on training data and transform both features and labels.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target labels
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, 
                   X_train_raw, X_test_raw)
        """
        print("\n🔧 PREPROCESSING PIPELINE")
        print("=" * 50)
        
        # Step 1: Train-Test Split (80/20 as per PDF)
        print(f"\n📊 Train-Test Split: {100*(1-self.test_size):.0f}% train, {100*self.test_size:.0f}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Maintain class balance
        )
        
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        # Step 2: Feature Scaling (StandardScaler)
        print("\n📏 Feature Scaling (StandardScaler)")
        print("   Formula: z = (x - μ) / σ")
        print("   Target: Mean=0, Variance=1")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Show before/after stats
        print(f"\n   BEFORE scaling (train set):")
        print(f"     Mean: {X_train.mean().values.round(3)}")
        print(f"     Std:  {X_train.std().values.round(3)}")
        
        print(f"\n   AFTER scaling (train set):")
        print(f"     Mean: {X_train_scaled.mean(axis=0).round(3)}")
        print(f"     Std:  {X_train_scaled.std(axis=0).round(3)}")
        
        # Step 3: Label Encoding
        print("\n🏷️  Label Encoding")
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)
        
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        print(f"   Encoded: {list(range(len(self.label_encoder.classes_)))}")
        
        self._is_fitted = True
        print("\n" + "=" * 50)
        print("✅ Preprocessing complete!")
        
        return (X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, 
                X_train, X_test)
    
    def save_scaler(self, path: str = "models/scaler.pkl"):
        """Save the fitted scaler to disk."""
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted yet!")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"💾 Scaler saved to {path}")
    
    def save_encoder(self, path: str = "models/encoder.pkl"):
        """Save the fitted label encoder to disk."""
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted yet!")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.label_encoder, path)
        print(f"💾 Encoder saved to {path}")
    
    def get_feature_names(self) -> list:
        """Return the feature names."""
        return ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    def get_class_names(self) -> list:
        """Return the class names."""
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted yet!")
        return list(self.label_encoder.classes_)


# Quick test
if __name__ == "__main__":
    from data_loader import IrisDataLoader
    
    print("🔧 Preprocessor Test Run\n")
    
    # Load data
    loader = IrisDataLoader()
    loader.load_data()
    X, y = loader.get_features_target()
    
    # Preprocess
    preprocessor = IrisPreprocessor(test_size=0.2, random_state=42)
    (X_train_s, X_test_s, y_train_e, y_test_e, 
     X_train_raw, X_test_raw) = preprocessor.fit_transform(X, y)
    
    # Save artifacts
    preprocessor.save_scaler()
    preprocessor.save_encoder()
    
    print(f"\n✨ Ready for model training!")
    print(f"   X_train shape: {X_train_s.shape}")
    print(f"   X_test shape:  {X_test_s.shape}")