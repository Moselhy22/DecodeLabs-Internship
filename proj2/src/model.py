"""
Model Module for Iris Classification Project
Implements K-Nearest Neighbors (KNN) with Elbow Method for K tuning.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import joblib
from pathlib import Path


class KNNModel:
    """
    K-Nearest Neighbors classifier with hyperparameter tuning.
    
    Attributes:
        k (int): Number of neighbors
        model (KNeighborsClassifier): The sklearn KNN model
        is_trained (bool): Whether the model has been trained
    """
    
    def __init__(self, k: int = 5):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.is_trained = False
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the KNN model on scaled training data.
        
        Args:
            X_train (np.ndarray): Scaled feature matrix
            y_train (np.ndarray): Encoded target labels
        """
        print(f"\n🤖 TRAINING KNN MODEL")
        print("=" * 50)
        print(f"Algorithm: K-Nearest Neighbors")
        print(f"K = {self.k} neighbors")
        print(f"Distance metric: Euclidean (Minkowski p=2)")
        print(f"Training samples: {len(X_train)}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print("\n✅ Model trained successfully!")
        print("=" * 50)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Scaled feature matrix
            
        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Scaled feature matrix
            
        Returns:
            np.ndarray: Probability for each class
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        return self.model.predict_proba(X)
    
    def save(self, path: str = "models/knn_model.pkl") -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")
    
    def load(self, path: str = "models/knn_model.pkl") -> None:
        """Load trained model from disk."""
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"📂 Model loaded from {path}")


class ElbowMethod:
    """
    Finds optimal K using the Elbow Method.
    
    As shown in PDF page 12: Plot error rate vs K, find the "elbow" point.
    """
    
    def __init__(self, k_range: range = range(1, 31)):
        self.k_range = k_range
        self.error_rates = []
        self.cv_scores = []
        self.optimal_k = None
        
    def find_optimal_k(self, X_train: np.ndarray, y_train: np.ndarray) -> int:
        """
        Find optimal K using cross-validation error rates.
        
        Args:
            X_train (np.ndarray): Scaled training features
            y_train (np.ndarray): Encoded training labels
            
        Returns:
            int: Optimal K value
        """
        print(f"\n📈 ELBOW METHOD: Finding Optimal K")
        print("=" * 50)
        print(f"Testing K values: {list(self.k_range)}")
        
        for k in self.k_range:
            # Use cross-validation for robust error estimation
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = scores.mean()
            error_rate = 1 - cv_mean
            
            self.cv_scores.append(cv_mean)
            self.error_rates.append(error_rate)
            
            print(f"  K={k:2d} | CV Accuracy: {cv_mean:.4f} | Error Rate: {error_rate:.4f}")
        
        # Find optimal K (minimum error rate)
        self.optimal_k = self.k_range.start + np.argmin(self.error_rates)
        
        print(f"\n🎯 Optimal K = {self.optimal_k}")
        print(f"   Minimum Error Rate: {min(self.error_rates):.4f}")
        print(f"   Maximum CV Accuracy: {max(self.cv_scores):.4f}")
        print("=" * 50)
        
        return self.optimal_k
    
    def plot(self, save_path: str = "results/elbow_plot.png") -> None:
        """
        Plot the Elbow curve and save to file.
        
        Args:
            save_path (str): Path to save the plot
        """
        if not self.error_rates:
            raise ValueError("Run find_optimal_k() first!")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.k_range, self.error_rates, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=self.optimal_k, color='red', linestyle='--', 
                   label=f'Optimal K = {self.optimal_k}')
        plt.xlabel('K Value (Number of Neighbors)', fontsize=12)
        plt.ylabel('Error Rate (1 - Accuracy)', fontsize=12)
        plt.title('Elbow Method for Optimal K\n(KNN Classifier - Iris Dataset)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(list(self.k_range))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Elbow plot saved to {save_path}")


# Quick test
if __name__ == "__main__":
    from data_loader import IrisDataLoader
    from preprocessor import IrisPreprocessor
    
    print("🤖 KNN Model Test Run\n")
    
    # Load and preprocess
    loader = IrisDataLoader()
    loader.load_data()
    X, y = loader.get_features_target()
    
    preprocessor = IrisPreprocessor(test_size=0.2, random_state=42)
    (X_train_s, X_test_s, y_train_e, y_test_e, 
     X_train_raw, X_test_raw) = preprocessor.fit_transform(X, y)
    
    # Find optimal K
    elbow = ElbowMethod(k_range=range(1, 21))
    optimal_k = elbow.find_optimal_k(X_train_s, y_train_e)
    elbow.plot()
    
    # Train final model with optimal K
    knn = KNNModel(k=optimal_k)
    knn.train(X_train_s, y_train_e)
    knn.save()
    
    # Quick test prediction
    y_pred = knn.predict(X_test_s)
    accuracy = (y_pred == y_test_e).mean()
    print(f"\n📊 Quick Test Accuracy: {accuracy:.4f}")
    
    print(f"\n✨ Model ready for evaluation!")