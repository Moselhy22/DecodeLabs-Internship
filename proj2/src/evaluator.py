"""
Evaluator Module for Iris Classification Project
Computes Confusion Matrix, Precision, Recall, F1-Score, and generates reports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    """
    Evaluates classification model performance with comprehensive metrics.
    
    Attributes:
        class_names (list): Names of the target classes
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
    """
    
    def __init__(self, class_names: list = None):
        self.class_names = class_names or ['setosa', 'versicolor', 'virginica']
        self.y_true = None
        self.y_pred = None
        self.cm = None
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels (encoded)
            y_pred (np.ndarray): Predicted labels (encoded)
            
        Returns:
            dict: Dictionary containing all metrics
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.cm = confusion_matrix(y_true, y_pred)
        
        print("\n📊 MODEL EVALUATION REPORT")
        print("=" * 60)
        
        # Overall Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class metrics
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        
        print(f"\n📋 PER-CLASS METRICS")
        print("-" * 60)
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print("-" * 60)
        
        for i, name in enumerate(self.class_names):
            print(f"{name:<15} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f}")
        
        # Macro averages (treat all classes equally)
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        print("-" * 60)
        print(f"{'MACRO AVG':<15} {macro_precision:>10.4f} {macro_recall:>10.4f} {macro_f1:>10.4f}")
        print("=" * 60)
        
        # Weighted averages (account for class imbalance)
        weighted_precision = precision_score(y_true, y_pred, average='weighted')
        weighted_recall = recall_score(y_true, y_pred, average='weighted')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\n⚖️  WEIGHTED AVG")
        print(f"   Precision: {weighted_precision:.4f}")
        print(f"   Recall:    {weighted_recall:.4f}")
        print(f"   F1-Score:  {weighted_f1:.4f}")
        
        # Why accuracy alone is not enough (PDF page 14)
        print(f"\n⚠️  WHY NOT JUST ACCURACY?")
        print(f"   Accuracy can hide poor performance on minority classes.")
        print(f"   F1-Score balances Precision & Recall — better for real-world use.")
        print(f"   Macro F1 treats all classes equally (best for balanced data like Iris).")
        
        metrics = {
            'accuracy': accuracy,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'confusion_matrix': self.cm.tolist()
        }
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: str = "results/confusion_matrix.png") -> None:
        """
        Plot and save the confusion matrix as a heatmap.
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.cm is None:
            raise ValueError("Run evaluate() first!")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            self.cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title('Confusion Matrix\nIris Classification (KNN, K=5)', fontsize=14, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add text annotations for TP/FP/FN/TN understanding
        plt.figtext(0.02, 0.02, 
                   "TP: Diagonal | FP: Off-diagonal (column) | FN: Off-diagonal (row)", 
                   fontsize=9, style='italic')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Confusion matrix plot saved to {save_path}")
    
    def explain_predictions(self) -> None:
        """
        Explain what each cell in the confusion matrix means.
        """
        if self.cm is None:
            raise ValueError("Run evaluate() first!")
        
        print(f"\n🔍 CONFUSION MATRIX BREAKDOWN")
        print("=" * 60)
        
        for i, true_class in enumerate(self.class_names):
            for j, pred_class in enumerate(self.class_names):
                count = self.cm[i, j]
                if i == j:
                    print(f"  [{true_class} → {pred_class}]: {count:2d} ✅ TRUE POSITIVE")
                else:
                    print(f"  [{true_class} → {pred_class}]: {count:2d} ❌ MISCLASSIFIED")
        
        print("=" * 60)
    
    def save_report(self, metrics: dict, save_path: str = "results/classification_report.txt") -> None:
        """
        Save full classification report to a text file.
        
        Args:
            metrics (dict): Metrics dictionary from evaluate()
            save_path (str): Path to save the report
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("IRIS CLASSIFICATION - CLASSIFICATION REPORT\n")
            f.write("Algorithm: K-Nearest Neighbors (K=5)\n")
            f.write("Dataset: Iris (150 samples, 3 classes, 4 features)\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            f.write(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
            f.write("-" * 50 + "\n")
            
            for i, name in enumerate(self.class_names):
                f.write(f"{name:<15} {metrics['precision_per_class'][i]:>10.4f} "
                       f"{metrics['recall_per_class'][i]:>10.4f} "
                       f"{metrics['f1_per_class'][i]:>10.4f}\n")
            
            f.write("-" * 50 + "\n")
            f.write(f"{'MACRO AVG':<15} {metrics['macro_precision']:>10.4f} "
                   f"{metrics['macro_recall']:>10.4f} {metrics['macro_f1']:>10.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(np.array(metrics['confusion_matrix']), separator=', ') + "\n")
        
        print(f"📝 Classification report saved to {save_path}")


# Quick test
if __name__ == "__main__":
    from data_loader import IrisDataLoader
    from preprocessor import IrisPreprocessor
    from model import KNNModel, ElbowMethod
    
    print("📊 Evaluator Test Run\n")
    
    # Full pipeline
    loader = IrisDataLoader()
    loader.load_data()
    X, y = loader.get_features_target()
    
    preprocessor = IrisPreprocessor(test_size=0.2, random_state=42)
    (X_train_s, X_test_s, y_train_e, y_test_e, 
     X_train_raw, X_test_raw) = preprocessor.fit_transform(X, y)
    
    elbow = ElbowMethod(k_range=range(1, 21))
    optimal_k = elbow.find_optimal_k(X_train_s, y_train_e)
    
    knn = KNNModel(k=optimal_k)
    knn.train(X_train_s, y_train_e)
    
    # Predictions
    y_pred = knn.predict(X_test_s)
    
    # Evaluate
    evaluator = ModelEvaluator(class_names=preprocessor.get_class_names())
    metrics = evaluator.evaluate(y_test_e, y_pred)
    evaluator.explain_predictions()
    evaluator.plot_confusion_matrix()
    evaluator.save_report(metrics)
    
    print(f"\n✨ Evaluation complete!")