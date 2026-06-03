#!/usr/bin/env python3
"""
Main Pipeline for Iris Classification Project (Project 2)

Full workflow:
    INPUT → PROCESS → OUTPUT
    Iris Data → Scaling + KNN → Confusion Matrix + F1 Score

Run: python main.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import IrisDataLoader
from preprocessor import IrisPreprocessor
from model import KNNModel, ElbowMethod
from evaluator import ModelEvaluator


def run_pipeline():
    """
    Execute the complete classification pipeline.
    """
    print("\n" + "=" * 70)
    print("🌸 DECODELABS - PROJECT 2: IRIS CLASSIFICATION")
    print("=" * 70)
    print("Goal: Build a KNN classifier to identify Iris flower species")
    print("Dataset: 150 samples | 3 classes | 4 features")
    print("=" * 70)
    
    # ============================================================
    # STEP 1: INPUT - Load and Understand Data
    # ============================================================
    print("\n📥 STEP 1: INPUT - Loading Iris Dataset")
    print("-" * 70)
    
    loader = IrisDataLoader(data_path="data/iris.csv")
    loader.load_data()
    stats = loader.explore()
    is_valid = loader.validate()
    
    if not is_valid:
        print("❌ Dataset validation failed. Exiting.")
        return 1
    
    X, y = loader.get_features_target()
    
    # ============================================================
    # STEP 2: PROCESS - Preprocess Data
    # ============================================================
    print("\n⚙️  STEP 2: PROCESS - Feature Scaling & Train-Test Split")
    print("-" * 70)
    
    preprocessor = IrisPreprocessor(test_size=0.2, random_state=42)
    (X_train_s, X_test_s, y_train_e, y_test_e,
     X_train_raw, X_test_raw) = preprocessor.fit_transform(X, y)
    
    # Save preprocessing artifacts
    preprocessor.save_scaler("models/scaler.pkl")
    preprocessor.save_encoder("models/encoder.pkl")
    
    # ============================================================
    # STEP 3: PROCESS - Find Optimal K (Elbow Method)
    # ============================================================
    print("\n🔧 STEP 3: PROCESS - Hyperparameter Tuning (Elbow Method)")
    print("-" * 70)
    
    elbow = ElbowMethod(k_range=range(1, 21))
    optimal_k = elbow.find_optimal_k(X_train_s, y_train_e)
    elbow.plot(save_path="results/elbow_plot.png")
    
    # ============================================================
    # STEP 4: PROCESS - Train KNN Model
    # ============================================================
    print("\n🤖 STEP 4: PROCESS - Training KNN Classifier")
    print("-" * 70)
    
    knn = KNNModel(k=optimal_k)
    knn.train(X_train_s, y_train_e)
    knn.save("models/knn_model.pkl")
    
    # ============================================================
    # STEP 5: OUTPUT - Evaluate Model
    # ============================================================
    print("\n📊 STEP 5: OUTPUT - Model Evaluation")
    print("-" * 70)
    
    y_pred = knn.predict(X_test_s)
    
    evaluator = ModelEvaluator(class_names=preprocessor.get_class_names())
    metrics = evaluator.evaluate(y_test_e, y_pred)
    evaluator.explain_predictions()
    evaluator.plot_confusion_matrix("results/confusion_matrix.png")
    evaluator.save_report(metrics, "results/classification_report.txt")
    
    # ============================================================
    # STEP 6: SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("🏁 PIPELINE COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"✅ Dataset:        {stats['shape'][0]} samples, {stats['shape'][1]} columns")
    print(f"✅ Train/Test:     80% / 20% ({len(X_train_s)} / {len(X_test_s)})")
    print(f"✅ Optimal K:      {optimal_k}")
    print(f"✅ Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"✅ Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"✅ Saved Model:    models/knn_model.pkl")
    print(f"✅ Saved Scaler:   models/scaler.pkl")
    print(f"✅ Saved Encoder:  models/encoder.pkl")
    print(f"✅ Elbow Plot:     results/elbow_plot.png")
    print(f"✅ Confusion Mat:  results/confusion_matrix.png")
    print(f"✅ Report:         results/classification_report.txt")
    print("=" * 70)
    print("🎉 Project 2 completed successfully!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit_code = run_pipeline()
    sys.exit(exit_code)