"""
Data Loader Module for Iris Classification Project
Handles loading, exploring, and validating the Iris dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class IrisDataLoader:
    """
    Loads and explores the Iris dataset.
    
    Attributes:
        data_path (Path): Path to the iris.csv file
        df (pd.DataFrame): Loaded dataframe
        features (list): List of feature column names
        target (str): Target column name
    """
    
    def __init__(self, data_path: str = "data/iris.csv"):
        self.data_path = Path(data_path)
        self.df = None
        self.features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.target = 'species'
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the Iris dataset from CSV.
        
        Returns:
            pd.DataFrame: The loaded dataset
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded successfully: {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df
    
    def explore(self) -> dict:
        """
        Explore the dataset and return key statistics.
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        stats = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'class_distribution': self.df[self.target].value_counts().to_dict(),
            'feature_stats': self.df[self.features].describe().to_dict()
        }
        
        print("\n📊 DATASET OVERVIEW")
        print("=" * 50)
        print(f"Shape: {stats['shape']}")
        print(f"Features: {stats['columns'][:-1]}")
        print(f"Target: {stats['columns'][-1]}")
        print(f"\nClass Distribution:")
        for species, count in stats['class_distribution'].items():
            print(f"  • {species}: {count} samples")
        print(f"\nMissing Values: {sum(stats['missing_values'].values())}")
        print("=" * 50)
        
        return stats
    
    def get_features_target(self) -> tuple:
        """
        Split dataframe into features (X) and target (y).
        
        Returns:
            tuple: (X, y) where X is features DataFrame and y is target Series
        """
        if self.df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
            
        X = self.df[self.features]
        y = self.df[self.target]
        
        print(f"\n📦 Data split: X={X.shape}, y={y.shape}")
        return X, y
    
    def validate(self) -> bool:
        """
        Validate dataset integrity.
        
        Returns:
            bool: True if dataset is valid
        """
        if self.df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        checks = {
            'has_150_rows': len(self.df) == 150,
            'has_5_columns': len(self.df.columns) == 5,
            'no_missing': self.df.isnull().sum().sum() == 0,
            '3_classes': self.df[self.target].nunique() == 3,
            'balanced': all(count == 50 for count in self.df[self.target].value_counts())
        }
        
        print("\n🔍 VALIDATION CHECKS")
        print("=" * 50)
        all_passed = True
        for check, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 All checks passed! Dataset is valid.")
        else:
            print("\n⚠️ Some checks failed. Please review the dataset.")
        print("=" * 50)
        
        return all_passed


# Quick test when running this file directly
if __name__ == "__main__":
    print("🌸 Iris Data Loader - Test Run\n")
    
    loader = IrisDataLoader()
    loader.load_data()
    loader.explore()
    loader.validate()
    X, y = loader.get_features_target()
    
    print(f"\n✨ Ready for preprocessing!")