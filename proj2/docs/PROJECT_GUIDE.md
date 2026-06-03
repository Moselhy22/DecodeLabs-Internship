# 📖 Project 2: Iris Classification - Step-by-Step Guide

**DecodeLabs Internship | Batch 2026**

---

## 🎯 Goal

Build a K-Nearest Neighbors (KNN) classifier to identify Iris flower species based on 4 physical measurements.

---

## 📁 Project Structure

```text
proj2/
├── data/iris.csv              # The dataset (150 samples)
├── src/
│   ├── data_loader.py         # Load & validate data
│   ├── preprocessor.py        # Scale features & split data
│   ├── model.py              # KNN + Elbow Method
│   ├── evaluator.py          # Confusion Matrix & F1 Score
│   └── utils.py              # Helper functions
├── notebooks/
│   └── 01_eda.ipynb          # Interactive exploration
├── models/                   # Saved models
├── results/                  # Plots & reports
├── docs/
│   ├── PROJECT_GUIDE.md      # This file
│   └── ALGORITHM_EXPLAINED.md # Theory behind the algorithms
├── main.py                   # Full pipeline script
└── requirements.txt          # Python dependencies
```

---

## 🚀 How to Run

### 1. Setup Environment

```bash
cd proj2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python main.py
```

This executes:

1. **Load** → Reads `data/iris.csv`
2. **Validate** → Checks 150 rows, 3 classes, no missing values
3. **Preprocess** → StandardScaler + 80/20 train-test split
4. **Tune** → Elbow Method finds optimal K
5. **Train** → KNN classifier with optimal K
6. **Evaluate** → Confusion Matrix, Precision, Recall, F1
7. **Save** → Model, scaler, encoder, plots, report

### 3. Run Jupyter Notebook (Optional)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## 🔍 Understanding Each Step

### Step 1: Data Loading

* **What**: Load CSV into pandas DataFrame
* **Why**: Structured data is easier to manipulate
* **Key Check**: 50 samples per class (balanced dataset)

### Step 2: Feature Scaling

* **What**: StandardScaler transforms each feature to mean=0, std=1
* **Why**: KNN uses distance calculations; different scales bias results
* **Formula**:

```text
z = (x - μ) / σ
```

### Step 3: Train-Test Split

* **What**: 80% training, 20% testing
* **Why**: Test on unseen data to measure generalization
* **Important**: Use `stratify=y` to maintain class balance

### Step 4: Elbow Method

* **What**: Test K values from 1-20, plot error rate
* **Why**: Find the "sweet spot" between overfitting (K=1) and underfitting (large K)
* **Result**: K=5 is optimal for the Iris dataset

### Step 5: KNN Training

* **What**: Memorize all training points, classify by majority vote of K neighbors
* **Why**: Simple, intuitive, no training phase needed
* **Distance**: Euclidean distance by default

### Step 6: Evaluation

* **Accuracy**: Overall correctness
* **Precision**: Of predicted positives, how many are correct?
* **Recall**: Of actual positives, how many did we find?
* **F1-Score**: Harmonic mean of Precision & Recall

---

## 📊 Expected Results

| Metric        | Value     |
| ------------- | --------- |
| Accuracy      | ~93.33%   |
| Macro F1      | ~0.93     |
| Optimal K     | 5         |
| Training Time | <1 second |

---

## 🐛 Troubleshooting

| Problem             | Solution                    |
| ------------------- | --------------------------- |
| FileNotFoundError   | Run from `proj2/` directory |
| ModuleNotFoundError | Activate venv               |
| Low accuracy        | Check data integrity        |
| Different K optimal | Ensure `random_state=42`    |

---

## 🎓 Learning Checklist

* [ ] I understand why we need feature scaling
* [ ] I can explain the Elbow Method
* [ ] I know the difference between Precision and Recall
* [ ] I understand why F1 is better than Accuracy alone
* [ ] I can interpret a Confusion Matrix

---


