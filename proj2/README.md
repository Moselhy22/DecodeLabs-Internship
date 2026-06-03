# 🌸 Project 2: Iris Classification Using AI

**DecodeLabs Internship | Batch 2026**

---

## 📋 Overview

A complete machine learning pipeline that classifies Iris flowers into three species using the K-Nearest Neighbors (KNN) algorithm.

**Dataset:** Iris Dataset (150 samples, 3 classes, 4 features)

**Algorithm:** K-Nearest Neighbors (K=5)

**Performance:** ~93.33% Accuracy | ~0.93 Macro F1

---

## 🖼️ Pipeline Architecture

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   INPUT     │ →  │   PROCESS   │ →  │   OUTPUT    │
│ Iris Data   │    │ Scale + KNN │    │ Metrics     │
│ 150 Rows    │    │ Elbow K=5   │    │ Confusion   │
│ 4 Features  │    │ 80/20 Split │    │ Matrix + F1 │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## 🚀 Quick Start

```bash
cd proj2

source venv/bin/activate

python main.py
```

---

## 📊 Results

| Metric          | Value  |
| --------------- | ------ |
| Accuracy        | 93.33% |
| Macro Precision | 94.44% |
| Macro Recall    | 93.33% |
| Macro F1-Score  | 93.27% |
| Optimal K       | 5      |

### Confusion Matrix

|                   | Predicted Setosa | Predicted Versicolor | Predicted Virginica |
| ----------------- | ---------------- | -------------------- | ------------------- |
| Actual Setosa     | 10               | 0                    | 0                   |
| Actual Versicolor | 0                | 10                   | 0                   |
| Actual Virginica  | 0                | 2                    | 8                   |

---

## 📁 Project Structure

```text
proj2/
├── data/iris.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── model.py
│   └── evaluator.py
├── notebooks/
│   └── 01_eda.ipynb
├── docs/
│   ├── PROJECT_GUIDE.md
│   └── ALGORITHM_EXPLAINED.md
├── main.py
└── requirements.txt
```

---

## 🛠️ Technologies Used

* Python 3.10
* scikit-learn
* pandas
* numpy
* matplotlib
* seaborn
* jupyter notebook

---

## 📖 Documentation

* PROJECT_GUIDE.md → Complete project walkthrough
* ALGORITHM_EXPLAINED.md → KNN, Scaling, Metrics, and Elbow Method

---

## 🎓 Key Concepts Learned

* Feature Scaling with StandardScaler
* Train-Test Split with Stratification
* Hyperparameter Tuning
* Elbow Method
* Confusion Matrix Analysis
* Precision, Recall, and F1-Score

---

## 👤 Author

**Moselhy**

DecodeLabs Internship Batch 2026

---

## 📄 License

This project is part of the DecodeLabs Industrial Training Kit.
