# 🧠 Algorithms Explained

**DecodeLabs Internship | Batch 2026**

---

## 1. K-Nearest Neighbors (KNN)

### What is it?

KNN is a **lazy learning** algorithm. It does not build a model during training. Instead, it stores the training data and classifies new samples based on the nearest neighbors.

### How it Works

1. Calculate distance to all training points
2. Select the K nearest points
3. Use majority voting to determine the class

### Example (K = 5)

```text
New flower: [5.1, 3.5, 1.4, 0.2]

Nearest neighbors:
setosa
setosa
setosa
versicolor
setosa

Prediction: setosa
```

### Why K Matters

| K   | Behavior      | Risk               |
| --- | ------------- | ------------------ |
| 1   | Very specific | Overfitting        |
| 5   | Balanced      | ✅ Optimal for Iris |
| 100 | Very general  | Underfitting       |

### Distance Metric

Euclidean Distance:

```text
d = √((x₁-x₂)² + (y₁-y₂)² + ...)
```

---

## 2. StandardScaler (Feature Scaling)

### The Problem

Different features have different numerical ranges:

* Sepal Length: 4.3–7.9 cm
* Petal Width: 0.1–2.5 cm

Without scaling, larger-range features dominate distance calculations.

### The Solution

Standardize each feature:

```text
z = (x - μ) / σ
```

Where:

* μ = mean
* σ = standard deviation

### Before vs After

| Feature      | Raw Range | Scaled Range |
| ------------ | --------- | ------------ |
| Sepal Length | 4.3–7.9   | ~ -2 to +2   |
| Sepal Width  | 2.0–4.4   | ~ -2 to +2   |
| Petal Length | 1.0–6.9   | ~ -2 to +2   |
| Petal Width  | 0.1–2.5   | ~ -2 to +2   |

---

## 3. Confusion Matrix

### Structure

```text
                 Predicted
              Setosa Versi Virgi

Actual Setosa   TP      0      0
Actual Versi     0     TP     FP
Actual Virgi     0     FN     TP
```

### Terminology

| Term | Meaning        |
| ---- | -------------- |
| TP   | True Positive  |
| FP   | False Positive |
| FN   | False Negative |
| TN   | True Negative  |

---

## 4. Precision, Recall, and F1-Score

### Precision

Measures prediction quality.

```text
Precision = TP / (TP + FP)
```

Question:

> Of all predicted positives, how many were correct?

### Recall

Measures detection completeness.

```text
Recall = TP / (TP + FN)
```

Question:

> Of all actual positives, how many did we find?

### F1-Score

Balances Precision and Recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Why use F1?

* Balances Precision and Recall
* Penalizes extreme values
* Useful for comparing models

---

## 5. Elbow Method

### Purpose

Find the best value of K.

### Process

1. Train KNN using K = 1 to 30
2. Measure error rate
3. Plot error vs K
4. Find the elbow point

### Visual

```text
Error
 ↑
 │\
 │ \
 │  \
 │   \____
 │        \____
 └────────────────→ K
         5
```

The elbow represents the best trade-off between complexity and generalization.

---

## 📚 References

* Scikit-learn KNN Documentation
* StandardScaler Documentation
* Precision and Recall Theory
* Machine Learning Model Evaluation Techniques
