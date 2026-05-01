# Pulsar Detection using ML with Synthetic Data Augmentation

Binary classification of pulsar signals from radio telescope data using the HTRU2 dataset. Addresses severe class imbalance through multivariate Gaussian synthetic data generation.

---

## Problem

Pulsars are rapidly rotating neutron stars that emit regular radio signals. Detecting them is critical in astronomy, but challenging — real pulsar signals are rare, representing less than 10% of observations in the HTRU2 dataset. This severe class imbalance causes standard classifiers to learn noise patterns rather than pulsar characteristics, resulting in high accuracy but poor recall.

---

## Dataset

**HTRU2 — High Time Resolution Universe Survey**
- 17,898 candidates: 16,259 non-pulsars, 1,639 pulsars (9:1 imbalance)
- 8 numerical features derived from integrated pulse profiles and DM-SNR curves
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/372/htru2)

**Features:**
- Mean, standard deviation, excess kurtosis, skewness of the integrated pulse profile
- Mean, standard deviation, excess kurtosis, skewness of the DM-SNR curve

---

## Approach

### 1. Baseline Classification
Six classifiers evaluated on the imbalanced dataset:
- Logistic Regression
- Linear SVM and RBF SVM
- Random Forest
- Gradient Boosting
- k-Nearest Neighbors

### 2. Synthetic Data Augmentation
To address class imbalance, 5,000 synthetic pulsar samples were generated using a **multivariate Gaussian distribution** fitted to real pulsar statistics:

```python
mu  = X_pulsar.mean().values        # Mean vector (8-dim)
cov = np.cov(X_pulsar.T)            # Covariance matrix (8x8)
X_synthetic = np.random.multivariate_normal(mu, cov, size=5000)
```

This preserves the statistical structure of real pulsar signals — correlations between features, feature distributions — rather than simply oversampling existing examples.

---

## Results

### Before Augmentation

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 96.96% | 0.847 | 0.973 |
| Linear SVM | 97.49% | 0.870 | 0.971 |
| RBF SVM | 97.18% | 0.856 | 0.966 |
| Random Forest | 98.18% | 0.898 | 0.969 |
| Gradient Boosting | 98.13% | 0.895 | 0.977 |
| k-Nearest Neighbors | 98.04% | 0.888 | 0.952 |

**Key observation:** High accuracy masks poor pulsar recall. Gradient Boosting achieves 98.1% accuracy but only 0.895 F1 — the model is predicting non-pulsars most of the time.

Confusion matrix (Gradient Boosting, before augmentation):
- True Pulsars correctly identified: **286**
- Pulsars missed: **42**

### After Augmentation

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 96.22% | 0.935 | 0.987 |
| Linear SVM | 96.27% | 0.935 | 0.987 |
| RBF SVM | 98.03% | 0.966 | 0.993 |
| Random Forest | 98.30% | 0.970 | 0.995 |
| Gradient Boosting | 98.30% | 0.970 | 0.995 |
| k-Nearest Neighbors | 97.55% | 0.957 | 0.983 |

**Key observation:** F1-score improves significantly across all models. True pulsar detections jump from 286 to **1,275** while false positives remain at 25 — the model now learns to identify pulsars rather than defaulting to non-pulsar predictions.

Confusion matrix (Gradient Boosting, after augmentation):
- True Pulsars correctly identified: **1,275**
- Pulsars missed: **53**

---

## Key Finding

**Accuracy is a misleading metric for rare signal detection.** Before augmentation, models achieve 97-98% accuracy while missing most pulsars. After synthetic augmentation, F1 improves from 0.895 to 0.970 and ROC-AUC from 0.977 to 0.996 — without sacrificing accuracy.

This has direct implications for any astronomical signal detection task where the signal of interest is rare compared to noise or interference.

---

## Tech Stack

- Python, NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

---

## Author

**Pooja Devi V**
