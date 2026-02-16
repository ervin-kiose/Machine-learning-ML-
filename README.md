# Machine Learning — Image Classification

A collection of supervised machine learning models for **binary image classification**, implemented in Python using PyTorch and scikit-learn. Each notebook trains a different algorithm on the same dataset and evaluates it with a consistent set of metrics.

---

## Task

Binary classification of images into two categories:
- `1` — Pleasant
- `0` — Unpleasant

Images are resized to **64×64 RGB**. Each notebook independently loads the dataset, trains a model, evaluates on a validation set, and outputs test predictions to `submission.csv`.

---

## Models Implemented

| Notebook | Algorithm | Framework |
|---|---|---|
| `CNN.ipynb` | Convolutional Neural Network | PyTorch |
| `MLP.ipynb` | Multi-Layer Perceptron | PyTorch |
| `SVM.ipynb` | Support Vector Machine (grid search) | scikit-learn |
| `EnsembleLearing.ipynb` | Random Forest, Bagging SVM, AdaBoost | scikit-learn |
| `kNN.ipynb` | k-Nearest Neighbors (cosine distance) | scikit-learn |
| `LogisticRegression.ipynb` | Logistic Regression | scikit-learn |

---

## Model Details

### CNN (`CNN.ipynb`)
- Architecture: 4 Conv layers (32→64→128→256 filters) + 2 FC layers
- Batch normalization + Dropout for regularization
- Optimizer: Adam (lr=0.0001), Loss: CrossEntropyLoss
- Training: 15 epochs

### MLP (`MLP.ipynb`)
- Architecture: Input → [512, 256] → 2 output units
- Activation: PReLU (configurable: relu/sigmoid/tanh/prelu)
- Optimizer: Adam (lr=0.0001), Loss: CrossEntropyLoss
- Training: 20 epochs, batch size 64

### SVM (`SVM.ipynb`)
- Grid search over 8 configurations: 2 kernels (linear, RBF) × 4 C values (0.1, 1, 10, 100)
- Best model selected by F1 score on validation set
- Comparative ROC curves for all configurations

### Ensemble Methods (`EnsembleLearing.ipynb`)
- **Random Forest**: 3 configurations (50, 100, 200 trees, varying depths)
- **Bagging + Linear SVM**: 20 estimators
- **AdaBoost**: Decision Tree base, 200 estimators, max_depth=3

### kNN (`kNN.ipynb`)
- k=3, cosine distance metric
- Flattened image vectors (64×64×3 = 12,288 features)

### Logistic Regression (`LogisticRegression.ipynb`)
- C=0.5, solver=lbfgs, max_iter=1000
- Normalized input features to [0, 1]

---

## Evaluation

Every model is evaluated with the same metrics on a held-out validation set (80/20 split, random_state=42):

- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- ROC Curve (AUC)

---

## Tech Stack

| Library | Purpose |
|---|---|
| PyTorch | CNN and MLP implementation |
| scikit-learn | SVM, ensemble methods, kNN, logistic regression |
| OpenCV | Image loading and resizing |
| pandas | Data handling |
| matplotlib | Visualization (ROC curves, confusion matrices) |

---

## Project Structure

```
Machine-Learning/
├── CNN.ipynb                # Convolutional Neural Network
├── MLP.ipynb                # Multi-Layer Perceptron
├── SVM.ipynb                # Support Vector Machine (grid search)
├── EnsembleLearing.ipynb    # Random Forest, Bagging, AdaBoost
├── kNN.ipynb                # k-Nearest Neighbors
└── LogisticRegression.ipynb # Logistic Regression
```

---

## Academic Context

Built as a university project for the **Machine Learning** course at the
**Department of Computer Engineering, University of Ioannina, Greece**.

Key concepts implemented:
- Supervised binary classification
- Feature extraction from raw images (flattening, normalization)
- Deep learning with PyTorch (CNN, MLP)
- Classical ML with scikit-learn (SVM, kNN, Logistic Regression, Ensembles)
- Hyperparameter search and model comparison
- Evaluation metrics: F1, AUC-ROC, confusion matrix
