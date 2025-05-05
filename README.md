# 🧠 Pattern Recognition and Machine Learning Assignment (2024–2025)

📚 *Course:* Pattern Recognition and Machine Learning  
🏛️ *Faculty:* AUTh - School of Electrical and Computer Engineering  
📅 *Semester:* 7th Semester, 2024–2025

---

## 🔍 Overview
This repository contains our implementation for the **Pattern Recognition & Machine Learning** assignment for 2024. The project is divided into four parts (**A, B, C, and D**), covering classification methods, probabilistic models, and machine learning techniques.


## 📁 Repository Structure
```
├── Team8-AC.ipynb         # Jupyter Notebook for Parts A, B, and C
├── Team8-D.ipynb          # Jupyter Notebook for Part D
├── PR_Assignment_2024.pdf # Assignment description and requirements
├── labels8.npy            # Predicted labels from Part D
└── README.md              # This documentation
```

---

## 🧪 Assignment Breakdown

### 🅰️ Part A: Maximum Likelihood Classifier
- **🎯 Objective:** Estimate parameters \(\theta_1\) and \(\theta_2\) using Maximum Likelihood Estimation (MLE).
- **⚙️ Implementation:**
  - Compute and visualize log-likelihood functions.
  - Implement classifier function \( g(x) \).
  - Evaluate classification accuracy and interpret results.

### 🅱️ Part B: Bayesian Estimation Classifier
- **🎯 Objective:** Improve parameter estimation using Bayesian methods.
- **⚙️ Implementation:**
  - Compute posterior distributions.
  - Visualize posterior densities.
  - Implement Bayesian decision function \( h(x) \).
  - Compare to MLE performance.

### 🌳 Part C: Decision Tree & Random Forest (Iris Dataset)
- **🎯 Objective:** Build interpretable models for the Iris dataset.
- **⚙️ Implementation:**
  - Use `DecisionTreeClassifier` from `sklearn` with varying depths.
  - Implement and tune `RandomForestClassifier`.
  - Visualize decision boundaries and analyze model strengths.

### 🧩 Part D: Custom Classification Model for Large Dataset
- **🎯 Objective:** Classify data from `datasetTV.csv` and evaluate generalization on `datasetTest.csv`.

#### 🧼 Preprocessing & Feature Engineering:
- Applied normalization and data cleaning.
- Reduced feature set to optimize performance.

#### 🤖 Model Development:
- Implemented and evaluated:
  - ✅ **Support Vector Machine (SVM)** — *from scratch*
    - Custom implementation using hinge loss and gradient descent.
    - Supported linear and RBF kernels.
  - 🌲 **Random Forest** — *via `sklearn`*
    - Tuned estimators, depth, and sampling strategies.
  - 🔍 Also explored KNN and Logistic Regression for benchmarking.

#### 🎛️ Hyperparameter Tuning:
- Used **RandomizedSearchCV** to explore parameters efficiently.
  - Tuned: `C`, `kernel`, `gamma` (SVM) and `n_estimators`, `max_depth` (RF).
- 5-fold cross-validation for robust selection.

#### 📊 Training & Evaluation:
- Trained over 25 epochs with 3 trials per config.
- Metrics used: **accuracy, precision, recall, F1-score**.
- Final accuracy: **🟢 84%**
  - F1 > 0.90 for classes 0 and 2.
  - Lower scores for classes 1 and 4.

#### ⚠️ Challenges:
- Class overlap and imbalance led to misclassifications (e.g., class 3 ↔ 4).
- Confusion matrix revealed class-specific difficulties.

#### 💾 Output:
- Final predictions saved as `labels8.npy`.

#### ✅ Conclusion:
- Strong generalization and performance.
- Improvement opportunities include advanced feature engineering and class balancing.

---

## 💡 Suggestions for Further Improvements
- 🔄 **Class Balancing Techniques:** Use SMOTE, class weighting, or data augmentation to address imbalance.
- 🧠 **Ensemble Strategies:** Combine models using stacking or voting for more robust predictions.
- 📈 **Learning Curves:** Add training/validation curves to monitor overfitting.
- 🎯 **Explainability Tools:** Use SHAP or LIME for model interpretability.
- 🌐 **Deployability:** Wrap the model in a Flask API or Streamlit app for interactive testing.

---
