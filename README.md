# Pattern Recognition and Machine Learning Assignment (2024–2025)

Assignment for the "Pattern Recognition and Machine Learning" Course  
Faculty of Engineering, AUTh  
School of Electrical and Computer Engineering  
Electronics and Computers Department, 7th Semester  
2024–2025

## Overview
This repository contains our implementation for the **Pattern Recognition & Machine Learning** assignment for 2024. The project is divided into four parts (A, B, C, and D), covering different classification methods, probability estimations, and machine learning models.

## Repository Structure
```
├── Team8-AC.ipynb         # Jupyter Notebook for Parts A, B, and C
├── Team8-D.ipynb          # Jupyter Notebook for Part D
├── PR_Assignment_2024.pdf # Assignment description and requirements
├── labels8.npy            # Predicted labels from Part D
└── README.md              # This documentation
```

## Assignment Breakdown

### Part A: Maximum Likelihood Classifier
- **Objective:** Estimate parameters \(\theta_1\) and \(\theta_2\) for two classes using the Maximum Likelihood Estimation (MLE) method.
- **Implementation:**
  - Compute log-likelihood functions.
  - Visualize the likelihoods.
  - Implement a classifier using the function \( g(x) \).
  - Analyze classification performance.

### Part B: Bayesian Estimation Classifier
- **Objective:** Use Bayesian estimation to refine parameter estimation.
- **Implementation:**
  - Compute posterior distributions.
  - Visualize posterior densities.
  - Implement a Bayesian classifier using \( h(x) \).
  - Compare results with MLE.

### Part C: Decision Tree and Random Forest Classifiers
- **Objective:** Implement classifiers for the **Iris dataset**.
- **Implementation:**
  - Use `DecisionTreeClassifier` from `sklearn`.
  - Experiment with different tree depths and analyze performance.
  - Implement a `RandomForestClassifier` using bootstrap samples.
  - Compare decision boundaries of both classifiers.

### Part D: Custom Classification Model for Large Dataset
- **Objective:** Develop a robust classification algorithm for `datasetTV.csv` and evaluate it on `datasetTest.csv`.

- **Preprocessing & Feature Engineering:**
  - Applied data cleaning and normalization to prepare features.
  - Selected a reduced, informative feature set to improve model efficiency.

- **Model Development:**
  - Implemented and evaluated several classification models:
    - **Support Vector Machine (SVM):**  
      - Developed **from scratch**, without using libraries like `sklearn`.
      - Used a custom implementation of the SVM algorithm with hinge loss and gradient descent optimization.
      - Incorporated kernel tricks (e.g., linear and RBF kernels) during experimentation.
    - **Random Forest Classifier:**  
      - Utilized the `RandomForestClassifier` from **scikit-learn**.
      - Configured the number of estimators, maximum tree depth, and feature sampling strategies.
    - Additional experiments were conducted using **K-Nearest Neighbors** and **Logistic Regression** via `sklearn` for baseline comparison.

  - **Hyperparameter Tuning:**
    - Employed **RandomizedSearchCV** from `sklearn.model_selection` to efficiently explore hyperparameter spaces.
    - For SVM, parameters like **C**, **kernel type**, and **gamma** (for RBF) were varied.
    - For Random Forest, tuned parameters included **n_estimators**, **max_depth**, **min_samples_split**, and **max_features**.
    - Cross-validation (5-fold) was used within each random search iteration to ensure model generalization.

- **Training & Evaluation:**
  - Trained over **25 epochs**, with **3 executions per trial** to ensure consistent results.
  - Evaluated using accuracy, precision, recall, F1-score, and confusion matrices.
  - Achieved **84% overall accuracy** with high performance for classes **0** and **2** (F1 > 0.90), and lower scores for classes **1** and **4**.

- **Challenges:**
  - Noted confusion between classes **3** and **4**, with 14 misclassifications.
  - Imbalance and feature overlap in certain classes affected precision.

- **Output:**
  - Saved final test predictions to `labels8.npy`.

- **Conclusion:**
  - The developed model shows strong generalization and robustness.
  - Further improvements could involve class rebalancing and deeper feature engineering.
