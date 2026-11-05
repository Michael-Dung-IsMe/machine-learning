# Coding Assignment: SVM for Classification and Regression

Train and compare linear and kernel support vector models. Explain how feature scaling and model parameters affect their predictions.

## Sample datasets

Choose **one real-world classification or regression dataset** with **at least 1,000 examples** and numerical features. Complete both problems for your chosen task. Binary and multiclass classification are both allowed. Use observations from this dataset for all experiments and illustrations.

### Classification datasets

| Dataset                                                                                                                          | Classes | Size                                                          |
| -------------------------------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------- |
| [Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication)                                        | 2       | 1,372 examples, 4 features                                    |
| [Rice](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik)                                                          | 2       | 3,810 examples, 7 features                                    |
| [HTRU2](https://archive.ics.uci.edu/dataset/372/htru2)                                                                            | 2       | 17,898 examples, 8 features                                   |
| [Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)                                                              | 7       | 13,611 examples, 16 features                                  |
| [Pen-Based Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits) | 10      | 10,992 examples across supplied train/test files, 16 features |

### Regression datasets

| Dataset                                                                                               | Prediction target                            | Size                                               |
| ----------------------------------------------------------------------------------------------------- | -------------------------------------------- | -------------------------------------------------- |
| [Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) | Compressive strength (MPa)                   | 1,030 examples, 8 features                         |
| [Wine Quality — White Wine](https://archive.ics.uci.edu/dataset/186/wine+quality)                     | Quality score, treated as a numerical target | 4,898 examples, 11 features                        |
| [Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)                       | Sound pressure level (dB)                    | 1,503 examples, 5 features                         |
| [Abalone](https://archive.ics.uci.edu/dataset/1/abalone)                                               | Ring count, treated as a numerical target    | 4,177 examples, 7 numerical features; exclude Sex  |
| [Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)       | Electrical power output (MW)                 | 9,568 examples, 4 features; use one worksheet only |

Use the same dataset and training/validation/test partitions for both problems. Fit preprocessing on training data only and reserve the test set for final evaluation.

---

## Problem 1 — Linear SVM or SVR (40 points)

Train a linear model using scikit-learn's **SVC(kernel="linear")** for classification or **SVR(kernel="linear")** for regression. Standardize the features and investigate how **C** affects training and validation performance and the number of support vectors. Keep other parameters at their default values.

Complete the illustration for your chosen task:

- **Classification:** Select two features and two classes from your dataset for a separate binary illustration. Fit a linear SVM on the corresponding training observations. For selected observations, calculate decision scores, signed distances and hinge losses using NumPy. Plot the boundary, margin lines and support vectors. Explain hard-margin and soft-margin SVMs.
- **Regression:** Select one feature from your dataset for a separate illustration and fit a linear SVR on the training observations. Calculate predictions from the learned parameters **w** and **b** and compute prediction errors using NumPy. Plot the fitted line, observed values and support vectors. Explain how changing **C** affects the fit.

Use the fitted models' standardized feature coordinates for these calculations. The reduced-feature illustrations supplement the main comparison, which uses all selected predictors and all target classes, where applicable.

You may use library models for training; implementing an optimizer is not required.

---

## Problem 2 — Kernel Models and Comparison (60 points)

Train **polynomial** and **RBF** models using **SVC** or **SVR**, according to your task. Compare them with the linear model from Problem 1.

Use validation data or cross-validation to select **C**, polynomial **degree**, and RBF **gamma**. Keep other model parameters at their default values. Explain how these parameters affect model complexity and identify signs of underfitting or overfitting. If you scale the regression target, fit that transformation on training data only and report errors in the original units.

Reuse the real observations and feature selection from the illustration in Problem 1 to visualize linear and RBF models. Fit both models on the same training observations. Show decision boundaries for classification or fitted curves for regression, and highlight support vectors. Briefly explain the kernel trick.

Compare the models using the same evaluation procedure and task-appropriate metrics. Recommend a model based on validation results, freeze all settings, and evaluate the models on the test set. Interpret the results and discuss the limitations of your recommendation.

---

## Submission

Submit **one runnable Jupyter notebook per problem**. Include your code, key plots, a comparison table and concise explanations. State your experimental questions before running experiments, justify your evaluation metrics, and provide dataset sources and random seeds so your work can be reproduced.
