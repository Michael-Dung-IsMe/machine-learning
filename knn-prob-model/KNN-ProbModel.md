# Coding Assignments: KNN and Probabilistic Models

## Sample datasets
Choose one dataset below, or another real classification dataset with numerical features. Binary and multiclass targets are both allowed. Choose one dataset for each assignment and use it throughout that assignment. You may reuse the same dataset for both assignments.

| Dataset | Prediction target | Size |
|---|---|---|
| [Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | Bean variety from grain shape measurements (7 classes) | 13,611 examples, 16 features |
| [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) | Handwritten digit, 0–9, from image block counts (10 classes) | 5,620 examples across supplied train/test files, 64 features |
| [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | Malignant or benign diagnosis from cell measurements | 569 examples, 30 features |
| [Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) | Banknote class from image statistics | 1,372 examples, 4 features |
| [Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere) | Good or bad radar returns from signal measurements | 351 examples, 34 features |
| [Rice (Cammeo and Osmancik)](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) | Cammeo or Osmancik rice variety from grain measurements | 3,810 examples, 7 features |
| [Raisin](https://archive.ics.uci.edu/dataset/850/raisin) | Kecimen or Besni raisin variety from shape measurements | 900 examples, 7 features |
| [Haberman's Survival](https://archive.ics.uci.edu/dataset/43/haberman+s+survival) | Survival for at least five years after surgery or death within five years | 306 examples, 3 features |
| [Blood Transfusion Service Center](https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center) | Whether a donor donated blood in March 2007 | 748 examples, 4 features |
| [HTRU2](https://archive.ics.uci.edu/dataset/372/htru2) | Pulsar or non-pulsar candidate from radio-signal statistics | 17,898 examples, 8 features |

---

## Assignment 1: K-Nearest Neighbors

Build and assess a KNN classifier. Justify your choices using evidence from your experiments.

### Task

Implement KNN classification using NumPy, without a library classifier or neighbor-search routine. Support two distance measures and both uniform and distance-weighted voting. State how you handle ties and zero distances, and check your predictions against scikit-learn.

Investigate how scaling, the number of neighbors, distance, and weighting affect performance. Use cross-validation to select a configuration, keeping preprocessing within training folds and reserving a test set for final assessment. Justify your primary metric.

Plot training and validation performance against the number of neighbors. 

Use one held-out example to explain the prediction through its neighbors. Briefly discuss an observed limitation of your model.

### Submission

Submit one Jupyter notebook that runs from beginning to end, with reproducible data access, random seeds, code, plots, result tables, and short explanations.

---

## Assignment 2: Probabilistic Models

Build and assess Gaussian classifiers. Investigate how their assumptions affect predictions and probabilities.

### Task

Implement Gaussian naive Bayes using NumPy, without a library classifier or probability-density routine. Estimate parameters by maximum likelihood and return normalized class probabilities. Address numerical stability and check your results against scikit-learn.

Compare your model with a Gaussian classifier using a separate full covariance matrix for each class. Use cross-validation for model selection, keeping preprocessing and tuning within training folds and reserving a test set for final assessment.

Explain one held-out prediction using its priors, likelihoods, and posterior probabilities.

Fix each model's configuration before reporting test accuracy, macro F1, log loss, and confusion matrices. 

Use your results to discuss conditional independence and whether better classification also produces better probability estimates.

### Submission

Submit one Jupyter notebook that runs from beginning to end, with reproducible data access, random seeds, code, result tables, and short explanations.
