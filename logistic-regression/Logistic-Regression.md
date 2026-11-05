# Coding Assignment: Logistic Regression

Build a classifier, estimate probabilities, and investigate how the decision threshold affects predictions. Justify your modeling and experimental choices. Use the same dataset where applicable; the multiclass comparison requires at least three classes.

## Sample datasets

Choose one of the following real datasets, or another real classification dataset. **Binary and multiclass targets are both allowed.** 

### Multiclass datasets

| Dataset | Prediction target | Size |
|---|---|---|
| [Iris](https://archive.ics.uci.edu/dataset/53/iris) | Iris species from flower measurements (3 classes) | 150 examples, 4 features |
| [Wine](https://archive.ics.uci.edu/dataset/109/wine) | Wine cultivar from chemical measurements (3 classes) | 178 examples, 13 features |
| [Seeds](https://archive.ics.uci.edu/dataset/236/seeds) | Wheat variety from kernel measurements (3 classes) | 210 examples, 7 features |
| [Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | Bean variety from grain shape measurements (7 classes) | 13,611 examples, 16 features |
| [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) | Handwritten digit, 0–9, from image block counts (10 classes) | 5,620 examples across supplied train/test files, 64 features |
| [Dermatology](https://archive.ics.uci.edu/dataset/33/dermatology) † | Skin disease type from clinical and tissue measurements (6 classes) | 366 examples, 34 features |
| [Heart Disease: Cleveland](https://archive.ics.uci.edu/dataset/45/heart+disease) † | Original diagnosis code, 0–4 (5 classes); use the processed Cleveland file | 303 examples, 13 features |

### Binary datasets

| Dataset | Prediction target | Size |
|---|---|---|
| [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | Malignant or benign diagnosis from cell measurements | 569 examples, 30 features |
| [Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) | Banknote class from image statistics | 1,372 examples, 4 features |
| [Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere) | Good or bad radar returns from signal measurements | 351 examples, 34 features |
| [Sonar: Mines vs. Rocks](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) | Metal cylinders or rocks from sonar measurements | 208 examples, 60 features |
| [Rice (Cammeo and Osmancik)](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) | Cammeo or Osmancik rice variety from grain measurements | 3,810 examples, 7 features |
| [Raisin](https://archive.ics.uci.edu/dataset/850/raisin) | Kecimen or Besni raisin variety from shape measurements | 900 examples, 7 features |
| [Haberman's Survival](https://archive.ics.uci.edu/dataset/43/haberman+s+survival) | Survival for at least five years after surgery or death within five years | 306 examples, 3 features |
| [Blood Transfusion Service Center](https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center) | Whether a donor donated blood in March 2007 | 748 examples, 4 features |
| [Breast Cancer Coimbra](https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra) | Breast cancer patient or healthy control from clinical measurements | 116 examples, 9 features |
| [HTRU2](https://archive.ics.uci.edu/dataset/372/htru2) | Pulsar or non-pulsar candidate from radio-signal statistics | 17,898 examples, 8 features |
| [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) † | Spam or non-spam email from word and character statistics | 4,601 examples, 57 features |
| [Fertility](https://archive.ics.uci.edu/dataset/244/fertility) † | Normal or altered semen quality from health and lifestyle attributes | 100 examples, 9 features |
| [Mammographic Mass](https://archive.ics.uci.edu/dataset/161/mammographic+mass) † | Benign or malignant mass; exclude the BI-RADS assessment column | 961 examples, 4 usable predictors after exclusion |
| [Statlog (Heart)](https://archive.ics.uci.edu/dataset/145/statlog+heart) † | Presence or absence of heart disease from clinical attributes | 270 examples, 13 features |
| [Hepatitis](https://archive.ics.uci.edu/dataset/46/hepatitis) † | Recorded survival outcome: live or die | 155 examples, 19 features |
| [Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) † | Annual income above $50,000 or at most $50,000 | 48,842 examples across supplied train/test files, 14 features |
| [Credit Approval](https://archive.ics.uci.edu/dataset/27/credit+approval) † | Positive or negative credit-application outcome from anonymized attributes | 690 examples, 15 features |
| [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) † | Good or bad credit risk | 1,000 examples, 20 original features in `german.data` |
| [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) † | Default or no default on the next payment | 30,000 examples, 23 features excluding ID |
| [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) † | Term-deposit subscription or no subscription; exclude call duration for predictions made before a call | 45,211 examples in `bank-full.csv`, 16 features before exclusion |
| [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) † | Whether a recorded browsing session ended in a purchase | 12,330 examples, 17 features |
| [Seismic Bumps](https://archive.ics.uci.edu/dataset/266/seismic+bumps) † | Whether a high-energy seismic bump occurred in the next mining shift | 2,584 examples, 18 features |

† These options need additional preparation, such as handling missing values or encoding categorical features.

---

## Problem 1 — Logistic Regression from First Principles (40 points)

Define your prediction task and justify the data preparation and evaluation split. Prevent information from the evaluation data from influencing training.

Implement logistic regression with an intercept using NumPy, without a library model or optimizer. Connect your implementation to the probability model, log loss, and gradient-based learning. Choose and justify the optimization settings.

For multiclass data, implement a one-vs-rest extension and explain its training and prediction rules.

Plot the training loss and report the initial, final, and validation losses. Define the loss you report and use evidence from your run to assess convergence.

---

## Problem 2 — Model Comparison and Thresholds (60 points)

### A. Compare multiclass models

On a real dataset with at least three classes, compare scikit-learn's **OneVsRestClassifier with logistic regression** and **multinomial logistic regression**. Design a fair comparison using the same data splits and justify your settings.

Report validation **accuracy, macro F1, and log loss** in one table. Show both models' probabilities and predicted class for one validation example. Briefly explain how the models differ and compare their results.

### B. Choose a threshold

Use scikit-learn logistic regression on your original dataset. Define the positive outcome; for multiclass data, evaluate one chosen class versus the rest while retaining all classes during model training.

Define and justify an error-cost function before comparing thresholds. Investigate several thresholds on validation data and report **precision, recall, F1, and cost**. Select a threshold using your cost criterion and explain the trade-off.

Keep the model, threshold, and cost function fixed for final test evaluation. Report the **confusion matrix, accuracy, precision, recall, F1, and cost**.

For multiclass data, also report the full multiclass confusion matrix, accuracy, and macro F1.

Briefly discuss how lowering the threshold changes errors, whether it changes probabilities, and why majority-class accuracy can be misleading. Include the test accuracy of a baseline that always predicts the training majority class, using Part B's positive/negative labels.

---

## Submission

Submit one Jupyter notebook for each problem or one for both problems that runs from beginning to end. Include your code, the loss plot, result tables, and short answers. 
