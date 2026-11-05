# Coding Assignment: Decision Trees, Random Forest and Gradient Boosting

Build and interpret decision trees, investigate how ensembles combine trees, and compare their generalization performance. Justify your modeling and experimental choices. Use the same classification dataset throughout the model comparison. **Binary and multiclass targets are both allowed.**

## Sample datasets

Choose one of the following public classification datasets, or another classification dataset with **at least 1,000 examples**. Every option below meets this minimum. For this assignment, select observed data with complete numerical predictors; the list also includes benchmarks with categorical or simulated data for reference.

### Multiclass datasets

| Dataset | Prediction target | Size |
|---|---|---|
| [Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | Bean variety from grain measurements (7 classes) | 13,611 examples, 16 features |
| [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) | Handwritten digit, 0–9, from image block counts (10 classes) | 5,620 examples across supplied train/test files, 64 features |
| [Pen-Based Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits) | Handwritten digit from pen coordinates (10 classes) | 10,992 examples across supplied train/test files, 16 features |
| [Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition) | Uppercase English letter from image statistics (26 classes) | 20,000 examples, 16 features |
| [Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation) † | Car acceptability in a decision-model benchmark (4 classes) | 1,728 examples, 6 categorical features |
| [Statlog (Landsat Satellite)](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite) | Land-cover class from satellite pixel neighborhoods (6 observed classes) | 6,435 examples across supplied train/test files, 36 features |
| [Page Blocks Classification](https://archive.ics.uci.edu/dataset/78/page+blocks+classification) | Document block type, such as text or graphics (5 classes) | 5,473 examples, 10 features |
| [Image Segmentation](https://archive.ics.uci.edu/dataset/50/image+segmentation) | Outdoor image-region class (7 classes) | 2,310 examples across supplied train/test files, 19 features |

### Binary datasets

| Dataset | Prediction target | Size |
|---|---|---|
| [Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) | Banknote class from image statistics | 1,372 examples, 4 features |
| [Rice (Cammeo and Osmancik)](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) | Rice variety from grain measurements | 3,810 examples, 7 features |
| [HTRU2](https://archive.ics.uci.edu/dataset/372/htru2) | Pulsar or non-pulsar candidate from radio-signal statistics | 17,898 examples, 8 features |
| [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) | Spam or non-spam email from word and character statistics | 4,601 examples, 57 features |
| [MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope) | Simulated gamma signal or hadron background from telescope image statistics | 19,020 examples, 10 features |
| [Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) † | Annual income above $50,000 or at most $50,000 | 48,842 examples across supplied train/test files, 14 features |
| [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) † | Good or bad credit risk | 1,000 examples, 20 original features in `german.data` |
| [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) † | Default or no default on the next payment | 30,000 examples, 23 features excluding ID |
| [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) † | Term-deposit subscription; exclude call duration for predictions made before a call | 45,211 examples in `bank-full.csv`, 15 predictors after excluding duration |
| [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) † | Whether a browsing session ended in a purchase | 12,330 examples, 17 features |
| [Seismic Bumps](https://archive.ics.uci.edu/dataset/266/seismic+bumps) † | Whether a high-energy seismic bump occurred in the next mining shift | 2,584 examples, 18 features |
| [Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom) † | Edible or poisonous class from mushroom descriptions | 8,124 examples, 22 categorical features |
| [Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites) † | Phishing or legitimate website from website indicators | 11,055 examples, 30 features |
| [Chess (King-Rook vs. King-Pawn)](https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn) † | Whether White can win from a chess endgame position | 3,196 examples; categorical board descriptors |
| [Occupancy Detection](https://archive.ics.uci.edu/dataset/357/occupancy+detection) † | Occupied or unoccupied room from sensor readings | 20,560 examples across supplied files, 5 sensor predictors excluding date |

---

## Problem 1 — Decision Trees and Pruning (40 points)

Choose a real dataset with at least 1,000 examples. Define the prediction task and justify your training/validation/test split. Use the same dataset and partitions for both problems, fit preprocessing on training data only, and keep the test set for final evaluation.

Implement **ID3** using NumPy or standard Python, without a library tree model. Discretize numerical features using boundaries learned from training data. Implement entropy and information gain, recompute gain at each node, and define stopping conditions, tie-breaking and a majority-class fallback for unseen categories.

Using scikit-learn and the original numerical features, compare an unconstrained **DecisionTreeClassifier**, a pre-pruned tree and a cost-complexity-pruned tree. Select `max_depth`, `min_samples_leaf` and `ccp_alpha` using validation data.

Design an investigation that explains how your tree makes decisions and how controlling its complexity affects generalization. Decide what evidence would establish that your implementation behaves correctly and support your choice of tree. Choose and justify the evaluation measures and ways of presenting results. Interpret your findings and identify where your conclusions remain uncertain.

---

## Problem 2 — Random Forest and Gradient Boosting (60 points)

Train **bagging and random forest classifiers** using scikit-learn's **BaggingClassifier** and **RandomForestClassifier**. Use decision trees as the bagging base learners and enable bootstrap sampling. Compare bagging with all features available at each split against a forest with random candidate features at each node.

Use the models' built-in **out-of-bag (OOB) evaluation** and investigate the effects of tree count, feature-subset size and random seed. Use only valid OOB predictions when evaluating results.

Train **one gradient boosting classifier** using scikit-learn's **GradientBoostingClassifier**, XGBoost, LightGBM or CatBoost. Investigate learning rate and tree complexity, and use validation-based early stopping. Compare the forest's impurity-based feature importance with repeated validation permutation importance, keeping the fitted model fixed during permutation. Library models may be used throughout this problem.

Design a fair comparison of your selected **decision tree**, **bagging**, **random forest** and **gradient boosting** models using the same evaluation partitions and a comparable tuning budget. Decide what evidence you need to explain their differences, assess the reliability of OOB estimates and feature importance, and recommend a model for your prediction task. Choose which results deserve attention and justify how they support your conclusions.

Select the final model using validation results and freeze all settings before test evaluation. Use the test results to assess your recommendation and explain the limitations of your findings.

---

## Submission

Submit one Jupyter notebook for each problem that runs from beginning to end. Organize each notebook around the questions you investigated and the conclusions supported by your experiments.

Before running experiments, propose the questions or hypotheses you want to investigate and explain what evidence would help answer them. Report appropriate metrics, and justify those choices. Explain how unexpected findings changed your investigation. 

Provide enough information for another student to reproduce your work, and distinguish your implementations from library models.
