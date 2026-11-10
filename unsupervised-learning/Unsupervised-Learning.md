# Coding Assignment: K-means and Hierarchical Clustering

Train and compare K-means and hierarchical clustering models. Explain how feature scaling, initialization, linkage and the number of clusters affect the groups you discover.

## Sample datasets

Choose **one real-world dataset** with **at least 1,000 examples** and **at least four numerical features**. Complete both problems using the same observations and selected features.

| Dataset                                                                                                                      | Size                                            | Numerical features                                       |
| ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------- |
| [Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication)                                    | 1,372 examples                                  | 4 numerical features |
| [Rice](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik)                                                      | 3,810 examples                                  | 7 numerical features |
| [Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)                                                          | 13,611 examples                                 | 16 numerical features |
| [Wine Quality — White Wine](https://archive.ics.uci.edu/dataset/186/wine+quality)                                            | 4,898 examples                                  | 11 numerical features |
| [HTRU2](https://archive.ics.uci.edu/dataset/372/htru2)                                                                        | 17,898 examples                                 | 8 numerical features |
| [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) | 5,620 examples across supplied train/test files | 64 numerical image features |
| [Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)                        | 1,030 examples                                  | 8 numerical features |
| [Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)                                              | 1,503 examples                                  | 5 numerical features |
| [Abalone](https://archive.ics.uci.edu/dataset/1/abalone)                                                                      | 4,177 examples                                  | 7 numerical measurements |
| [Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)                              | 9,568 examples; use one worksheet only          | 4 numerical features |

Describe the dataset and justify your feature selection, preprocessing and experimental setup. Choose suitable options for your data and explain your decisions.

---

## Problem 1 — K-means Clustering (40 points)

Train **KMeans** using scikit-learn. Investigate how feature scaling, the number of clusters and initialization affect the results.

- Experiment with different values of **K**. Use inertia, silhouette scores and cluster visualizations to justify your choice.
- Compare **random** initialization with **k-means++** and investigate stability across repeated runs.
- Select two numerical features for an illustration using real observations from your dataset. Perform **one assignment step and one centroid update using NumPy**. Show the assignments, updated centroids and change in inertia. Explain why these steps do not increase the objective and why convergence does not guarantee a global optimum.

You may use library models for training; only the illustrated assignment and centroid update must use NumPy directly.

---

## Problem 2 — Hierarchical Clustering and Comparison (60 points)

Train **AgglomerativeClustering** using scikit-learn and compare its groupings with K-means from Problem 1.

- Compare **single, complete, average and Ward linkage**. Choose compatible distance measures and justify your choices.
- Plot dendrograms for at least two linkage rules. Interpret their merge heights and justify where you cut the trees.
- Investigate how linkage and the number of clusters affect the results. Discuss sensitivity to outliers, cluster shape and computational cost.

Design a fair comparison of the methods using suitable metrics and visualizations. You may use a reproducible subset if computational cost is a concern; justify its size and selection.

Include a comparison table summarizing the configurations, clustering quality, cluster sizes and runtime. Interpret your preferred clusters using the original features. Recommend a method based on your results, discuss its limitations and assess whether the groups are useful for your chosen dataset.

---

## Optional extension — Imputation or Outlier Detection (ungraded)

Choose one application from the lecture:

- **Imputation:** Use cluster information to estimate missing entries. Design an evaluation using hidden known values and compare your method with a simple baseline.
- **Outlier detection:** Use distances to cluster centers to identify unusual observations. Justify your scoring and threshold choices, and interpret several flagged examples.

Choose and justify a suitable training and evaluation procedure for your application.

---

## Submission

Submit **one runnable Jupyter notebook per problem**. Include your code, key plots, comparison tables and concise explanations. State your experimental questions, justify your methods and evaluation choices, and provide dataset sources and enough details to reproduce your results. Include the optional extension in the second notebook if completed.
