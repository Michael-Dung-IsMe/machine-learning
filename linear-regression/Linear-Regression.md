# Coding Assignments: Linear Regression

Students must choose an appropriate dataset, structure their code, design experiments, validate numerical behavior, and support their conclusions.

## Sample datasets

These datasets intentionally allow more than one defensible formulation. The instructor may assign a particular file, target, scenario, group, or prediction horizon to each student or team. Consequently, numerical results are not expected to be identical across submissions.

| Problem | Dataset | Prediction target and scale | Why it is difficult |
|---|---|---|---|
| 1 | [Year Prediction MSD](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd) | Release year from 90 audio features; 515,345 songs | Large enough to expose inefficient loops and optimization choices; features arise from different timbre summaries; the repository specifies a non-random split intended to avoid artist leakage. |
| 1 | [Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity) | Number of article shares from 58 features; 39,797 articles | The target is strongly skewed, extreme articles are influential, and the publication sequence complicates evaluation. |
| 1 | [SGEMM GPU Kernel Performance](https://archive.ics.uci.edu/dataset/440/sgemm+gpu+kernel+performance) | GPU execution time from 14 kernel parameters; 241,600 configurations with four measured runs each | The runtime distribution is highly uneven, parameters are discrete rather than naturally continuous, and students must decide how the four repeated measurements define the target. |
| 1 | [Large-scale Wave Energy Farm](https://archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm) | Total power from converter coordinates; 63,600 configurations and as many as 149 features | Four files represent different locations and farm sizes; coordinate interactions are important, feature counts are high, and results depend strongly on which scenario is assigned. |
| 2 | [Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data) | Hourly PM2.5 concentration from weather and time variables; 43,824 observations | Missing values, temporal dependence, seasonal cycles, and nonlinear weather effects make a simple straight-line model inadequate. |
| 2 | [Forest Fires](https://archive.ics.uci.edu/dataset/162/forest+fires) | Burned forest area from spatial, calendar, fire-index, and weather variables; 517 observations | The target has many zeros and a long right tail; the dataset is small enough that an unnecessarily rich basis can overfit severely. |
| 2 | [Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | Rental count from calendar and weather variables; hourly and daily files with 17,389 total records | Hour, weekday, and season invite cyclic bases; random splitting can hide temporal drift; and two component-count columns sum exactly to the target, creating an easy-to-miss leakage trap. |
| 2 | [3D Road Network](https://archive.ics.uci.edu/dataset/246/3d+road+network+north+jutland+denmark) | Elevation from geographic coordinates; 434,874 ordered road points | The surface is spatially nonlinear, observations are grouped by road segment, and random, road-group, and geographic holdout strategies produce very different conclusions. |
| 2 | [Residential Building](https://archive.ics.uci.edu/dataset/437/residential+building+data+set) | Construction cost or sale price from roughly 103 project and lagged economic variables; 372 projects | The number of predictors is large relative to the sample size, and lagged economic measurements create substantial redundancy. |
| 2 | [Communities and Crime](https://archive.ics.uci.edu/dataset/183/communities+and+crime) | Per-capita violent-crime rate from 122 candidate predictors; 1,994 communities | Many predictors are related, some values are missing, and sensitive demographic variables demand careful interpretation and limitation statements. |
| 2 | [Superconductivity](https://archive.ics.uci.edu/dataset/464/superconductivty+data) | Critical temperature from 81 engineered material features; 21,263 superconductors | Many variables are alternative summaries of the same elemental properties, producing related coefficient groups; a second linked file contains chemical compositions and creates additional formulation choices. |
| 2 | [Facebook Comment Volume](https://archive.ics.uci.edu/dataset/363/facebook+comment+volume+) | Future comment count from post and page statistics; 40,949 observations | Five supplied data variants represent different prediction settings, the count target is strongly skewed, and several historical aggregates are redundant or potentially unstable. |

---

## Problem 1 — Linear Regression from First Principles (40 points)

Formulate the prediction task in your own words and identify the numerical target. Follow any split defined by the dataset source and explain its purpose.

Develop a linear regression model beginning with your own implementation rather than a ready-made regression estimator. Connect the implementation to residuals, squared error, gradients, and parameter updates. Investigate how data representation, feature scales, and optimization choices influence the learning process.

Compare the result with an independently obtained least-squares solution. Decide what evidence is needed to show that the comparison is fair and that the implementation is credible.

---

## Problem 2 — Basis Expansion, Regularization, and Diagnostics (60 points)


Construct and test a basis expansion that allows a linear regression model to represent nonlinear structure. The expansion may involve polynomial, piecewise, spline-like, periodic, interaction, or other defensible basis functions. Compare the expanded representation with the original feature representation and investigate more than one level of complexity.

Fit ordinary least squares, Ridge regression, and LASSO to the resulting representation. Examine whether regularization controls the additional complexity introduced by the basis expansion and how it changes prediction, coefficient stability, and sparsity.

Develop a process for selecting and comparing the models without allowing the final evaluation data to influence those decisions. Use regression metrics and residual behavior to judge the models from different perspectives. Explain why the chosen basis is plausible, what evidence supports the final model, and what important limitations remain.

---
