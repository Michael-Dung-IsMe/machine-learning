# Linear Regression Assignment Report

**Assignment:** Linear Regression from First Principles, Basis Expansion, and Regularization  
**Dataset:** Bike Sharing Dataset (UCI Machine Learning Repository)  
**Prediction Target:** Hourly Rental Bike Count (`cnt`)  

---

## 1. Project Overview & Problem Formulation

### 1.1. Problem Statement
Predict the hourly count of rented bikes (`cnt`) based on environmental, weather, and seasonal variables (hour, day of week, month, season, working day indicator, temperature, feeling temperature, humidity, and windspeed).

### 1.2. Key Data Challenges
1. **Non-linear & Cyclical Demand:** Bike usage exhibits strong non-linear hourly patterns (sharp peaks at 8 AM and 5 PM commuting hours) as well as seasonal cycles.
2. **Conflicting Behavioral Patterns:** Commuting days (`workingday = 1`) exhibit a bi-modal peak pattern (8 AM & 5 PM), whereas non-working days (`workingday = 0`) feature a smooth uni-modal afternoon peak (1 PM - 4 PM).
3. **Data Leakage Prevention:** The sum of `casual` (unregistered users) and `registered` (registered users) equals the total target (`casual + registered = cnt`). To prevent target leakage, `casual` and `registered` columns must be strictly dropped during preprocessing.
4. **Target Distribution Skewness:** The target `cnt` is non-negative count data with strong right-skewness and heteroscedasticity (variance scales with magnitude).

---

## 2. Project Architecture

The codebase is structured following clean Machine Learning engineering principles, separating pure NumPy core algorithms (`src/`), execution scripts (`scripts/`), Jupyter evaluation notebooks (`notebooks/`), and generated outputs (`outputs/`):

```text
linear-regression/
├── main.py
├── README.md
├── draft_vi.md
├── requirements.txt
├── data/
├── src/
│   ├── models/
│   ├── preprocessing.py
│   ├── basic_expansion.py
│   ├── advanced_models.py
│   └── metrics.py
├── scripts/
│   ├── run_problem1.py
│   ├── run_problem2.py
│   └── run_improvements.py
├── notebooks/
└── outputs/
```

---

## 3. Baseline Experiments

### 3.1. Problem 1 — Linear Regression from First Principles
- **Methodology:** Implemented two OLS solvers from scratch using pure NumPy:
  1. **Normal Equation:** $\boldsymbol{w} = (\boldsymbol{X}^T \boldsymbol{X})^{-1} \boldsymbol{X}^T \boldsymbol{y}$
  2. **Gradient Descent:** Iterative weight updates optimizing MSE loss.
- **Dataset Split:** Chronological split (Train 70%, Validation 15%, Test 15%). Standardized continuous features using `StandardScaler`.

### 3.2. Problem 2 — Basis Expansion & Regularization
- **Basis Expansion:** Extended raw features using Fourier periodic terms ($k=2$ harmonics for cyclic variables: `hr`, `mnth`, `weekday`, `season`), degree-2 polynomials for weather variables (`temp`, `hum`, `windspeed`), and pairwise interaction terms (expanded feature space from 11 to 32 dimensions).
- **Regularization:** Tuned hyperparameter $\alpha$ for **Ridge Regression ($L_2$)** and **LASSO Regression ($L_1$)** via Coordinate Descent on the Validation set.

### 3.3. Baseline Test Set Evaluation Summary

| Model | RMSE (↓) | MAE (↓) | R2 Score (↑) | Remarks |
| :--- | :---: | :---: | :---: | :--- |
| **OLS (Raw Features)** | 178.9896 | 137.3295 | **0.3051** | Initial raw features baseline |
| **GD (Raw Features)** | 179.0613 | 137.4379 | **0.3045** | Gradient Descent converges to Normal Eq solution |
| **OLS (Basis Expansion)** | 155.5845 | 114.9906 | **0.4750** | Fourier $k=2$ + Polynomial weather features |
| **Ridge (Basis Expansion)** | 159.6373 | 119.4131 | **0.4472** | L2 weight shrinkage regularization |
| **LASSO (Basis Expansion)** | 159.7083 | 119.5229 | **0.4468** | L1 feature selection & sparsity trace |

---

## 4. Advanced Technical Methods Overview

To break the baseline performance ceiling ($R^2 \approx 0.4750$), three specialized technical enhancements were investigated and implemented:

### 4.1. Categorical One-Hot Encoding
- **Technical Mechanism:** Converted categorical features (`hr` 0-23, `mnth` 1-12, `weekday` 0-6, `weathersit` 1-4, `season` 1-4) into binary dummy indicator columns ($0/1$). Used `drop_first=True` to avoid perfect multicollinearity (Dummy Variable Trap).
- **Rationale:** Hour `hr` is inherently a discrete categorical state rather than a continuous linear scalar. Forcing `hr` into a single numeric column forces OLS to fit a monotonic straight line across hours. One-hot encoding enables OLS to learn independent, unconstrained weight offsets for every specific hour (e.g., sharp peaks at 8 AM and 5 PM).

### 4.2. Segmented Linear Regression (Piecewise Linear Modeling)
- **Technical Mechanism:** Partitioned the dataset into two logical subsets based on `workingday`:
  $$\text{Data} = \text{Data}_{\text{workingday}=1} \cup \text{Data}_{\text{workingday}=0}$$
  Trained two independent sub-models for commuting days vs weekends/holidays:
  $$\hat{y} = \begin{cases} f_{\text{work}}(\boldsymbol{X}) & \text{if } \text{workingday} = 1 \\ f_{\text{rest}}(\boldsymbol{X}) & \text{if } \text{workingday} = 0 \end{cases}$$
- **Rationale:** Commuting day demand (sharp morning/evening rush hour spikes) and non-workingday demand (smooth afternoon leisure peak) have conflicting functional forms. Segmenting the data completely eliminates pattern interference between working days and weekends.

### 4.3. Target Log Transformation
- **Technical Mechanism:** Applied a log-transformation $y \to z = \log(1 + y)$ prior to model fitting. At prediction time, inverse-transformed predictions back to the original scale:
  $$\hat{y} = \max\left(0, \exp(\hat{z}) - 1\right)$$
- **Rationale:** Hourly count data `cnt` exhibits strong right-skewness and heteroscedasticity. Log-transforming the target stabilizes variance, converts additive error assumptions into multiplicative relative errors, and prevents negative count predictions ($\hat{y} < 0$).

---

## 5. Experimental Results & In-Depth Technical Takeaways

### 5.1. Comprehensive Test Set Results Summary

| Index | Model Variation | RMSE (↓) | MAE (↓) | R2 Score (↑) | $\Delta R^2$ vs Baseline | % RMSE Reduction |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **Baseline OLS (Raw Features)** | 178.9896 | 137.3295 | **0.3051** | Ref | 0.0% |
| **0b**| **Baseline OLS (Basis Expansion)** | 155.5845 | 114.9906 | **0.4750** | $+0.1699$ | $-13.1\%$ |
| **1** | **Model A: One-Hot Encoding** | 133.1308 | 97.5136 | **0.6156** | $+0.3105$ | $-25.6\%$ |
| **2** | **Model B: Segmented OLS (workingday split)** | 177.1191 | 135.7294 | **0.3196** | $+0.0145$ | $-1.0\%$ |
| **3** | **Model C: Combined (Segmented + One-Hot)** | 106.3100 | 78.6623 | **0.7549** | $+0.4498$ | $-40.6\%$ |
| **4** | **Model D: Combined + Target Log-Transform ($\log(1+y)$)** | **78.9733** | **47.5416** | **0.8647** | **$+0.5596$** | **$-55.9\%$** |

---

### 5.2. In-Depth Technical Takeaways
1. **Decisive Impact of One-Hot Encoding (Model A)**:
   - Simply one-hot encoding discrete categories increased $R^2$ dramatically from **0.3051 to 0.6156** (+0.3105 boost), outperforming the trigonometric Fourier expansion ($R^2 = 0.4750$).
   - *Insight:* Hourly demand spikes are highly localized binary states rather than smooth sinusoidal waves. One-hot encoding grants OLS 23 degrees of freedom to fit exact hourly offsets.

2. **Synergistic Power of Segmentation (Model C)**:
   - While segmenting raw features alone (Model B) yielded only a modest $+0.0145$ $R^2$ improvement, combining segmentation with One-Hot Encoding (Model C) skyrocketed $R^2$ to **0.7549** (reducing MAE from 137.33 down to 78.66).
   - *Insight:* This combination allows the model to learn a specialized hourly weight profile for working days (rush hour spikes) and a separate hourly weight profile for weekends (afternoon leisure peak).

3. **Mastery of Residuals via Log Target Transformation (Model D)**:
   - Incorporating $\log(1+y)$ target transformation in Model D pushed $R^2$ to an outstanding **0.8647**, with MAE dropping to **47.54 bikes/hour** (a $65.4\%$ reduction in absolute error compared to baseline).
   - *Insight:* The log transform handles target scale variance, forces residual distribution into a symmetric Gaussian bell curve, and guarantees strictly non-negative predictions.

4. **Conclusion**:
   - Linear Regression, when paired with domain-appropriate **Feature Representation**, **Segmentation**, and **Target Transformations**, can elevate performance from poor ($R^2 = 0.3051$) to outstanding ($R^2 = 0.8647$) without resorting to black-box ensemble methods (such as Random Forest or XGBoost).

---

## 6. Installation & Execution Guide

### 6.1. Environment Setup
- Python 3.9+
- Dependencies listed in `requirements.txt`: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

### 6.2. Installation
```bash
pip install -r requirements.txt
```

### 6.3. Running Experiments via CLI
```bash
# 1. Run Problem 1 (First Principles OLS & GD)
python scripts/run_problem1.py

# 2. Run Problem 2 (Basis Expansion, Ridge & LASSO)
python scripts/run_problem2.py

# 3. Run Advanced Improvements (Models A, B, C, D)
python scripts/run_improvements.py

# Or run full pipeline via Orchestrator CLI:
python main.py --task all
```
Generated diagnostic plots are saved to `outputs/figures/` and metrics summaries are exported to `outputs/metrics_summary.csv`.
