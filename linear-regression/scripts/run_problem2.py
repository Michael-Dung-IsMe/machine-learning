import os
import sys
import numpy as np
import pandas as pd

# Configure UTF-8 encoding for Windows console compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    load_and_preprocess_bike_sharing,
    train_val_test_split,
    StandardScaler
)
from src.models.linear_regression import LinearRegression
from src.models.regularization import (
    RidgeRegression,
    LassoRegression,
    compute_coefficient_paths
)
from src.basic_expansion import BikeSharingBasisExpansion
from src.metrics import rmse, mae, r2_score
from src.utils import (
    plot_coefficient_paths,
    plot_residuals,
    plot_model_comparison,
    plot_multi_metric_comparison
)


def run_problem2(data_path: str = None, output_dir: str = None, p1_data: dict = None) -> dict:
    """
    Executes Problem 2 workflow.
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(script_dir, "data", "raw", "bike_sharing.zip")
    if output_dir is None:
        output_dir = os.path.join(script_dir, "outputs")

    fig_dir = os.path.join(output_dir, "figures")

    print("\n" + "=" * 60)
    print("RUNNING PROBLEM 2: BASIS EXPANSION & REGULARIZATION (RIDGE & LASSO)")
    print("=" * 60)

    # 1. Load Data if not passed from Problem 1
    if p1_data is None:
        X, y, df_clean = load_and_preprocess_bike_sharing(
            file_path=data_path, target_col="cnt", drop_leakage_cols=True, as_numpy=True
        )
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, val_size=0.15, test_size=0.15, shuffle=False
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit baseline OLS for comparison
        model_ols_raw = LinearRegression()
        model_ols_raw.fit_normEqn(X_train_scaled, y_train)
        model_gd_raw = LinearRegression(lr=0.05, iterations=2000)
        model_gd_raw.fit_gd(X_train_scaled, y_train)
    else:
        X_train_scaled = p1_data["X_train"]
        X_val_scaled = p1_data["X_val"]
        X_test_scaled = p1_data["X_test"]
        y_train = p1_data["y_train"]
        y_val = p1_data["y_val"]
        y_test = p1_data["y_test"]
        model_ols_raw = p1_data["model_norm"]
        model_gd_raw = p1_data["model_gd"]

    # 2. Basis Expansion
    print("\n[1] Applying Basis Expansion (Periodic sin/cos + Polynomial + Interactions)...")
    basis = BikeSharingBasisExpansion(n_harmonics=2, poly_degree=2)
    X_train_exp = basis.fit_transform(X_train_scaled)
    X_val_exp = basis.transform(X_val_scaled)
    X_test_exp = basis.transform(X_test_scaled)

    # Re-scale expanded features
    exp_scaler = StandardScaler()
    X_train_exp = exp_scaler.fit_transform(X_train_exp)
    X_val_exp = exp_scaler.transform(X_val_exp)
    X_test_exp = exp_scaler.transform(X_test_exp)

    print(f"   Features expanded from {X_train_scaled.shape[1]} to {X_train_exp.shape[1]} features.")

    # 3. OLS on Expanded Features
    print("\n[2] Training OLS on Expanded Features...")
    model_exp_ols = LinearRegression()
    model_exp_ols.fit_normEqn(X_train_exp, y_train)
    y_val_exp_ols = model_exp_ols.predict(X_val_exp)
    print(f"   OLS (Expanded) Validation RMSE: {rmse(y_val, y_val_exp_ols):.4f}")

    # 4. Hyperparameter Tuning for Ridge & LASSO on Validation Set
    print("\n[3] Tuning Regularization Alphas on Validation Set...")
    ridge_alphas = np.logspace(-3, 3, 20)
    best_ridge_alpha, best_ridge_rmse, best_ridge_model = None, float("inf"), None
    for alpha in ridge_alphas:
        m = RidgeRegression(alpha=alpha).fit(X_train_exp, y_train)
        score_rmse = rmse(y_val, m.predict(X_val_exp))
        if score_rmse < best_ridge_rmse:
            best_ridge_rmse = score_rmse
            best_ridge_alpha = alpha
            best_ridge_model = m

    print(f"   Best Ridge Alpha: {best_ridge_alpha:.4f} | Val RMSE: {best_ridge_rmse:.4f}")

    lasso_alphas = np.logspace(-3, 1, 15)
    best_lasso_alpha, best_lasso_rmse, best_lasso_model = None, float("inf"), None
    for alpha in lasso_alphas:
        m = LassoRegression(alpha=alpha, max_iter=200, tol=1e-3).fit(X_train_exp, y_train)
        score_rmse = rmse(y_val, m.predict(X_val_exp))
        if score_rmse < best_lasso_rmse:
            best_lasso_rmse = score_rmse
            best_lasso_alpha = alpha
            best_lasso_model = m

    print(f"   Best LASSO Alpha: {best_lasso_alpha:.4f} | Val RMSE: {best_lasso_rmse:.4f}")
    zero_weights = np.sum(best_lasso_model.w.ravel() == 0)
    print(f"   LASSO Sparsity: {zero_weights}/{X_train_exp.shape[1]} feature weights set to 0.0")

    # 5. Coefficient Paths & Residual Diagnostics
    print("\n[4] Generating Coefficient Paths & Residual Diagnostics...")

    lasso_coefs, _ = compute_coefficient_paths(
        LassoRegression, X_train_exp, y_train, lasso_alphas, max_iter=300, tol=1e-3
    )
    plot_coefficient_paths(
        lasso_alphas, lasso_coefs,
        best_alpha=best_lasso_alpha,
        title="Problem 2: LASSO Coefficient Paths (Sparsity Trace)",
        save_path=os.path.join(fig_dir, "p2_lasso_coef_paths.png"),
        show=False
    )

    ridge_coefs, _ = compute_coefficient_paths(RidgeRegression, X_train_exp, y_train, ridge_alphas)
    plot_coefficient_paths(
        ridge_alphas, ridge_coefs,
        best_alpha=best_ridge_alpha,
        title="Problem 2: Ridge Coefficient Paths (L2 Shrinkage)",
        save_path=os.path.join(fig_dir, "p2_ridge_coef_paths.png"),
        show=False
    )

    y_val_pred_best = best_ridge_model.predict(X_val_exp)
    plot_residuals(
        y_val, y_val_pred_best,
        title="Problem 2: Residual Analysis (Best Ridge Model)",
        save_path=os.path.join(fig_dir, "p2_residual_analysis.png"),
        show=False
    )

    # 6. Final Evaluation on Test Set
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION SUMMARY")
    print("=" * 60)

    y_test_ols_raw = model_ols_raw.predict(X_test_scaled)
    y_test_gd_raw = model_gd_raw.predict(X_test_scaled)
    y_test_exp_ols = model_exp_ols.predict(X_test_exp)
    y_test_ridge = best_ridge_model.predict(X_test_exp)
    y_test_lasso = best_lasso_model.predict(X_test_exp)

    summary_rows = [
        {"Model": "OLS (Raw Features)", "RMSE": rmse(y_test, y_test_ols_raw), "MAE": mae(y_test, y_test_ols_raw), "R2": r2_score(y_test, y_test_ols_raw)},
        {"Model": "GD (Raw Features)", "RMSE": rmse(y_test, y_test_gd_raw), "MAE": mae(y_test, y_test_gd_raw), "R2": r2_score(y_test, y_test_gd_raw)},
        {"Model": "OLS (Basis Expansion)", "RMSE": rmse(y_test, y_test_exp_ols), "MAE": mae(y_test, y_test_exp_ols), "R2": r2_score(y_test, y_test_exp_ols)},
        {"Model": "Ridge (Basis Expansion)", "RMSE": rmse(y_test, y_test_ridge), "MAE": mae(y_test, y_test_ridge), "R2": r2_score(y_test, y_test_ridge)},
        {"Model": "LASSO (Basis Expansion)", "RMSE": rmse(y_test, y_test_lasso), "MAE": mae(y_test, y_test_lasso), "R2": r2_score(y_test, y_test_lasso)},
    ]

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(output_dir, "metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved metrics summary to: {summary_path}")

    metrics_dict = {row["Model"]: row["RMSE"] for row in summary_rows}
    plot_model_comparison(
        metrics_dict,
        metric_name="Test RMSE (Lower is Better)",
        title="Model Performance Comparison on Test Set",
        save_path=os.path.join(fig_dir, "p2_model_comparison.png"),
        show=False
    )

    plot_multi_metric_comparison(
        summary_df,
        title="Comprehensive Model Performance Comparison (Test Set)",
        save_path=os.path.join(fig_dir, "p2_multi_metric_comparison.png"),
        show=False
    )

    print("✅ Problem 2 executed successfully!")
    return summary_df


if __name__ == "__main__":
    run_problem2()
