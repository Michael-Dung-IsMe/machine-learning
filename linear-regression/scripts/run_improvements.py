"""
Script for executing Advanced Improvement Experiments (Models A, B, C, D).
Evaluates One-Hot Encoding, Segmented Linear Regression, and Target Log Transformation on Test Set.
"""

import os
import sys

# Ensure root project directory is in sys.path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configure UTF-8 encoding for Windows console compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.preprocessing import (
    load_and_preprocess_bike_sharing,
    train_val_test_split,
    StandardScaler
)
from src.models.linear_regression import LinearRegression
from src.basic_expansion import BikeSharingBasisExpansion
from src.advanced_models import (
    prepare_advanced_features,
    SegmentedLinearRegression,
    log1p_transform,
    expm1_inverse_transform
)
from src.metrics import rmse, mae, r2_score


def run_improvements(data_path: str = None, output_dir: str = None) -> pd.DataFrame:
    """
    Executes advanced improvements evaluation workflow.
    """
    if data_path is None:
        data_path = os.path.join(script_dir, "data", "raw", "bike_sharing.zip")
    if output_dir is None:
        output_dir = os.path.join(script_dir, "outputs")

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("RUNNING ADVANCED IMPROVEMENT EXPERIMENTS (MODELS A, B, C, D)")
    print("=" * 60)

    # Load cleaned DataFrame
    _, _, df_clean = load_and_preprocess_bike_sharing(
        file_path=data_path, target_col="cnt", drop_leakage_cols=True, as_numpy=False
    )

    summary_rows = []

    # -------------------------------------------------------------
    # 0. Baseline 1: OLS Raw Features
    # -------------------------------------------------------------
    print("\n[0a] Running Baseline OLS (Raw Features)...")
    X_raw, y_raw, feature_names_raw = prepare_advanced_features(df_clean, use_one_hot=False)
    X_tr_raw, X_v_raw, X_te_raw, y_tr_raw, y_v_raw, y_te_raw = train_val_test_split(
        X_raw, y_raw, val_size=0.15, test_size=0.15, shuffle=False
    )
    scaler_raw = StandardScaler()
    X_tr_raw_sc = scaler_raw.fit_transform(X_tr_raw)
    X_te_raw_sc = scaler_raw.transform(X_te_raw)

    m_ols_raw = LinearRegression()
    m_ols_raw.fit_normEqn(X_tr_raw_sc, y_tr_raw)
    y_pred_ols_raw = m_ols_raw.predict(X_te_raw_sc)

    summary_rows.append({
        "Model": "Baseline OLS (Raw Features)",
        "RMSE": rmse(y_te_raw, y_pred_ols_raw),
        "MAE": mae(y_te_raw, y_pred_ols_raw),
        "R2": r2_score(y_te_raw, y_pred_ols_raw),
        "Description": "Standard raw features baseline"
    })

    # -------------------------------------------------------------
    # 0b. Baseline 2: OLS Basis Expansion
    # -------------------------------------------------------------
    print("[0b] Running Baseline OLS (Basis Expansion)...")
    basis = BikeSharingBasisExpansion(n_harmonics=2, poly_degree=2)
    X_tr_exp = basis.fit_transform(X_tr_raw_sc)
    X_te_exp = basis.transform(X_te_raw_sc)

    scaler_exp = StandardScaler()
    X_tr_exp_sc = scaler_exp.fit_transform(X_tr_exp)
    X_te_exp_sc = scaler_exp.transform(X_te_exp)

    m_ols_exp = LinearRegression()
    m_ols_exp.fit_normEqn(X_tr_exp_sc, y_tr_raw)
    y_pred_ols_exp = m_ols_exp.predict(X_te_exp_sc)

    summary_rows.append({
        "Model": "Baseline OLS (Basis Expansion)",
        "RMSE": rmse(y_te_raw, y_pred_ols_exp),
        "MAE": mae(y_te_raw, y_pred_ols_exp),
        "R2": r2_score(y_te_raw, y_pred_ols_exp),
        "Description": "Fourier k=2 + Polynomial weather"
    })

    # -------------------------------------------------------------
    # 1. Model A: One-Hot Encoding (hr, mnth, weekday, weathersit, season)
    # -------------------------------------------------------------
    print("\n[1] Training Model A: One-Hot Encoding...")
    X_oh, y_oh, feature_names_oh = prepare_advanced_features(df_clean, use_one_hot=True)
    X_tr_oh, X_v_oh, X_te_oh, y_tr_oh, y_v_oh, y_te_oh = train_val_test_split(
        X_oh, y_oh, val_size=0.15, test_size=0.15, shuffle=False
    )
    scaler_oh = StandardScaler()
    X_tr_oh_sc = scaler_oh.fit_transform(X_tr_oh)
    X_te_oh_sc = scaler_oh.transform(X_te_oh)

    m_model_a = LinearRegression()
    m_model_a.fit_normEqn(X_tr_oh_sc, y_tr_oh)
    y_pred_model_a = m_model_a.predict(X_te_oh_sc)

    summary_rows.append({
        "Model": "Model A: One-Hot Encoding",
        "RMSE": rmse(y_te_oh, y_pred_model_a),
        "MAE": mae(y_te_oh, y_pred_model_a),
        "R2": r2_score(y_te_oh, y_pred_model_a),
        "Description": "Dummy indicators for hr, mnth, etc."
    })

    # -------------------------------------------------------------
    # 2. Model B: Segmented OLS (workingday == 1 vs 0 on Raw Features)
    # -------------------------------------------------------------
    print("[2] Training Model B: Segmented OLS (workingday split)...")
    workingday_idx_raw = feature_names_raw.index("workingday")
    m_model_b = SegmentedLinearRegression(segment_col_idx=workingday_idx_raw)
    m_model_b.fit(X_tr_raw_sc, y_tr_raw)
    y_pred_model_b = m_model_b.predict(X_te_raw_sc)

    summary_rows.append({
        "Model": "Model B: Segmented OLS",
        "RMSE": rmse(y_te_raw, y_pred_model_b),
        "MAE": mae(y_te_raw, y_pred_model_b),
        "R2": r2_score(y_te_raw, y_pred_model_b),
        "Description": "2 separate models for workingday vs weekend"
    })

    # -------------------------------------------------------------
    # 3. Model C: Combined (Segmented OLS + One-Hot Encoding)
    # -------------------------------------------------------------
    print("[3] Training Model C: Combined (Segmented + One-Hot)...")
    workingday_idx_oh = feature_names_oh.index("workingday")
    m_model_c = SegmentedLinearRegression(segment_col_idx=workingday_idx_oh)
    m_model_c.fit(X_tr_oh_sc, y_tr_oh)
    y_pred_model_c = m_model_c.predict(X_te_oh_sc)

    summary_rows.append({
        "Model": "Model C: Combined (Segmented + One-Hot)",
        "RMSE": rmse(y_te_oh, y_pred_model_c),
        "MAE": mae(y_te_oh, y_pred_model_c),
        "R2": r2_score(y_te_oh, y_pred_model_c),
        "Description": "Segmented OLS with full One-Hot features"
    })

    # -------------------------------------------------------------
    # 4. Model D: Combined + Target Log-Transform (log1p(y))
    # -------------------------------------------------------------
    print("[4] Training Model D: Combined + Log Target Transform...")
    y_tr_log = log1p_transform(y_tr_oh)
    m_model_d = SegmentedLinearRegression(segment_col_idx=workingday_idx_oh)
    m_model_d.fit(X_tr_oh_sc, y_tr_log)
    y_pred_log_d = m_model_d.predict(X_te_oh_sc)
    y_pred_model_d = expm1_inverse_transform(y_pred_log_d)

    summary_rows.append({
        "Model": "Model D: Combined + Log Target",
        "RMSE": rmse(y_te_oh, y_pred_model_d),
        "MAE": mae(y_te_oh, y_pred_model_d),
        "R2": r2_score(y_te_oh, y_pred_model_d),
        "Description": "Segmented + One-Hot + log(1+y) Target"
    })

    # -------------------------------------------------------------
    # 5. Print & Save Results Summary
    # -------------------------------------------------------------
    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("FINAL ADVANCED IMPROVEMENTS EVALUATION SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    summary_csv_path = os.path.join(output_dir, "metrics_summary.csv")
    df_summary[["Model", "RMSE", "MAE", "R2"]].to_csv(summary_csv_path, index=False)
    print(f"\nSaved updated metrics summary to: {summary_csv_path}")

    # -------------------------------------------------------------
    # 6. Generate Comparison & Diagnostic Plots
    # -------------------------------------------------------------
    print("\n[5] Generating diagnostic plots for Improvements...")
    plt.figure(figsize=(10, 6))
    models = df_summary["Model"]
    r2_scores = df_summary["R2"]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(models)))

    bars = plt.barh(models, r2_scores, color=colors, edgecolor="black", height=0.6)
    plt.title("R2 Score Comparison Across Improvement Models (Higher is Better)", fontsize=13, fontweight="bold")
    plt.xlabel("R2 Score", fontsize=11)
    plt.ylabel("Model Variation", fontsize=11)
    plt.xlim(0, 1.0)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    plot_comp_path = os.path.join(fig_dir, "p3_improvements_comparison.png")
    plt.savefig(plot_comp_path, dpi=300)
    plt.close()
    print(f"   Saved comparison plot to: {plot_comp_path}")

    # Plot Residuals for Best Model (Model D)
    residuals_d = y_te_oh.ravel() - y_pred_model_d.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot: Actual vs Predicted
    axes[0].scatter(y_te_oh, y_pred_model_d, alpha=0.4, color="teal", s=15)
    max_val = max(y_te_oh.max(), y_pred_model_d.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2, label="Ideal Fit (y = y_hat)")
    axes[0].set_title("Model D: Actual vs Predicted", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Actual cnt", fontsize=11)
    axes[0].set_ylabel("Predicted cnt", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Residuals distribution
    axes[1].hist(residuals_d, bins=30, color="crimson", edgecolor="black", alpha=0.7)
    axes[1].set_title("Model D: Residuals Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Residual (y - y_hat)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_res_path = os.path.join(fig_dir, "p3_model_d_residuals.png")
    plt.savefig(plot_res_path, dpi=300)
    plt.close()
    print(f"   Saved residual plot for Model D to: {plot_res_path}")

    print("\n✅ Advanced Improvement Experiments Completed Successfully!")
    return df_summary


if __name__ == "__main__":
    run_improvements()
