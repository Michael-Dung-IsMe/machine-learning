"""
Script for executing Problem 1: Linear Regression from First Principles.
Runs OLS via Normal Equation and Gradient Descent, evaluates metrics, and saves plots based on chosen cost_func.
"""

import os
import sys
import numpy as np

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
from src.metrics import mse, rmse, mae, r2_score
from src.utils import plot_loss_convergence, plot_predictions_vs_actual


def run_problem1(data_path: str = None, output_dir: str = None, cost_func: str = "mse") -> dict:
    """
    Executes Problem 1 workflow.
    
    Args:
        data_path: Path to dataset file
        output_dir: Output directory for plots and metrics
        cost_func: Cost function to track ('mse', 'rmse', 'mae', 'r2')
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(script_dir, "data", "raw", "bike_sharing.zip")
    if output_dir is None:
        output_dir = os.path.join(script_dir, "outputs")

    fig_dir = os.path.join(output_dir, "figures")

    print("\n" + "=" * 60)
    print(f"RUNNING PROBLEM 1: LINEAR REGRESSION FROM FIRST PRINCIPLES (Cost Func: {cost_func.upper()})")
    print("=" * 60)

    # 1. Load & Preprocess Data
    print("\n[1] Loading and preprocessing dataset...")
    X, y, df_clean = load_and_preprocess_bike_sharing(
        file_path=data_path,
        target_col="cnt",
        drop_leakage_cols=True,
        as_numpy=True
    )

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=0.15, test_size=0.15, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 2. OLS via Normal Equation
    print("\n[2] Training OLS via Normal Equation...")
    model_norm = LinearRegression()
    model_norm.fit_normEqn(X_train_scaled, y_train, cost_func=cost_func)
    y_pred_val_norm = model_norm.predict(X_val_scaled)

    rmse_norm = rmse(y_val, y_pred_val_norm)
    mae_norm = mae(y_val, y_pred_val_norm)
    r2_norm = r2_score(y_val, y_pred_val_norm)
    print(f"   Validation Metrics -> RMSE: {rmse_norm:.4f} | MAE: {mae_norm:.4f} | R2: {r2_norm:.4f}")

    # 3. OLS via Gradient Descent
    print(f"\n[3] Training OLS via Gradient Descent (lr=0.05, iter=2000, cost_func={cost_func.upper()})...")
    model_gd = LinearRegression(lr=0.05, iterations=2000)
    model_gd.fit_gd(X_train_scaled, y_train, cost_func=cost_func)
    y_pred_val_gd = model_gd.predict(X_val_scaled)

    rmse_gd = rmse(y_val, y_pred_val_gd)
    mae_gd = mae(y_val, y_pred_val_gd)
    r2_gd = r2_score(y_val, y_pred_val_gd)
    print(f"   Validation Metrics -> RMSE: {rmse_gd:.4f} | MAE: {mae_gd:.4f} | R2: {r2_gd:.4f}")

    # 4. Save Plots
    print("\n[4] Generating diagnostic plots for Problem 1...")
    plot_loss_convergence(
        model_gd.cost_history,
        cost_name=cost_func,
        title=f"Problem 1: GD Loss Convergence ({cost_func.upper()})",
        save_path=os.path.join(fig_dir, "p1_loss_convergence.png"),
        show=False
    )

    plot_predictions_vs_actual(
        y_val, y_pred_val_norm,
        title="Problem 1: OLS Normal Eq (Actual vs Predicted)",
        save_path=os.path.join(fig_dir, "p1_actual_vs_predicted.png"),
        show=False
    )

    print("✅ Problem 1 executed successfully!")

    return {
        "X_train": X_train_scaled, "y_train": y_train,
        "X_val": X_val_scaled,     "y_val": y_val,
        "X_test": X_test_scaled,   "y_test": y_test,
        "scaler": scaler,
        "feature_names": [c for c in df_clean.columns if c != "cnt"],
        "model_norm": model_norm,
        "model_gd": model_gd
    }


if __name__ == "__main__":
    run_problem1()
