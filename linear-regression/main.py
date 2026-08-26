"""
Main Entry Point for Linear Regression Machine Learning Pipeline.
Orchestrates execution of Problem 1, Problem 2, and Advanced Improvements scripts.
"""

import argparse
from scripts.run_problem1 import run_problem1
from scripts.run_problem2 import run_problem2
from scripts.run_improvements import run_improvements


def main():
    parser = argparse.ArgumentParser(description="Linear Regression ML Pipeline Orchestrator")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "improvements"],
        help="Task to run: '1' for Problem 1, '2' for Problem 2, '3' for Advanced Improvements, 'all' for full pipeline"
    )
    parser.add_argument(
        "--cost_func",
        type=str,
        default="mse",
        choices=["mse", "rmse", "mae", "r2"],
        help="Cost function to track during Gradient Descent ('mse', 'rmse', 'mae', 'r2')"
    )
    args = parser.parse_args()

    p1_data = None

    if args.task in ["1", "all"]:
        p1_data = run_problem1(cost_func=args.cost_func)

    if args.task in ["2", "all"]:
        run_problem2(p1_data=p1_data)

    if args.task in ["3", "improvements", "all"]:
        run_improvements()

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()