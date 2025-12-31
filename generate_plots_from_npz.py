"""
Utility script for visualizing solar irradiance forecasting results.

The script reads the file `forecast_results.npz` produced during evaluation
and generates:
- error curves (MSE, MAE, RMSE) across forecast horizons
- actual vs predicted time-series plots for selected horizons
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_error_vs_horizon(errors, metric_name, target_names, save_dir="plots"):
    """
    Plots a given error metric as a function of forecast horizon
    for each predicted target variable.
    """
    os.makedirs(save_dir, exist_ok=True)
    horizons = np.arange(1, errors.shape[0] + 1)

    plt.figure(figsize=(8, 5))
    for i, name in enumerate(target_names):
        plt.plot(
            horizons,
            errors[:, i],
            marker="o",
            label=name.upper(),
        )

    plt.xlabel("Forecast Horizon")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs Forecast Horizon")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(
        save_dir, f"{metric_name.lower()}_vs_horizon.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved -> {save_path}")


def plot_predictions_vs_time(
    preds,
    targets,
    target_names,
    horizon=1,
    num_samples=None,
    save_dir="plots",
):
    """
    Plots predicted and ground-truth values over time for a specific
    forecast horizon.
    """
    os.makedirs(save_dir, exist_ok=True)
    t_idx = horizon - 1

    if num_samples is None:
        num_samples = preds.shape[0]
    num_samples = min(num_samples, preds.shape[0])

    for i, name in enumerate(target_names):
        plt.figure(figsize=(10, 5))
        plt.plot(
            targets[:num_samples, t_idx, i],
            label=f"Actual {name.upper()}",
            linewidth=2,
        )
        plt.plot(
            preds[:num_samples, t_idx, i],
            linestyle="--",
            label=f"Predicted {name.upper()}",
            linewidth=2,
        )

        plt.xlabel("Sample Index (Time)")
        plt.ylabel(f"{name.upper()} (W/m²)")
        plt.title(
            f"Actual vs Predicted {name.upper()} at Horizon {horizon}"
        )
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(
            save_dir,
            f"pred_vs_actual_{name.lower()}_h{horizon}.png",
        )
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved -> {save_path}")


def main(file_path="forecast_results.npz"):
    """
    Loads forecast results from disk and generates all evaluation plots.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_path} not found. Run the evaluation script first."
        )

    print(f"Loading results from {file_path}")
    data = np.load(file_path, allow_pickle=True)

    preds = data["preds"]
    targets = data["targets"]
    mse = data["mse"]
    mae = data["mae"]
    rmse = data["rmse"]
    target_names = data["target_names"]

    print("Data loaded successfully")
    print(f"Predictions shape: {preds.shape}")
    print(f"MSE shape: {mse.shape}")

    print("\nGenerating error vs horizon plots")
    plot_error_vs_horizon(mse, "MSE", target_names)
    plot_error_vs_horizon(mae, "MAE", target_names)
    plot_error_vs_horizon(rmse, "RMSE", target_names)

    horizons_to_plot = [1, 5, 10]
    print("\nGenerating predicted vs actual plots")
    for h in horizons_to_plot:
        plot_predictions_vs_time(
            preds,
            targets,
            target_names,
            horizon=h,
            num_samples=None,
        )

    print("\nAll plots saved in the 'plots' directory")


if __name__ == "__main__":
    main()
