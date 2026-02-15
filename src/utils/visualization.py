"""Visualization utilities."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


class TrafficVisualizer:
    """Visualization tools."""

    @staticmethod
    def plot_time_series(
        data: pd.DataFrame,
        time_column: str,
        value_columns: List[str],
        title: str = "Traffic Time Series",
        figsize: tuple = (14, 6),
        save_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=figsize)

        for col in value_columns:
            if col in data.columns:
                ax.plot(data[time_column], data[col], label=col, alpha=0.7)

        ax.set_xlabel("Time")
        ax.set_ylabel("Traffic (bps)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        time_index: Optional[np.ndarray] = None,
        title: str = "Predictions vs Ground Truth",
        figsize: tuple = (14, 6),
        save_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=figsize)

        if time_index is None:
            time_index = np.arange(len(y_true))

        ax.plot(time_index, y_true, label="Ground Truth", color="blue", alpha=0.7, linewidth=2)
        ax.plot(time_index, y_pred, label="Predictions", color="red", alpha=0.7, linewidth=2, linestyle="--")

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Traffic Value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_residuals(
        residuals: np.ndarray,
        time_index: Optional[np.ndarray] = None,
        title: str = "Residual Plot",
        figsize: tuple = (14, 8),
        save_path: Optional[str] = None,
        max_samples: int = 10000,  # Limit samples for performance
    ):
        """Plot residual analysis with performance optimization.

        Args:
            residuals: Residual values
            time_index: Time indices (optional)
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            max_samples: Maximum samples to plot (for performance)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Downsample if too many points for performance
        if len(residuals) > max_samples:
            print(f"Downsampling residuals from {len(residuals)} to {max_samples} for plotting performance")
            indices = np.linspace(0, len(residuals) - 1, max_samples, dtype=int)
            residuals_plot = residuals[indices]
            time_index_plot = indices if time_index is None else time_index[indices]
        else:
            residuals_plot = residuals
            time_index_plot = np.arange(len(residuals)) if time_index is None else time_index

        # 1. Residuals vs Time (scatter plot)
        axes[0, 0].scatter(time_index_plot, residuals_plot, alpha=0.5, s=10)
        axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=2)
        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Time")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residual Distribution (histogram)
        axes[0, 1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        axes[0, 1].axvline(x=0, color="r", linestyle="--", linewidth=2)
        axes[0, 1].set_xlabel("Residuals")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Residual Distribution")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q Plot (check normality)
        from scipy import stats

        stats.probplot(residuals_plot, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Autocorrelation (optimized)
        # Use manual ACF calculation instead of pandas for better performance
        max_lags = min(40, len(residuals_plot) // 4)  # Limit lags

        # Compute autocorrelation manually
        residuals_centered = residuals_plot - np.mean(residuals_plot)
        c0 = np.dot(residuals_centered, residuals_centered) / len(residuals_centered)

        acf_values = []
        for lag in range(max_lags + 1):
            if lag == 0:
                acf_values.append(1.0)
            else:
                c_lag = np.dot(residuals_centered[:-lag], residuals_centered[lag:]) / len(residuals_centered)
                acf_values.append(c_lag / c0)

        # Plot ACF
        lags = np.arange(max_lags + 1)
        axes[1, 1].stem(lags, acf_values, linefmt="C0-", markerfmt="C0o", basefmt="C0-")
        axes[1, 1].axhline(y=0, color="k", linestyle="-", linewidth=0.5)

        # Add confidence interval
        conf_int = 1.96 / np.sqrt(len(residuals_plot))
        axes[1, 1].axhline(y=conf_int, color="r", linestyle="--", linewidth=1, alpha=0.5)
        axes[1, 1].axhline(y=-conf_int, color="r", linestyle="--", linewidth=1, alpha=0.5)

        axes[1, 1].set_xlabel("Lag")
        axes[1, 1].set_ylabel("Autocorrelation")
        axes[1, 1].set_title("Residual Autocorrelation")
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_comparison(
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        time_index: Optional[np.ndarray] = None,
        title: str = "Model Comparison",
        figsize: tuple = (14, 6),
        save_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=figsize)

        if time_index is None:
            time_index = np.arange(len(y_true))

        ax.plot(time_index, y_true, label="Ground Truth", color="black", linewidth=2, alpha=0.8)

        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))

        for (model_name, y_pred), color in zip(predictions_dict.items(), colors):
            ax.plot(time_index, y_pred, label=model_name, alpha=0.7, linewidth=1.5, linestyle="--", color=color)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Traffic Value")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_metrics_comparison(
        metrics_dict: Dict[str, Dict[str, float]],
        metrics_to_plot: List[str] = ["mse", "mae", "rmse"],
        title: str = "Metrics Comparison",
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ):
        models = list(metrics_dict.keys())
        num_metrics = len(metrics_to_plot)

        fig, axes = plt.subplots(1, num_metrics, figsize=figsize)

        if num_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics_to_plot):
            values = [metrics_dict[model].get(metric, 0) for model in models]

            axes[idx].bar(models, values, alpha=0.7, edgecolor="black")
            axes[idx].set_xlabel("Model")
            axes[idx].set_ylabel(metric.upper())
            axes[idx].set_title(f"{metric.upper()} Comparison")
            axes[idx].grid(True, alpha=0.3, axis="y")
            axes[idx].tick_params(axis="x", rotation=45)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_training_history(
        history: Dict[str, List[float]],
        title: str = "Training History",
        figsize: tuple = (12, 5),
        save_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=figsize)

        epochs = range(1, len(history["train_loss"]) + 1)

        ax.plot(epochs, history["train_loss"], label="Training Loss", marker="o", linewidth=2)

        if "val_loss" in history:
            ax.plot(epochs, history["val_loss"], label="Validation Loss", marker="s", linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
