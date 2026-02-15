"""Evaluation metrics."""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class PredictionMetrics:
    """Compute evaluation metrics."""

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return r2_score(y_true, y_pred)

    @staticmethod
    def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_true - y_pred

    @staticmethod
    def residual_statistics(residuals: np.ndarray) -> Dict[str, float]:
        return {
            "mean": np.mean(residuals),
            "std": np.std(residuals),
            "min": np.min(residuals),
            "max": np.max(residuals),
            "median": np.median(residuals),
        }

    @classmethod
    def compute_all_metrics(
        cls, y_true: np.ndarray, y_pred: np.ndarray, include_residuals: bool = True
    ) -> Dict[str, float]:
        metrics = {
            "mse": cls.mse(y_true, y_pred),
            "mae": cls.mae(y_true, y_pred),
            "rmse": cls.rmse(y_true, y_pred),
            "r2": cls.r2(y_true, y_pred),
        }

        if include_residuals:
            residuals = cls.compute_residuals(y_true, y_pred)
            residual_stats = cls.residual_statistics(residuals)
            metrics.update({f"residual_{k}": v for k, v in residual_stats.items()})

        return metrics

    @staticmethod
    def compare_models(y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        results = {}
        for model_name, y_pred in predictions_dict.items():
            results[model_name] = PredictionMetrics.compute_all_metrics(y_true, y_pred, include_residuals=False)
        return results
