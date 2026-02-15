"""Data preprocessing utilities for traffic prediction."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler


class TrafficPreprocessor:
    """Preprocess traffic data for time series forecasting."""

    def __init__(self, window_length: int = 51, polyorder: int = 3, normalize: bool = True):
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")

        self.window_length = window_length
        self.polyorder = polyorder
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None

    def apply_savgol_filter(
        self, data: np.ndarray, window_length: Optional[int] = None, polyorder: Optional[int] = None
    ) -> np.ndarray:
        """Apply Savitzky-Golay smoothing filter."""
        wl = window_length if window_length is not None else self.window_length
        po = polyorder if polyorder is not None else self.polyorder

        if len(data) < wl:
            wl = len(data) if len(data) % 2 == 1 else len(data) - 1
            if wl < po + 2:
                return data

        if data.ndim == 1:
            return savgol_filter(data, wl, po)
        else:
            return np.apply_along_axis(lambda x: savgol_filter(x, wl, po), axis=0, arr=data)

    def normalize_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize data using StandardScaler."""
        if not self.normalize:
            return data

        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Handle NaN and inf values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            normalized = self.scaler.fit_transform(data)
            # Check for zero variance (which causes NaN)
            if hasattr(self.scaler, "scale_") and np.any(self.scaler.scale_ == 0):
                print("Warning: Some features have zero variance. Adding small epsilon.")
                self.scaler.scale_[self.scaler.scale_ == 0] = 1e-8
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            normalized = self.scaler.transform(data)

        # Ensure no NaN in output
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        return normalized.reshape(original_shape)

    def inverse_normalize(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data."""
        if not self.normalize or self.scaler is None:
            return data

        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        denormalized = self.scaler.inverse_transform(data)
        return denormalized.reshape(original_shape)

    def split_data(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        time_column: str = "time",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically into train/validation/test sets."""
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")

        if time_column in data.columns:
            data = data.sort_values(time_column).reset_index(drop=True)

        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = data.iloc[:train_end].copy()
        val_df = data.iloc[train_end:val_end].copy()
        test_df = data.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def preprocess_pipeline(
        self, data: pd.DataFrame, target_columns: list, smooth: bool = True, normalize: bool = True, split: bool = True
    ) -> dict:
        """Complete preprocessing pipeline."""
        result = {"original_data": data.copy()}

        target_data = data[target_columns].values

        if smooth:
            target_data = self.apply_savgol_filter(target_data)
            result["smoothed_data"] = target_data.copy()

        if normalize and self.normalize:
            target_data = self.normalize_data(target_data, fit=True)
            result["normalized_data"] = target_data.copy()

        processed_df = data.copy()
        for i, col in enumerate(target_columns):
            processed_df[col] = target_data[:, i] if target_data.ndim > 1 else target_data

        result["processed_data"] = processed_df

        if split:
            train_df, val_df, test_df = self.split_data(processed_df)
            result["train_data"] = train_df
            result["val_data"] = val_df
            result["test_data"] = test_df
            result["split_info"] = {"train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df)}

        return result
