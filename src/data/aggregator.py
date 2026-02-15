"""Traffic aggregation utilities for time series prediction."""

from typing import Optional

import numpy as np
import pandas as pd


class TrafficAggregator:
    """Aggregate traffic measurements for time series forecasting."""

    def __init__(self, aggregation_window: str = "500ms"):
        self.aggregation_window = aggregation_window

    def aggregate_traffic(
        self,
        data: pd.DataFrame,
        time_column: str = "time",
        value_columns: Optional[list] = None,
        aggregation_func: str = "sum",
    ) -> pd.DataFrame:
        """Aggregate traffic measurements over time windows."""
        if time_column not in data.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")

        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            data = data.copy()
            data[time_column] = pd.to_datetime(data[time_column])

        df = data.set_index(time_column)

        if value_columns is None:
            value_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if aggregation_func == "sum":
            aggregated = df[value_columns].resample(self.aggregation_window).sum()
        elif aggregation_func == "mean":
            aggregated = df[value_columns].resample(self.aggregation_window).mean()
        elif aggregation_func == "max":
            aggregated = df[value_columns].resample(self.aggregation_window).max()
        elif aggregation_func == "min":
            aggregated = df[value_columns].resample(self.aggregation_window).min()
        else:
            raise ValueError(f"Unknown aggregation function: {aggregation_func}")

        aggregated = aggregated.reset_index()

        non_numeric_cols = [col for col in df.columns if col not in value_columns]
        if non_numeric_cols:
            for col in non_numeric_cols:
                grouped = df[col].resample(self.aggregation_window).first()
                aggregated[col] = grouped.values

        return aggregated

    def create_sequences(
        self,
        data: pd.DataFrame,
        input_length: int,
        output_length: int,
        target_column: str,
        feature_columns: Optional[list] = None,
        stride: int = 1,
    ) -> tuple:
        """Create input-output sequences for time series prediction."""
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        if feature_columns is None:
            feature_columns = [target_column]
        else:
            if target_column not in feature_columns:
                feature_columns = [target_column] + feature_columns

        feature_data = data[feature_columns].values
        target_data = data[target_column].values

        X, y = [], []

        for i in range(0, len(data) - input_length - output_length + 1, stride):
            x_seq = feature_data[i : i + input_length]
            y_seq = target_data[i + input_length : i + input_length + output_length]

            X.append(x_seq)
            y.append(y_seq)

        return np.array(X), np.array(y)
