"""Dataset loader for Colosseum O-RAN COMMAG dataset."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


class ColosseumDatasetLoader:
    """Load and manage Colosseum O-RAN dataset files."""

    def __init__(self, dataset_root: str):
        """Initialize dataset loader."""
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise ValueError(f"Dataset root not found: {dataset_root}")

        self.slice_traffic_path = self.dataset_root / "slice_traffic"
        self.slice_mixed_path = self.dataset_root / "slice_mixed"

    def get_available_scenarios(self, slice_type: str = "slice_traffic") -> List[str]:
        """Get list of available RF scenarios."""
        base_path = self.dataset_root / slice_type
        if not base_path.exists():
            return []

        scenarios = [d.name for d in base_path.iterdir() if d.is_dir()]
        return sorted(scenarios)

    def get_training_configs(self, scenario: str, slice_type: str = "slice_traffic") -> List[str]:
        """Get list of training configurations for a scenario."""
        scenario_path = self.dataset_root / slice_type / scenario
        if not scenario_path.exists():
            return []

        configs = [d.name for d in scenario_path.iterdir() if d.is_dir() and d.name.startswith("tr")]
        return sorted(configs)

    def load_bs_data(
        self, scenario: str, training_config: str, experiment: str, base_station: str, slice_type: str = "slice_traffic"
    ) -> pd.DataFrame:
        """Load base station aggregate data."""
        file_path = (
            self.dataset_root
            / slice_type
            / scenario
            / training_config
            / experiment
            / base_station
            / f"{base_station}.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(f"BS data file not found: {file_path}")

        df = pd.read_csv(file_path)

        if "time" in df.columns and df["time"].dtype in [np.int64, np.float64]:
            df["time"] = pd.to_datetime(df["time"], unit="ms")

        return df

    def load_scenario_data(
        self, scenario: str, training_configs: Optional[List[str]] = None, slice_type: str = "slice_traffic"
    ) -> pd.DataFrame:
        """Load and concatenate data from multiple training configurations."""
        if training_configs is None:
            training_configs = self.get_training_configs(scenario, slice_type)

        all_data = []

        for tr_config in training_configs:
            tr_path = self.dataset_root / slice_type / scenario / tr_config
            if not tr_path.exists():
                continue

            for exp_dir in sorted(tr_path.iterdir()):
                if not exp_dir.is_dir() or not exp_dir.name.startswith("exp"):
                    continue

                experiment = exp_dir.name

                for bs_dir in sorted(exp_dir.iterdir()):
                    if not bs_dir.is_dir() or not bs_dir.name.startswith("bs"):
                        continue

                    bs_name = bs_dir.name
                    bs_file = bs_dir / f"{bs_name}.csv"

                    if bs_file.exists():
                        df = pd.read_csv(bs_file)

                        df["scenario"] = scenario
                        df["training_config"] = tr_config
                        df["experiment"] = experiment
                        df["base_station"] = bs_name

                        if "time" in df.columns and df["time"].dtype in [np.int64, np.float64]:
                            df["time"] = pd.to_datetime(df["time"], unit="ms")

                        all_data.append(df)

        if not all_data:
            raise ValueError(f"No data found for scenario: {scenario}")

        return pd.concat(all_data, ignore_index=True)

    def get_dataset_summary(self, slice_type: str = "slice_traffic") -> pd.DataFrame:
        """Get summary statistics of the dataset.

        Args:
            slice_type: Either "slice_traffic" or "slice_mixed"

        Returns:
            DataFrame with dataset structure summary
        """
        scenarios = self.get_available_scenarios(slice_type)

        summary_data = []
        for scenario in scenarios:
            configs = self.get_training_configs(scenario, slice_type)
            for config in configs:
                config_path = self.dataset_root / slice_type / scenario / config
                experiments = [d.name for d in config_path.iterdir() if d.is_dir() and d.name.startswith("exp")]

                summary_data.append(
                    {
                        "scenario": scenario,
                        "training_config": config,
                        "num_experiments": len(experiments),
                        "experiments": ", ".join(sorted(experiments)),
                    }
                )

        return pd.DataFrame(summary_data)
