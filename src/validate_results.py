"""Compute baselines and error-tail summaries from generated project data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "natural_validation": "co2_validation.csv",
    "natural_test": "co2_test.csv",
    "stress_validation": "co2_stress_validation.csv",
    "stress_test": "co2_stress_test.csv",
}


def evaluate_prediction(
    dataset: pd.DataFrame,
    predicted_delta_z: np.ndarray,
    model: str,
    dataset_name: str,
) -> dict[str, float | str]:
    true_delta_z = dataset["delta_Z"].to_numpy()
    ideal_pressure = dataset["p_ideal_Pa"].to_numpy()
    true_pressure = dataset["p_real_Pa"].to_numpy()
    predicted_pressure = ideal_pressure * (1.0 + predicted_delta_z)

    delta_error = predicted_delta_z - true_delta_z
    pressure_error = (
        np.abs(predicted_pressure - true_pressure)
        / np.abs(true_pressure)
        * 100.0
    )

    return {
        "model": model,
        "dataset": dataset_name,
        "count": len(dataset),
        "delta_Z_RMSE": float(np.sqrt(np.mean(delta_error**2))),
        "delta_Z_MAE": float(np.mean(np.abs(delta_error))),
        "pressure_MAPE_percent": float(np.mean(pressure_error)),
        "pressure_median_error_percent": float(np.median(pressure_error)),
        "pressure_95th_error_percent": float(np.percentile(pressure_error, 95)),
        "pressure_max_error_percent": float(np.max(pressure_error)),
    }


def build_ideal_gas_baseline(data_dir: Path, output_path: Path) -> pd.DataFrame:
    rows = []
    for dataset_name, filename in DATASETS.items():
        dataset = pd.read_csv(data_dir / filename)
        rows.append(
            evaluate_prediction(
                dataset,
                np.zeros(len(dataset), dtype=float),
                "ideal_gas",
                dataset_name,
            )
        )

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "tables"
        / "ideal_gas_baseline.csv",
    )
    args = parser.parse_args()
    result = build_ideal_gas_baseline(args.data_dir, args.output)
    print(result.to_string(index=False))
    print(f"Saved baseline to {args.output}")


if __name__ == "__main__":
    main()
