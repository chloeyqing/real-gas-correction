"""Peng–Robinson CO2 baseline evaluated on the project's existing held-out grids.

Requires CoolProp and pandas/numpy from requirements.txt. No model fitting or
changes to the original experiment splits are performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "natural_validation": "co2_validation.csv",
    "natural_test": "co2_test.csv",
    "stress_validation": "co2_stress_validation.csv",
    "stress_test": "co2_stress_test.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tables" / "peng_robinson_baseline.csv")
    args = parser.parse_args()

    config = json.loads((ROOT / "configs" / "experiment_config.json").read_text())
    molar_mass = PropsSI("MOLARMASS", "CO2")
    rows = []
    for split, filename in DATASETS.items():
        df = pd.read_csv(args.data_dir / filename)
        required = {"T_K", "rho_molar_mol_per_m3", "p_real_Pa"}
        if not required.issubset(df.columns):
            raise ValueError(f"{filename}: expected columns {sorted(required)}; got {list(df.columns)}")
        predictions = []
        for t, rho in zip(df["T_K"], df["rho_molar_mol_per_m3"]):
            # CoolProp PR backend uses mass density (kg/m^3), not molar density.
            predictions.append(PropsSI("P", "T", float(t), "Dmass", float(rho) * molar_mass, "PR::CO2"))
        predicted = np.asarray(predictions)
        actual = df["p_real_Pa"].to_numpy(dtype=float)
        if not np.all(np.isfinite(predicted)) or np.any(actual <= 0):
            raise ValueError(f"{split}: non-finite PR predictions or invalid reference pressures")
        ape = 100.0 * np.abs(predicted - actual) / actual
        rows.append({
            "model": "peng_robinson",
            "dataset": split,
            "count": len(df),
            "pressure_MAPE_percent": float(np.mean(ape)),
            "pressure_median_error_percent": float(np.median(ape)),
            "pressure_95th_error_percent": float(np.percentile(ape, 95)),
            "pressure_max_error_percent": float(np.max(ape)),
        })
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"Saved {args.output}; reference CoolProp version: {config['coolprop_version']}")


if __name__ == "__main__":
    main()
