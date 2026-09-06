"""Reusable PySR training utility for the CO₂ correction project.

The notebooks orchestrate the complete multi-stage experiment. This module
provides a small, reproducible entry point for fitting one symbolic model from
a CSV file without depending on notebook state.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


def create_symbolic_model(
    *,
    niterations: int = 100,
    populations: int = 20,
    population_size: int = 50,
    maxsize: int = 20,
    maxdepth: int = 10,
    random_state: int = 42,
):
    """Build the project-standard PySR estimator."""

    from pysr import PySRRegressor

    return PySRRegressor(
        niterations=niterations,
        populations=populations,
        population_size=population_size,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        maxsize=maxsize,
        maxdepth=maxdepth,
        model_selection="best",
        elementwise_loss=(
            "loss(prediction, target) = "
            "(prediction - target)^2"
        ),
        parsimony=0.001,
        random_state=random_state,
        deterministic=True,
        parallelism="serial",
        verbosity=1,
    )


def train_from_csv(
    input_csv: Path,
    feature_columns: list[str],
    target_column: str,
    model_output: Path,
    equations_output: Path,
    random_state: int = 42,
):
    """Fit a symbolic model and save both the model and equation table."""

    data = pd.read_csv(input_csv)
    missing = [
        column
        for column in [*feature_columns, target_column]
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    model = create_symbolic_model(random_state=random_state)
    model.fit(
        data[feature_columns].to_numpy(),
        data[target_column].to_numpy(),
        variable_names=feature_columns,
    )

    model_output.parent.mkdir(parents=True, exist_ok=True)
    equations_output.parent.mkdir(parents=True, exist_ok=True)

    with model_output.open("wb") as file:
        pickle.dump(model, file)

    model.equations_.to_csv(equations_output, index=False)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit one project-standard PySR model from a CSV file."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("results/models/symbolic_model.pkl"),
    )
    parser.add_argument(
        "--equations-output",
        type=Path,
        default=Path("results/tables/symbolic_equations.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_from_csv(
        input_csv=args.input_csv,
        feature_columns=args.features,
        target_column=args.target,
        model_output=args.model_output,
        equations_output=args.equations_output,
        random_state=args.seed,
    )
    print(f"Saved model to {args.model_output}")
    print(f"Saved equations to {args.equations_output}")


if __name__ == "__main__":
    main()
