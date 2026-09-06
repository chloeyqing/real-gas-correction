"""Regenerate the figures referenced by the project README."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGIME_ORDER = [
    "near_ideal",
    "attraction_dominated",
    "excluded_volume_dominated",
]
REGIME_LABELS = {
    "near_ideal": "Near ideal",
    "attraction_dominated": "Negative residual",
    "excluded_volume_dominated": "Positive residual",
}


def plot_pressure_mape(repo_root: Path, output_dir: Path) -> None:
    table = pd.read_csv(
        repo_root / "results" / "tables" / "final_test_comparison.csv"
    )
    baseline_path = repo_root / "results" / "tables" / "ideal_gas_baseline.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        if not (table["model"] == "ideal_gas").any():
            table = pd.concat([table, baseline], ignore_index=True)

    table = table.drop_duplicates(
        subset=["model", "dataset"],
        keep="last",
    )

    table = table[table["dataset"].isin(["natural_test", "stress_test"])]
    model_order = [
        "ideal_gas",
        "global_symbolic",
        "oracle_regime_specific",
        "hard_gate",
        "final_soft_gate_model",
    ]
    model_labels = {
        "ideal_gas": "Ideal gas",
        "global_symbolic": "Global symbolic",
        "oracle_regime_specific": "Oracle regime-specific",
        "hard_gate": "Hard gate",
        "final_soft_gate_model": "Final soft gate",
    }
    pivot = table.pivot(index="model", columns="dataset", values="pressure_MAPE_percent")
    pivot = pivot.reindex(model_order).dropna(how="all")

    ax = pivot.rename(index=model_labels).plot.bar(
        figsize=(11, 6),
        color={"natural_test": "#4C78A8", "stress_test": "#F58518"},
    )
    ax.set_title("Held-out pressure MAPE comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Pressure MAPE (%)")
    ax.legend(["Natural test", "Stress test"], title="Dataset")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "model_pressure_mape_comparison.png", dpi=220)
    plt.close()


def plot_regime_distribution(repo_root: Path, output_dir: Path) -> None:
    data_path = repo_root / "data" / "processed" / "co2_train.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            "Run 01_CO2_Data_Generation.ipynb before generating the regime figure."
        )
    data = pd.read_csv(data_path)
    colors = {
        "near_ideal": "#4C78A8",
        "attraction_dominated": "#E45756",
        "excluded_volume_dominated": "#59A14F",
    }
    markers = {
        "near_ideal": "o",
        "attraction_dominated": "s",
        "excluded_volume_dominated": "^",
    }
    fig, ax = plt.subplots(figsize=(10, 7))
    for regime in REGIME_ORDER:
        subset = data[data["regime"] == regime]
        ax.scatter(
            subset["T_reduced"],
            subset["rho_reduced"],
            s=12,
            alpha=0.55,
            color=colors[regime],
            marker=markers[regime],
            label=REGIME_LABELS[regime],
            linewidths=0,
        )
    ax.set_title("Final training split by net compressibility residual")
    ax.set_xlabel(r"Reduced temperature, $T_r$")
    ax.set_ylabel(r"Reduced molar density, $\rho_r$")
    ax.set_yscale("log")
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "regime_distribution_map.png", dpi=220)
    plt.close()


def plot_prediction_vs_reference(repo_root: Path, output_dir: Path) -> None:
    data = pd.read_csv(
        repo_root
        / "results"
        / "tables"
        / "inference_sanity_check_20_points.csv"
    )
    reference = data["p_coolprop_Pa"].to_numpy() / 1e6
    predicted = data["p_prediction_MPa"].to_numpy()
    error = data["pressure_error_percent"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 7))
    low = min(reference.min(), predicted.min())
    high = max(reference.max(), predicted.max())
    ax.scatter(reference, predicted, c=error, cmap="viridis", s=70)
    ax.plot([low, high], [low, high], "k--", label="Perfect agreement")
    ax.set_title("Predicted pressure versus CoolProp reference")
    ax.set_xlabel("CoolProp pressure (MPa)")
    ax.set_ylabel("Predicted pressure (MPa)")
    ax.text(
        0.04,
        0.94,
        f"Mean error = {error.mean():.3f}%\nMax error = {error.max():.3f}%",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85},
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    colorbar = fig.colorbar(ax.collections[0], ax=ax)
    colorbar.set_label("Absolute pressure error (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_vs_coolprop.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    output_dir = args.repo_root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_pressure_mape(args.repo_root, output_dir)
    plot_regime_distribution(args.repo_root, output_dir)
    plot_prediction_vs_reference(args.repo_root, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
