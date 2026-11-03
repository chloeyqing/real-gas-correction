# IMAT 2026 revision experiments

The revision extends the originally reported run at commit
`fe96181c44626792f47c33c9f7313dfc8316d451`. Its equations and original
tracked evaluation tables are retained. New results are in `results/revision/`.

## Frozen-data reproduction and classical EOS comparison

`src/revision_experiments.py` reconstructs the original CoolProp 8.0.0 grids
and applies `configs/split_manifest.json`. It checks the training count (7,799),
the training regime counts (1,468 / 4,324 / 2,007), and the absence of shared
training/evaluation isotherms. The original equation strings and a refitted
depth-10 gate with seed 47 reproduce the archived natural/stress test MAPE
to within `1e-10` percentage points.

`src/peng_robinson_baseline.cpp` computes pressure directly from temperature
and molar density using the pure-fluid Peng-Robinson equation, with no fit,
volume translation, or pressure-to-density flash. Constants are
`R = 8.31446261815324 J/(mol K)`, `Tc = 304.1282 K`, `pc = 7377300 Pa`,
and `omega = 0.22394`. The full-precision cubic coefficients match the
[CoolProp 8.0.0 source](https://github.com/CoolProp/CoolProp/blob/v8.0.0/src/Backends/Cubics/GeneralizedCubic.cpp).
The C++ pressures are cross-checked at every validation/test state against
the CoolProp PR backend, with tolerance `1e-4 Pa`. All states are retained.

Run only the reproduction and classical comparison:

```sh
python3 src/revision_experiments.py --baseline-only
```

The C++ baseline can also be compiled and run independently:

```sh
g++ -std=c++17 -O2 src/peng_robinson_baseline.cpp -o /tmp/co2_pr
/tmp/co2_pr data/processed/co2_test.csv > /tmp/co2_pr_predictions.csv
```

## Full training repeats

Three additional master seeds were declared before evaluation: **52, 62, 72**.
For each master seed `s`, the original search roles use the following seeds:

| Search role | RNG seed |
| --- | --- |
| Global symbolic model | `s` |
| Standard near-ideal expert | `s + 1` |
| Standard attraction expert | `s + 2` |
| Standard excluded-volume expert | `s + 3` |
| Expanded attraction expert | `s + 4` |
| Decision-tree gate | `s + 5` |

These are full training repeats, rather than gate-only perturbations. Each
standard symbolic search uses 100 iterations, 20 populations, population size
50, size/depth limits 20/10, the four binary arithmetic operators, parsimony
0.001, deterministic serial execution, and squared loss. The expanded
attraction search retains 150 iterations, population size 60, the square
operator, and size/depth limits 25/12.

Equation selection uses the equally weighted natural/stress validation RMSE
with selected-complexity limits 15 (standard) and 25 (expanded). The expanded
attraction model replaces the standard attraction model, matching the original
notebook protocol. Gate depth is selected from 3, 4, 5, 6, 8, 10, 12 by mean
hard-gate validation MAPE, then worst validation error, then lower depth for a
tie. Every selection is saved before either test set is evaluated. Test errors
never choose a seed, equation, or depth.

Run all three repeats:

```sh
python3 src/revision_experiments.py --seeds 52 62 72
python3 src/generate_figures.py --include-pr --output-dir results/revision/figures
```

Use the pinned Python packages in `results/revision/requirements_revision_lock.txt`.
The revision ran with Python 3.12.14, Julia 1.13.3, PySR 2.2.1,
SymbolicRegression.jl 2.2.1, and DynamicExpressions.jl 2.10.1. Julia project
and manifest snapshots are included alongside the metrics. The original
training environment was not fully archived, so the revision repeats should
not be treated as a bitwise re-execution of the historical training run.

## Reading the artifacts

- `classical_baseline_comparison.csv`: mean, median, 95th percentile, maximum
  pressure APE and residual errors for the ideal gas, PR, and original soft gate.
- `predictions_*.csv`: per-state reference, PR, and original soft-gate pressures.
- `baseline_checks.json`: numerical cross-checks and baseline constants.
- `repeated_seed_metrics.csv`: all global/oracle/hard/soft validation and test
  metrics, with invalid-prediction counts.
- `repeated_seed_summary.csv`: mean, sample SD (`ddof=1`), and range of the
  three additional soft-gate repeats. The historical run is not included.
- `seed_*/selection_frozen_before_test.json`: all selected equations and gate
  depths; `validation_selection.csv` and `pareto_front.csv` preserve candidates.
- `provenance.json`: base source commit, declared seeds, package versions,
  and SHA-256 hashes of source/configuration/data used when the run started.

Any non-finite or nonpositive pressure prediction invalidates that aggregate;
the implementation never silently removes difficult states. The repeats
demonstrate stochastic-search variability, not experimental uncertainty.
Claims remain restricted to approximation of the selected reference EOS over
the documented CO2 domain. Serialized model/checkpoint files remain local;
they and exact data still require a durable archival release for publication.
