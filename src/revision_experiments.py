"""IMAT revision: frozen-model reproduction, C++ PR baseline, full PySR seed repeats.

Python bridges the project's existing CoolProp/PySR/scikit-learn workflow.
The independently implemented numerical PR baseline is in C++17.
Run from the repository root. Original tracked tables are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results/revision"
FEATURES = ["T_reduced", "rho_reduced"]
REGIMES = ["near_ideal", "attraction_dominated", "excluded_volume_dominated"]
FILES = {"train": "co2_train.csv", "natural_validation": "co2_validation.csv",
         "stress_validation": "co2_stress_validation.csv", "natural_test": "co2_test.csv",
         "stress_test": "co2_stress_test.csv"}
BASE_COMMIT = "fe96181c44626792f47c33c9f7313dfc8316d451"
R = 8.31446261815324


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def generate_data():
    """Reconstruct original grids and frozen manifest, without resampling isotherms."""
    config = json.loads((ROOT / "configs/experiment_config.json").read_text())
    manifest = json.loads((ROOT / "configs/split_manifest.json").read_text())
    if CP.get_global_param_string("version") != config["coolprop_version"]:
        raise RuntimeError("Install the exact original CoolProp version first")
    tc, rhoc = config["critical_temperature_K"], config["critical_molar_density_mol_per_m3"]

    def grid(temperatures, densities):
        t, rho = np.meshgrid(temperatures, densities, indexing="ij")
        t, rho = t.ravel(), rho.ravel()
        p = np.array([CP.PropsSI("P", "T", ti, "Dmolar", ri, "CO2") for ti, ri in zip(t, rho)])
        if not (np.isfinite(p).all() and (p > 0).all()):
            raise RuntimeError("Reference data contain invalid pressure")
        d = pd.DataFrame({"T_K": t, "rho_mol_m3": rho, "p_real_Pa": p,
                          "p_ideal_Pa": rho * R * t, "T_reduced": t / tc,
                          "rho_reduced": rho / rhoc})
        d["Z"] = p / d.p_ideal_Pa
        d["delta_Z"] = d.Z - 1
        d["regime"] = np.where(abs(d.delta_Z) <= .03, REGIMES[0],
                                np.where(d.delta_Z < -.03, REGIMES[1], REGIMES[2]))
        return d

    natural = grid(np.linspace(310, 500, 50), np.geomspace(100, 20000, 100))
    dense = grid(np.linspace(360, 500, 80), np.linspace(12000, 20000, 80))

    def mask(frame, temperatures):
        return np.any(np.isclose(frame.T_K.to_numpy()[:, None], temperatures, atol=1e-9), axis=1)

    keys = {"natural_validation": "natural_validation_temperatures_K",
            "natural_test": "natural_test_temperatures_K",
            "stress_validation": "high_density_validation_temperatures_K",
            "stress_test": "high_density_test_temperatures_K"}
    datasets = {name: (natural if name.startswith("natural") else dense).loc[
        mask(natural if name.startswith("natural") else dense, manifest[key])].reset_index(drop=True)
        for name, key in keys.items()}
    union = pd.concat([natural, dense], ignore_index=True).drop_duplicates(["T_K", "rho_mol_m3"])
    held = np.concatenate([manifest[key] for key in keys.values()])
    datasets["train"] = union.loc[~mask(union, held)].reset_index(drop=True)
    expected = {"train": 7799, "natural_validation": 1000, "natural_test": 1000,
                "stress_validation": 800, "stress_test": 800}
    for name, d in datasets.items():
        if len(d) != expected[name]:
            raise RuntimeError(f"Wrong {name} count: {len(d)}")
        if name != "train" and np.intersect1d(np.round(d.T_K, 10),
                                             np.round(datasets["train"].T_K, 10)).size:
            raise RuntimeError("Isotherm leakage detected")
    if datasets["train"].regime.value_counts().to_dict() != {
            REGIMES[0]: 1468, REGIMES[1]: 4324, REGIMES[2]: 2007}:
        raise RuntimeError("Original regime counts were not reproduced")
    DATA.mkdir(parents=True, exist_ok=True)
    for name, d in datasets.items():
        d.to_csv(DATA / FILES[name], index=False)
    return datasets


def metrics(d, dz, name, model):
    dz = np.asarray(dz, dtype=float)
    p = d.p_ideal_Pa.to_numpy() * (1 + dz)
    valid = np.isfinite(p) & (p > 0)
    row = {"dataset": name, "model": model, "count": len(d),
           "invalid_count": int((~valid).sum())}
    # A failed prediction invalidates the aggregate; never silently drop states.
    fields = ["pressure_MAPE_percent", "pressure_median_error_percent",
              "pressure_95th_error_percent", "pressure_max_error_percent", "delta_Z_RMSE", "delta_Z_MAE"]
    if not valid.all():
        return dict(row, **{f: np.nan for f in fields})
    ape = 100 * abs(p - d.p_real_Pa.to_numpy()) / d.p_real_Pa.to_numpy()
    err = dz - d.delta_Z.to_numpy()
    return dict(row, **dict(zip(fields, [np.mean(ape), np.median(ape), np.percentile(ape, 95),
                                        np.max(ape), np.sqrt(np.mean(err ** 2)), np.mean(abs(err))])))


def frozen_experts(d):
    x, y = d.T_reduced.to_numpy(), d.rho_reduced.to_numpy()
    return np.column_stack([
        .7075689 / (-x / y + y / (x - (.6079937 + 1.2324635 / y))),
        y * (-x - .60809004 * y + 2.3568213) / (-x + .13378955 * (x*x*y*y - x*x*x)) - .0013054596,
        .3010038 * y * (y + .0014043123) * (1.5343235*x + y - 3.7023797)])


def blend(gate, d, experts, soft=True):
    if not soft:
        classes = gate.predict(d[FEATURES])
        return experts[np.arange(len(d)), [REGIMES.index(c) for c in classes]]
    probabilities = gate.predict_proba(d[FEATURES])[:, [list(gate.classes_).index(k) for k in REGIMES]]
    return (probabilities * experts).sum(axis=1)


def make_gate(train, seed, depth):
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=10,
                                  class_weight="balanced", random_state=seed).fit(train[FEATURES], train.regime)


def baselines(datasets):
    OUT.mkdir(parents=True, exist_ok=True)
    binary = OUT / "peng_robinson_baseline"
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
                    str(ROOT / "src/peng_robinson_baseline.cpp"), "-o", str(binary)], check=True)
    gate = make_gate(datasets["train"], 47, 10)
    rows, max_crosscheck = [], 0.
    original = pd.read_csv(ROOT / "results/tables/final_test_comparison.csv")
    pr = CP.AbstractState("PR", "CO2")
    pr.specify_phase(CP.iphase_gas)
    for name, d in datasets.items():
        if name == "train":
            continue
        text = subprocess.check_output([str(binary), str(DATA / FILES[name])], text=True)
        cpp = pd.read_csv(io.StringIO(text))
        if not (np.allclose(cpp.T_K, d.T_K, rtol=0, atol=1e-12)
                and np.allclose(cpp.rho_mol_m3, d.rho_mol_m3, rtol=0, atol=1e-10)):
            raise RuntimeError("C++ baseline state alignment failed")
        cross = []
        for t, rho in zip(d.T_K, d.rho_mol_m3):
            pr.update(CP.DmolarT_INPUTS, rho, t)
            cross.append(pr.p())
        difference = float(max(abs(cpp.p_PR_Pa - np.array(cross))))
        max_crosscheck = max(max_crosscheck, difference)
        if difference > 1e-4:
            raise RuntimeError("Independent C++ PR disagrees with the CoolProp cubic backend")
        soft = blend(gate, d, frozen_experts(d))
        row = metrics(d, soft, name, "original_soft_gate")
        if name.endswith("test"):
            recorded = original[(original.dataset == name) & (original.model == "final_soft_gate_model")].iloc[0]
            if abs(row["pressure_MAPE_percent"] - recorded.pressure_MAPE_percent) > 1e-10:
                raise RuntimeError("Original soft-gate metric was not reproduced")
        rows.extend([metrics(d, np.zeros(len(d)), name, "ideal_gas"),
                     metrics(d, cpp.p_PR_Pa / d.p_ideal_Pa - 1, name, "peng_robinson"), row])
        pd.DataFrame({"T_K": d.T_K, "rho_mol_m3": d.rho_mol_m3,
                      "p_reference_Pa": d.p_real_Pa, "p_PR_Pa": cpp.p_PR_Pa,
                      "p_soft_gate_Pa": d.p_ideal_Pa * (1 + soft)}).to_csv(OUT / f"predictions_{name}.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "classical_baseline_comparison.csv", index=False)
    save_json(OUT / "baseline_checks.json", {"reference": "HEOS::CO2, CoolProp 8.0.0",
        "baseline": "unfitted pure-fluid Peng-Robinson p(T, rho)", "R": R,
        "Tc_K": 304.1282, "pc_Pa": 7377300., "omega": .22394,
        "a_coefficient": .45723552892138218938, "b_coefficient": .07779607390388455972,
        "maximum_cpp_vs_coolprop_PR_difference_Pa": max_crosscheck,
        "original_soft_gate_reproduced": True, "test_rows": {"natural_test": 1000, "stress_test": 800}})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def select_equation(model, datasets, regime, limit, directory):
    rows = []
    for index, eq in model.equations_.iterrows():
        if eq.complexity > limit:
            continue
        errors = []
        for name in ["natural_validation", "stress_validation"]:
            d = datasets[name]
            if regime is not None:
                d = d[d.regime == regime]
            pred = model.predict(d[FEATURES].to_numpy(), index=int(index))
            errors.append(float(np.sqrt(np.mean((pred - d.delta_Z.to_numpy()) ** 2))))
        score = .5 * errors[0] + .5 * errors[1]
        if not np.isfinite(score):
            continue
        rows.append({"index": int(index), "complexity": int(eq.complexity),
                     "natural_validation_RMSE": errors[0], "stress_validation_RMSE": errors[1],
                     "selection_score": score, "equation": str(eq.equation)})
    if not rows:
        raise RuntimeError(f"No finite validation candidate for {regime}")
    selection = pd.DataFrame(rows).sort_values(["selection_score", "complexity", "index"])
    selection.to_csv(directory / "validation_selection.csv", index=False)
    return int(selection.iloc[0]["index"])


def repeat_seed(datasets, seed):
    from train_pysr import create_symbolic_model

    directory = OUT / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    models, indices, selections, seed_map = {}, {}, {}, {}
    specs = [("global", None, 0, False), (REGIMES[0], REGIMES[0], 1, False),
             ("attraction_standard", REGIMES[1], 2, False),
             (REGIMES[2], REGIMES[2], 3, False), (REGIMES[1], REGIMES[1], 4, True)]
    for label, regime, offset, expanded in specs:
        fitdir = directory / label
        fitdir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        model = create_symbolic_model(random_state=seed + offset)
        model.set_params(verbosity=0, progress=False, output_directory=str(fitdir), run_id="search")
        if expanded:
            model.set_params(niterations=150, population_size=60, unary_operators=["square"],
                             maxsize=25, maxdepth=12)
        d = datasets["train"]
        if regime is not None:
            d = d[d.regime == regime]
        print(f"Seed {seed}: fit {label}, RNG seed {seed+offset}, n={len(d)}", flush=True)
        model.fit(d[FEATURES].to_numpy(), d.delta_Z.to_numpy(), variable_names=["T_r", "rho_r"])
        index = select_equation(model, datasets, regime, 25 if expanded else 15, fitdir)
        with (fitdir / "model.pkl").open("wb") as f:
            pickle.dump(model, f)
        model.equations_.drop(columns=["lambda_format"], errors="ignore").to_csv(fitdir / "pareto_front.csv", index=False)
        models[label], indices[label] = model, index
        seed_map[label] = seed + offset
        selections[label] = {"index": index, "equation": str(model.sympy(index=index)),
                             "seed": seed + offset, "fit_seconds": time.monotonic() - started}
        print(f"Seed {seed}: selected {label} = {selections[label]['equation']}", flush=True)

    def experts(d):
        return np.column_stack([models[k].predict(d[FEATURES].to_numpy(), index=indices[k]) for k in REGIMES])

    gate_rows, candidates = [], {}
    for depth in [3, 4, 5, 6, 8, 10, 12]:
        gate = make_gate(datasets["train"], seed + 5, depth)
        candidates[depth] = gate
        scores = [metrics(datasets[name], blend(gate, datasets[name], experts(datasets[name]), False),
                          name, "hard_gate") for name in ["natural_validation", "stress_validation"]]
        if all(r["invalid_count"] == 0 for r in scores):
            gate_rows.append({"depth": depth,
                              "mean_validation_MAPE": np.mean([r["pressure_MAPE_percent"] for r in scores]),
                              "worst_validation_error": max(r["pressure_max_error_percent"] for r in scores)})
    if not gate_rows:
        raise RuntimeError("No gate has valid validation predictions")
    gs = pd.DataFrame(gate_rows).sort_values(["mean_validation_MAPE", "worst_validation_error", "depth"])
    gs.to_csv(directory / "gate_validation_selection.csv", index=False)
    depth = int(gs.iloc[0].depth)
    gate = candidates[depth]
    # Persist all choices before inspecting either test set.
    save_json(directory / "selection_frozen_before_test.json", {
        "master_seed": seed, "symbolic_seeds": seed_map, "gate_seed": seed+5,
        "selected_depth": depth, "selection_data": ["natural_validation", "stress_validation"],
        "symbolic_selections": selections})
    with (directory / "gate.pkl").open("wb") as f:
        pickle.dump(gate, f)
    rows = []
    for name in ["natural_validation", "stress_validation", "natural_test", "stress_test"]:
        d = datasets[name]
        fs = experts(d)
        predictions = {"global_symbolic": models["global"].predict(d[FEATURES].to_numpy(), index=indices["global"]),
                       "oracle": fs[np.arange(len(d)), [REGIMES.index(k) for k in d.regime]],
                       "hard_gate": blend(gate, d, fs, False), "soft_gate": blend(gate, d, fs)}
        for label, pred in predictions.items():
            row = metrics(d, pred, name, label)
            row.update(master_seed=seed, gate_depth=depth)
            rows.append(row)
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    print(pd.DataFrame(rows)[lambda x: x.dataset.str.endswith("test")].to_string(index=False), flush=True)
    return rows


def provenance(seeds):
    save_json(OUT / "provenance.json", {
        "base_commit": BASE_COMMIT, "planned_master_seeds": seeds,
        "original_seeds": {"global":42, "near_ideal":43, "attraction_standard":44,
                           "excluded_volume_dominated":45, "attraction_expanded":46, "gate":47},
        "seed_policy": "predeclared master seeds; offsets 0..5 match original search roles; no seed selected using test errors",
        "python":sys.version, "platform":platform.platform(),
        "versions":{m:importlib.metadata.version(m) for m in ["CoolProp","pysr","numpy","pandas","scikit-learn"]},
        "sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [ROOT / "configs/experiment_config.json", ROOT / "configs/split_manifest.json",
                   ROOT / "src/train_pysr.py", ROOT / "src/revision_experiments.py",
                   ROOT / "src/peng_robinson_baseline.cpp", *[DATA / f for f in FILES.values()]]}})
    (OUT / "python_environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=[52, 62, 72])
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    datasets = generate_data()
    baselines(datasets)
    provenance(args.seeds)
    if args.baseline_only:
        return
    rows = []
    for seed in args.seeds:
        rows.extend(repeat_seed(datasets, seed))
        pd.DataFrame(rows).to_csv(OUT / "repeated_seed_metrics.csv", index=False)
    test = pd.DataFrame(rows)
    test = test[(test.model == "soft_gate") & test.dataset.str.endswith("test")]
    summary = test.groupby("dataset").agg(
        repetitions=("master_seed", "count"), failed_predictions=("invalid_count", "sum"),
        mean_MAPE=("pressure_MAPE_percent", "mean"), sample_SD_MAPE=("pressure_MAPE_percent", "std"),
        min_MAPE=("pressure_MAPE_percent", "min"), max_MAPE=("pressure_MAPE_percent", "max"))
    if (test.invalid_count > 0).any():
        # Do not summarize only the successful seeds as if all trials succeeded.
        summary.loc[summary.failed_predictions > 0, ["mean_MAPE","sample_SD_MAPE","min_MAPE","max_MAPE"]] = np.nan
    summary.to_csv(OUT / "repeated_seed_summary.csv")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
