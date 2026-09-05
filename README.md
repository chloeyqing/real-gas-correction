# Ideal Gas Correction Model

Physics-aware symbolic correction of the ideal-gas law for CO₂.

## Run order

Run the notebooks in this order:

1. 01_CO2_Data_Generation.ipynb — generate CoolProp data and frozen splits.
2. 02_Global_Symbolic_Regression.ipynb — train and select the global model.
3. 03_Regime_Specific_Regression.ipynb — train regime models and save final equation indices.
4. 04_Regime_Gate.ipynb — train the gate and evaluate the deployable soft-gate model.
5. 05_Final_Inference.ipynb — reload the saved artifacts, make predictions from `T_K` and `rho_mol_m3`, compare with CoolProp, and run a small stress sanity check.

Expected repository layout:

    data/processed/
    results/models/
    results/tables/
    notebooks/

The data split holds out complete temperature isotherms. The stress split is a separate high-density grid. Regimes are defined by ΔZ = Z − 1 with epsilon = 0.03.

The oracle_regime_specific result is an upper bound because it receives the true regime. The deployable model is final_soft_gate_model, which predicts regime probabilities from T_reduced and rho_reduced.

The serialized deployable artifacts are `final_regime_models.pkl`, `final_regime_indices.json`, `final_gate.pkl`, and `gate_config.json`. The inference notebook expects temperature in kelvin and molar density in mol/m³.

Attraction equation selection is performed using natural and stress validation only. The test sets are evaluated after the selection is frozen.
