# Ideal Gas Correction Model

Physics-aware symbolic correction of the ideal-gas law for CO₂.

## Run order

Run the notebooks in this order:

1. 01_CO2_Data_Generation.ipynb — generate CoolProp data and frozen splits.
2. 02_Global_Symbolic_Regression.ipynb — train and select the global model.
3. 03_Regime_Specific_Regression.ipynb — train regime models and save final equation indices.
4. 04_Regime_Gate.ipynb — train the gate and evaluate the deployable soft-gate model.

Expected repository layout:

    data/processed/
    results/models/
    results/tables/
    notebooks/

The data split holds out complete temperature isotherms. The stress split is a separate high-density grid. Regimes are defined by ΔZ = Z − 1 with epsilon = 0.03.

The oracle_regime_specific result is an upper bound because it receives the true regime. The deployable model is final_soft_gate_model, which predicts regime probabilities from T_reduced and rho_reduced.

Attraction equation selection is performed using natural and stress validation only. The test sets are evaluated after the selection is frozen.
