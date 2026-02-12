MISO Rolling-Window Dispatch Models (LAED / RP) with Storage
============================================================

Contents
 - `laed_rp_with_storage.py`: rolling stochastic versions (ED and LAED) with storage, forecast errors.
 - `laed_rp_storage_no_error.py`: deterministic counterparts (no forecast errors) with storage.
 - `laed_rp_analysis.py`: baseline deterministic analysis without storage.
 - `laed_rp_random_error.py`: original stochastic code without storage.
 - Data files: `toy_data.dat` (no storage), `toy_data_storage.dat` (adds storage params), `MISO_Projection.json` (hourly load projection).
 - Utilities/tests: `rp_test.py`, `laed_test.py`, `data_exploration.py`.

Prerequisites
 - Python 3.10+ with Pyomo and matplotlib; Gurobi available and licensed (scripts default to `gurobi_direct`).
 - Install deps: `pip install -r requirements.txt` (Pyomo, tqdm, matplotlib, numpy, gurobipy).

Key Concepts
 - Rolling-window energy dispatch with ramping and reserve constraints.
 - LAED (look-ahead economic dispatch) vs RP/ED (single-window reserve product).
 - Storage modeled with continuous charge/discharge, SoC balance, efficiency, and capacity limits; remains a linear program.
 - Two modes:
     * Stochastic forecast error: `laed_rp_with_storage.py` (Gaussian/Laplace/Student-t innovations, correlated across windows).
     * Deterministic/no-error: `laed_rp_storage_no_error.py` and `laed_rp_analysis.py`.

Data Inputs
 - `toy_data_storage.dat` defines generators, ramp limits, loads, and storage parameters:
     * `N_s`, `E_cap`, `P_ch_cap`, `P_dis_cap`, `SoC_init`, `SoC_min`, `SoC_max`, `charge_cost`, `discharge_cost`, `eta_c`, `eta_d`, `delta_t`.
 - `toy_data.dat` lacks storage; use it only with non-storage scripts.
 - `MISO_Projection.json` provides monthly load series; scripts rescale August 2032 to a reference capacity (`ref_cap` in code).

Running the deterministic storage model
 - From repo root:
   `python3 laed_rp_storage_no_error.py`
 - The script:
     * Loads `toy_data_storage.dat`.
     * Rescales Aug-2032 load to `ref_cap=350`.
     * Sweeps commitment window sizes 1..19 hours.
     * Solves ED and LAED for each window, plots load shedding and storage SoC/charge/discharge for the final sweep run.

Running the stochastic storage model
 - `python3 laed_rp_with_storage.py`
 - Same sweep as above, but adds correlated forecast errors controlled by:
     * `sigma` (relative std), `rho` (error correlation), `error_type` (`student-t` default).
 - Outputs LMP/TLMP and storage trajectories for the last sweep.

Switching data cases
 - Change `case_name` near the bottom of each script to point to `toy_data_storage.dat` (with storage) or another `.dat` constructed similarly such as `10GEN_MASKED.dat`.
 - To alter storage size, edit `toy_data_storage.dat` parameters; no code changes needed.

Plot outputs
 - Load-shedding vs window size.
 - For final sweep run: SoC and charge/discharge power for each storage unit for both ED (RP) and LAED.

Notes on feasibility/debugging
 - Keep `eta_c`, `eta_d` ≤1; `SoC_min`/`SoC_max` within `E_cap`.
 - If infeasible, reduce ramp factors or storage power limits, and ensure `Gen_init` is feasible for the first load.
 - All formulations are LPs; if you add binary no-simultaneity, they will become MILPs.

Directory hygiene
 - `.gitignore` should include solver logs and `__pycache__/`; adjust as needed.
