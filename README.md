<<<<<<< HEAD
# MISO Rolling-Window Dispatch Models (LAED/RP) with Storage

Research code for rolling-window economic dispatch experiments in Python/Pyomo, including:

- **RP/ED**: single-window dispatch with ramp/reserve product-style constraints (commit the current interval).
- **LAED**: look-ahead economic dispatch solved in a receding-horizon loop (commit the first interval each window).
- Optional **storage** (continuous charge/discharge + SoC dynamics) and **stochastic load forecast errors**.

This repo is script-first (not packaged); run commands from the repo root.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python laed_rp_storage_no_error.py
```

### Solver

Most scripts default to `SolverFactory("gurobi_direct")`. If you don't have Gurobi, switch to a Pyomo solver you do have installed (e.g. `highs`, `glpk`, `cbc`), or use `--solver` for the CLI scripts below.

## Repository layout

Core models:

- `laed_rp_analysis.py`: baseline deterministic RP/ED and LAED models (no storage).
- `laed_rp_with_storage.py`: rolling RP/ED and LAED with optional storage and correlated forecast errors.
- `laed_rp_storage_no_error.py`: deterministic (no-forecast-error) runs using the storage model.
- `laed_rp_random_error.py`: earlier stochastic (no-storage) experiments.

CLI analysis scripts (run with `--help`):

- `compare_shedding_storage.py`: compare total load shedding with vs without storage (same forecast-error path).
- `battery_size_dispatch_analysis.py`: sweep battery sizing and write summary CSV/NPZ + plots.
- `storage_attempt2_nt_sensitivity.py`: sweep look-ahead window size `N_t` and plot total load shed.
- `uq_demand_error_compare.py`, `uq_sigma_shedding_plot.py`: uncertainty / demand error experiments.
- `price_vol_compare.py`: price/volatility comparison utilities.

Data & notebooks:

- `.dat` case files: `toy_data.dat`, `toy_data_storage.dat`, `10GEN_MASKED.dat`
- Demand projection: `MISO_Projection.json` (e.g. `2032_Aug` is 288 × 5-minute intervals)
- Notebooks: `notebooks/`
- Generated plots: `figures/` (ignored by default via `.gitignore`)

## Data formats

`.dat` files are loaded via `pyomo.environ.DataPortal`.

### Storage (current format)

Storage-enabled scripts read per-storage-unit parameters:

- `N_s`
- `E_cap`, `P_ch_cap`, `P_dis_cap`
- `SoC_init`, `SoC_min`, `SoC_max`
- `charge_cost`, `discharge_cost`
- `eta_c`, `eta_d`, `delta_t`

If a case file does **not** define `N_s`, the storage model runs with `N_s=0` (no storage). Some older `.dat` files in this repo contain legacy fields like `storage_capacity`/`storage_duration`; those are currently not consumed by the multi-storage formulation.

## Common runs

Baseline (no storage):

```bash
python laed_rp_analysis.py
```

Deterministic storage:

```bash
python laed_rp_storage_no_error.py
```

Stochastic storage (forecast errors):

```bash
python laed_rp_with_storage.py
```

Compare shedding with/without storage:

```bash
python compare_shedding_storage.py --help
mkdir -p figures
python compare_shedding_storage.py --case toy_data_storage.dat --laed-window 13 --sigma 0.05 --rho 0.99 --seed 42 --out figures/shedding_compare.png --no-show
```

Battery sizing sweep:

```bash
python battery_size_dispatch_analysis.py --help
python battery_size_dispatch_analysis.py --case toy_data_storage.dat --laed-window 13 --avg-demands 300,350,400 --p-fracs 0,0.05,0.1 --out-dir battery_size_dispatch_results --no-show
```

## Notes / troubleshooting

- Some scripts set `case_name = ...` near the bottom instead of using CLI args; edit that value to switch `.dat` cases.
- Many `.dat` files set `Gen_init = 0`; several scripts overwrite `Gen_init` with a feasible ED initialization to avoid artificial load shedding at the first commit step.
- Storage is modeled as an LP (continuous charge/discharge). If you add no-simultaneous-charge/discharge binaries, it becomes a MILP.
- For GitHub hygiene, avoid committing local virtual environments (`.venv/`, `newvenv/`, `miso_venv/`) and caches (`__pycache__/`, `.ipynb_checkpoints/`).

## License

Add a license file if you plan to share/reuse this code publicly.
=======
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
>>>>>>> origin/main
