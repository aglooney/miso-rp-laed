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
