# Experiments: Pricing Volatility Sweep

This folder contains a lightweight experiment runner + logging harness for the pricing-volatility comparisons used in the NAPS paper draft.

## Quick start

Run the default sweep defined in `experiments/config.yaml` and write outputs under a run folder:

```bash
python -m experiments.run_sweep --config experiments/config.yaml --out runs
```

Select subsets:

```bash
python -m experiments.run_sweep \
  --config experiments/config.yaml \
  --out runs \
  --systems 2gen 10gen \
  --mechanisms lmp la tlmp \
  --ramp_regimes alphahigh alphamed alphalow \
  --horizons 13
```

Prereqs: install `requirements.txt` and ensure the configured Pyomo solver (default `gurobi_direct`) is available/licensed.

## Output layout

For each scenario (system × ramp_regime × horizon), outputs are written to:

`<out>/<run_name>/<scenario_id>/`

Run-level files written at `<out>/<run_name>/`:
- `config.yaml` (exact config file used for the run)
- `summary_metrics.csv`
- `summary_metrics.md`

Per-scenario files:
- `meta.json` (config + environment metadata + runtime)
- `timeseries.csv` (at minimum: `t`, `demand`, `pi_energy`)
- `metrics.json` (volatility metrics for the available series)
- `make_whole.csv` (make-whole payments by generator and mechanism)
- `ramp_diagnostics.json` (load ramp vs ramp-limit diagnostics + TLMP activity)
- `log.txt` (full scenario logs)
- `FAILED.txt` (only if the scenario errors; sweep continues)

When enabled via `--mechanisms`, `timeseries.csv` also includes mechanism-specific columns like:
- `pi_lmp_energy`, `pi_la_energy`, `pi_tlmp_load`, `pi_tlmp_marginal`
- `p_ed_g*` and/or `p_laed_g*` (committed dispatch)

`p_g*` is always written when dispatch is available for the selected `pi_energy_source` (see `meta.json`).

The `--no-run-subdir` flag restores the legacy layout that writes directly to `<out>/`.

## Notes
- `experiments/config.yaml` is **JSON-compatible YAML** so it can be parsed without extra dependencies. If you prefer full YAML syntax, install `pyyaml` and the runner will use it automatically.
- `ramp_regimes.*.multiplier` can be a single float (applies to all systems) or a mapping like `{ "2gen": 0.15, "10gen": 0.3 }`.
- Make-whole uses linear costs: `MW_i = max(0, sum_t (c_i p_{i,t} - pi_t p_{i,t}) * dt_hours)`. TLMP settles each generator at its own TLMP.
