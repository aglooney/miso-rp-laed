# Codex Agent Task: Experiment Runner + Logging Harness (NAPS volatility paper)

## Goal
Implement a single command that runs a full experiment sweep for pricing volatility comparisons and logs all results reproducibly, without manual intervention.

We already have a "simulation sort of setup" in this repo. Your job is to:
1) discover the existing simulation entrypoint(s),
2) create a runner script that calls them across scenarios (2-gen and 10-gen),
3) log everything,
4) write all results to disk in a consistent folder structure,
5) produce summary tables used directly in the paper.

## Paper context
We are comparing energy-price volatility across:
- Single-interval LMP (energy price = lambda_t from single-interval problem)
- Look-ahead LA price (energy price = lambda_t from look-ahead problem)
- TLMP (generator-specific) defined as:
  pi_TLMP[g,t] = lambda_t + (mu_bar[g,t] - mu_under[g,t]) - (mu_bar[g,t+1] - mu_under[g,t+1])

Volatility metrics to compute (for a single time series pi_t):
- std dev sigma
- mean absolute intertemporal change V_MA
- max intertemporal spike Delta_max
- optional: total variation TV

Important: TLMP is (g,t)-indexed. For volatility comparison, define a single TLMP time series using ONE of the following options, chosen by best availability in code:
A) Marginal-unit TLMP: pick generator g* at time t with nonzero dispatch and not at bounds, or by max |dL/dp| proxy. If hard, use:
B) Price paid by load: pi_load_t = sum_g (p[g,t]/d[t]) * pi_TLMP[g,t] (dispatch-weighted TLMP).
Implement B by default if marginal unit detection is messy. Save BOTH if easy.

## What to build
Create:

1) `experiments/run_sweep.py` (main runner)
2) `experiments/config.yaml` (scenario definitions)
3) `experiments/utils.py` (helpers: metrics, io)
4) `experiments/README.md` (how to run)

If the repo already has an experiments folder, integrate cleanly.

### Runner requirements
- Must be runnable as:
  `python -m experiments.run_sweep --config experiments/config.yaml --out runs/2026-03-03_test`
- Must support selecting subsets:
  `--systems 2gen 10gen --mechanisms lmp la tlmp --ramp_regimes loose moderate tight --load_profiles smooth aggressive`
- Must set a random seed and record it in outputs (even if deterministic).
- Must not require interactive input.

### Discover existing simulation API
Search the repo for:
- functions like `run_sim`, `simulate`, `solve_dispatch`, `build_model`, `LAED`, `TLMP`.
- scripts/notebooks that currently produce price time series.

Decide on a stable internal API adapter:
- Create wrapper functions inside `experiments/utils.py` such as:
  - `run_single_interval_case(system, load, ramps, seed) -> dict`
  - `run_lookahead_case(system, load, ramps, horizon, seed) -> dict`
where dict returns all primal/dual time series needed.

### Output contract (must write these files per run)
For each scenario (system x load_profile x ramp_regime x mechanism):
Write into:
`<out>/<scenario_id>/`

Files:
- `meta.json` with full config, git commit hash (if available), python version, solver name/version if detectable, timestamp
- `timeseries.csv` with columns:
  - t, demand
  - for each generator: p_g (if available)
  - lambda (if available)
  - mu_bar_g_t, mu_under_g_t (if available)
  - pi_energy (the energy price series used for volatility)
  - if TLMP: pi_tlmp_load (and pi_tlmp_marginal if implemented)
- `metrics.json` with volatility metrics for each relevant series:
  - lmp_energy, la_energy, tlmp_load, tlmp_marginal
- `log.txt` full logs for that scenario

Also write global summary outputs at `<out>/`:
- `summary_metrics.csv` one row per scenario_id per series, with sigma, V_MA, Delta_max, TV, mean price
- `summary_metrics.md` pretty table for quick copy into paper draft

### Logging requirements
- Use Python `logging` with both console + file handlers.
- Include scenario_id in every log line.
- Log start/end time and wall-clock runtime per scenario.
- If a scenario fails, catch exception, log stack trace, write `FAILED.txt`, and continue to next scenario.

### Plotting (optional but useful)
If matplotlib is already in repo deps, create:
- `<scenario_id>/prices.png` plot energy price series for mechanisms present
- `<scenario_id>/ramp_duals.png` plot (mu_bar - mu_under) aggregated or selected generator(s)
Keep plots simple and readable.

## Config.yaml design
Make `experiments/config.yaml` with:
- systems: {2gen: path or inline params, 10gen: ...}
- load_profiles: smooth, aggressive (either file paths or param generators)
- ramp_regimes: loose/moderate/tight as multipliers on base ramp limits
- mechanisms: lmp, la, tlmp
- lookahead: horizon H list (default [4]) and rolling window settings if applicable
- solver settings

If repo already has system/load data files, reference those.

## Implementation details
- Use pathlib for paths.
- Use pandas for CSV if available; otherwise csv module.
- Volatility metrics:
  - sigma: sample std (ddof=1)
  - V_MA: mean(|diff|)
  - Delta_max: max(|diff|)
  - TV: sum(|diff|)
- Ensure time indices align; if your sim uses 0..T-1, keep that.

## Acceptance criteria (must pass)
1) `python -m experiments.run_sweep --help` works.
2) Running default config produces:
   - an output folder with per-scenario subfolders
   - summary_metrics.csv
3) If one scenario raises an exception, others still run.
4) timeseries.csv includes pi_energy and demand at minimum.
5) metrics.json contains all four metrics for each available series.

## Deliverables
Commit the new files plus any small refactors needed to expose simulation functions cleanly.
Do not change scientific logic of existing solvers unless required to expose outputs.

## Notes to you (Codex)
- Prefer minimal invasive changes.
- If simulation is notebook-only, extract the key functions into a module and keep notebooks unchanged.
- If you cannot find TLMP ramp duals, log that TLMP is skipped and still compute LMP/LA metrics.

---

# LOC Metric Implementation (Pyomo + Gurobi, 5-minute intervals)

## Goal

Add computation of **Lost Opportunity Cost (LOC)** for each generator (i) under each price mechanism. Use the following definitions:

### Realized profit under dispatch (g_{i,t})

\[
\Pi_i^{real}(\pi) = \sum_{t=0}^{T-1}\big(\pi_t, g_{i,t} - f_i(g_{i,t})\big)\Delta t
\]

### Self-scheduling maximum profit

\[
Q_i(\pi) = \max_{p_{i,0},\dots,p_{i,T-1}} \sum_{t=0}^{T-1}\big(\pi_t, p_{i,t} - f_i(p_{i,t})\big)\Delta t
\]
subject to:
\[
0 \le p_{i,t} \le \bar g_i \quad \forall t
\]
\[
-\underline r_i \le p_{i,t+1}-p_{i,t} \le \overline r_i \quad \forall t=0,\dots,T-2
\]

### Lost opportunity cost

\[
LOC_i(\pi,g_i)= Q_i(\pi) - \Pi_i^{real}(\pi)
\]

LOC should be (\ge 0) up to solver tolerance.

---

## Time resolution (CRITICAL)

Markets clear at **5-minute** intervals. Use:

- `dt_hours = 5/60`.

Prices are in $/MWh and dispatch is MW, so multiply all profit terms by `dt_hours` to get dollars.

---

## Which prices and dispatch to use

Compute LOC for these mechanisms:

1. **RP (single-interval / ramp-product)**

- realized dispatch: `g_RP[i,t]`
- price series: `pi_RP[t] = lambda_RP[t]`
- profit uses `pi_RP[t]` and `g_RP[i,t]`

2. **LAED-LMP**

- realized dispatch: `g_LA[i,t]` (implemented/settled interval each roll)
- price series: `pi_LA[t] = lambda_LA[t]` (implemented/settled interval each roll)

3. **TLMP**

- realized dispatch: `g_LA[i,t]`
- generator-specific price series: `pi_TLMP[i,t]`

  - TLMP is discriminatory; LOC must use each unit’s own TLMP series, not a representative TLMP.

### TLMP boundary handling

If TLMP formula uses (t+1), define `pi_TLMP[i,T-1]` with the (t+1) adjustment set to 0 (or compute it using your existing boundary convention). Ensure the TLMP series length is exactly `T`.

---

## Implement (Q_i(\pi)) as a Pyomo LP (per generator)

Add a helper function:

`compute_Q_i(pi_t: array[T], gen_params: dict, cost_params: dict) -> float`

### Pyomo model for one generator (i)

- Sets:
  - `T = range(T_len)` for `t=0..T-1`
- Variables:
  - `p[t] >= 0`
- Constraints:
  - capacity: `p[t] <= Pmax`
  - ramp up:   `p[t+1] - p[t] <= Rup` for `t=0..T-2`
  - ramp down: `p[t] - p[t+1] <= Rdn` for `t=0..T-2`
- Objective (maximize profit):
  - `sum( (pi[t]*p[t] - cost(p[t])) * dt_hours )`

### Cost function

Match your dispatch cost model:

- If linear: `cost(p) = c * p`
- If piecewise/quadratic exists, implement the same expression used in the dispatch objective.

This must remain an LP if you want speed; if you add quadratic cost it becomes QP (Gurobi can still solve). Use whatever matches your existing model.

### Solver

Use Gurobi:

- `SolverFactory("gurobi")`
- Set silent output unless debugging.

Return:

- `Q_i` = `value(model.obj)`

---

## Compute realized profit (\Pi_i^{real}(\pi))

For each generator (i) and mechanism:

- `profit_real = sum( (pi[t]*g[i,t] - cost(g[i,t])) * dt_hours for t )`

Then:

- `LOC_i = Q_i - profit_real`

---

## Outputs

Per scenario, write:

### `loc.csv`

Columns:

- `generator`
- `Q_rp`, `profit_rp`, `loc_rp`
- `Q_la`, `profit_la`, `loc_la`
- `Q_tlmp`, `profit_tlmp`, `loc_tlmp`

Also write scenario totals into `metrics.json` and global `summary_metrics.csv`:

- `loc_total_rp = sum_i loc_rp`
- `loc_total_la = sum_i loc_la`
- `loc_total_tlmp = sum_i loc_tlmp`
- normalized versions per MWh served:
  - `loc_per_mwh_* = loc_total_* / (sum_t served_load[t]*dt_hours)`

---

## Sanity checks (must implement + log)

1. **Nonnegativity:** if any `loc_* < -1e-6`, log error with generator id and dump:
   - min/max of `pi`, `g`, `Pmax`, `Rup/Rdn`
2. **TLMP check:** expect `loc_total_tlmp` near 0 in convex cases. If `loc_total_tlmp > 1e-3`, log warning and dump:
   - max absolute TLMP adjustment term
   - count binding ramp constraints
   - any load-shedding activity
3. **Alignment:** ensure `len(pi)==T` and `g.shape==(N_g,T)` for each mechanism.

---

## Performance notes

- Solving one LP per generator per mechanism per scenario is OK for N_g=10, T=288.
- Reuse a single Pyomo model template if possible, updating only objective coefficients `pi[t]` each time (optional optimization).
- Cache computed `Q_i` if the same `(pi, params)` repeats (unlikely).

---

## Deliverable

Implement LOC computation and export `loc.csv` + summary fields, using 5-minute settlement (`dt_hours=5/60`), Pyomo models, and Gurobi solver.
