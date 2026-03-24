from __future__ import annotations

"""
Engineered-feature regression helper for the question:

    Which generator characteristics (and engineered dispatch/price features)
    explain lost opportunity cost (LOC) under RP vs LAED?

It loads scenario outputs from `experiments.run_sweep` (typically under
`runs/<timestamp>/...`), joins generator parameters from the case file, and
computes engineered features like:

- Ramp tightness:      |Δp| / R
- Margin opportunity:  max(π - MC, 0)
- Headroom fraction:   (Pmax - p) / Pmax
- Binding frequency:   share of intervals where |Δp| ≈ R
- Deviation from best-response (“withheld”): Δq = p* - p_dispatch

It then writes a regression-ready dataset and a compact report (simple
comparisons for:

- LOC vs ramp tightness
- LOC vs margin opportunity
- LOC vs bind frequency
- ΔLOC (LA - RP) vs the same (using LA-RP feature deltas)

Run:
  ./newvenv/bin/python linreg_for_gen_param_to_loc.py --run-dir runs/2026-03-19_094423

Optional:
  ./newvenv/bin/python linreg_for_gen_param_to_loc.py --system 10gen --ramp-regime alphalow
"""

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_run_timestamp(name: str) -> _dt.datetime | None:
    try:
        return _dt.datetime.strptime(name, "%Y-%m-%d_%H%M%S")
    except ValueError:
        return None


def find_most_recent_run(run_root: Path) -> Path:
    candidates: list[tuple[_dt.datetime, Path]] = []
    # Current layout: runs/<timestamp>/config.yaml
    # Legacy layout:  run/<group>/<timestamp>/config.yaml
    cfg_paths = list(run_root.glob("*/config.yaml")) + list(run_root.glob("*/*/config.yaml"))
    for cfg_path in cfg_paths:
        run_dir = cfg_path.parent
        ts = _parse_run_timestamp(run_dir.name) or _dt.datetime.fromtimestamp(run_dir.stat().st_mtime)
        candidates.append((ts, run_dir))
    if not candidates:
        raise FileNotFoundError(
            f"No runs found under {run_root} (expected */config.yaml or */*/config.yaml)."
        )
    return max(candidates, key=lambda x: x[0])[1]


def load_run_config(path: Path) -> dict[str, Any]:
    from experiments.utils import load_config

    return load_config(path)


def load_gen_params(case_path: Path) -> pd.DataFrame:
    from pyomo.environ import DataPortal  # type: ignore

    dp = DataPortal()
    dp.load(filename=str(case_path))
    data = dp.data()
    n_g = int(data["N_g"][None])
    rows: list[dict[str, Any]] = []
    for g in range(1, n_g + 1):
        rows.append(
            {
                "generator": int(g),
                "cost": float(data["Cost"][g]),
                "capacity": float(data["Capacity"][g]),
                "ramp_limit_base": float(data["Ramp_lim"][g]),
            }
        )
    return pd.DataFrame(rows)

def _load_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_gen_init(
    *,
    case_path: Path,
    load_ini: float,
    reserve_factor: float,
    solver: Any,
) -> np.ndarray:
    """Reproduce `experiments.utils.initialize_gen_init` for a given scenario's initial load."""
    from pyomo.environ import DataPortal  # type: ignore

    from experiments.utils import initialize_gen_init

    dp = DataPortal()
    dp.load(filename=str(case_path))
    # Only Load[1] is used by `initialize_gen_init`.
    dp.data()["Load"][1] = float(load_ini)
    initialize_gen_init(dp, solver, reserve_factor=float(reserve_factor))
    data = dp.data()
    n_g = int(data["N_g"][None])
    return np.array([float(data["Gen_init"][g]) for g in range(1, n_g + 1)], dtype=float)


def _self_schedule_optimal_dispatch(
    *,
    pi_t: np.ndarray,
    cost_coef: np.ndarray,
    pmax: np.ndarray,
    ramp: np.ndarray,
    gen_init: np.ndarray,
    solver: Any,
) -> np.ndarray:
    """
    Solve self-scheduling for all generators at once (separable across g but solved as one LP):

        max_p  sum_{g,t} (pi[t] - c[g]) * p[g,t]

    s.t. 0 <= p[g,t] <= pmax[g]
         -ramp[g] <= p[g,t] - p[g,t-1] <= ramp[g]
         -ramp[g] <= p[g,0] - gen_init[g] <= ramp[g]
    """
    pi = np.asarray(pi_t, dtype=float).reshape(-1)
    cost = np.asarray(cost_coef, dtype=float).reshape(-1)
    cap = np.asarray(pmax, dtype=float).reshape(-1)
    r = np.asarray(ramp, dtype=float).reshape(-1)
    g0 = np.asarray(gen_init, dtype=float).reshape(-1)

    if pi.ndim != 1:
        raise ValueError("pi_t must be 1D.")
    if not np.isfinite(pi).all():
        raise ValueError("pi_t contains non-finite values.")
    if cap.size != cost.size or cap.size != r.size or cap.size != g0.size:
        raise ValueError("Generator array length mismatch.")

    n_g = int(cap.size)
    n_steps = int(pi.size)

    from pyomo.environ import (  # type: ignore
        ConcreteModel,
        Constraint,
        NonNegativeReals,
        Objective,
        RangeSet,
        Var,
        maximize,
        value,
    )

    m = ConcreteModel()
    m.G = RangeSet(0, n_g - 1)
    m.T = RangeSet(0, n_steps - 1)
    m.p = Var(m.G, m.T, within=NonNegativeReals)

    def _cap_rule(m, g, t):
        return m.p[g, t] <= float(cap[g])

    m.cap = Constraint(m.G, m.T, rule=_cap_rule)

    if n_steps >= 2:
        m.T_ramp = RangeSet(1, n_steps - 1)

        def _rup_rule(m, g, t):
            return m.p[g, t] - m.p[g, t - 1] <= float(r[g])

        def _rdn_rule(m, g, t):
            return m.p[g, t - 1] - m.p[g, t] <= float(r[g])

        m.rup = Constraint(m.G, m.T_ramp, rule=_rup_rule)
        m.rdn = Constraint(m.G, m.T_ramp, rule=_rdn_rule)

    def _rup0_rule(m, g):
        return m.p[g, 0] - float(g0[g]) <= float(r[g])

    def _rdn0_rule(m, g):
        return float(g0[g]) - m.p[g, 0] <= float(r[g])

    m.rup0 = Constraint(m.G, rule=_rup0_rule)
    m.rdn0 = Constraint(m.G, rule=_rdn0_rule)

    def _obj_rule(m):
        return sum((float(pi[t]) - float(cost[g])) * m.p[g, t] for g in m.G for t in m.T)

    m.obj = Objective(rule=_obj_rule, sense=maximize)

    res = solver.solve(m, tee=False)
    term = str(res.solver.termination_condition).lower()
    status = str(res.solver.status).lower()
    if term != "optimal":
        raise RuntimeError(f"Self-schedule solve failed: status={status}, term={term}")

    p_opt = np.array([[float(value(m.p[g, t])) for t in m.T] for g in m.G], dtype=float)
    return p_opt


def _engineer_features(
    *,
    dispatch_by_gen: np.ndarray,
    pi_t: np.ndarray,
    cost_coef: np.ndarray,
    capacity: np.ndarray,
    ramp: np.ndarray,
    gen_init: np.ndarray,
    dt_hours: float,
    solver: Any | None,
    include_best_response: bool,
) -> pd.DataFrame:
    """Return one row per generator with engineered features."""
    p = np.asarray(dispatch_by_gen, dtype=float)
    pi = np.asarray(pi_t, dtype=float).reshape(-1)
    cost = np.asarray(cost_coef, dtype=float).reshape(-1)
    cap = np.asarray(capacity, dtype=float).reshape(-1)
    r = np.asarray(ramp, dtype=float).reshape(-1)
    g0 = np.asarray(gen_init, dtype=float).reshape(-1)

    if p.ndim != 2:
        raise ValueError(f"dispatch_by_gen must be 2D; got {p.shape}")
    n_g, n_steps = p.shape
    if pi.size != n_steps:
        raise ValueError(f"Price length mismatch: pi has {pi.size}, dispatch has T={n_steps}")
    if cost.size != n_g or cap.size != n_g or r.size != n_g or g0.size != n_g:
        raise ValueError("Generator dimension mismatch in engineered features.")

    dt = float(dt_hours)
    out: dict[str, np.ndarray] = {}

    out["mwh_dispatch"] = np.sum(p, axis=1) * dt

    # Δp including the initial transition from Gen_init -> first committed dispatch.
    dp = np.empty_like(p)
    dp[:, 0] = p[:, 0] - g0
    if n_steps >= 2:
        dp[:, 1:] = p[:, 1:] - p[:, :-1]

    # Ramp tightness = |Δp| / R.
    r_safe = np.where(r > 0.0, r, np.nan).reshape(-1, 1)
    tight = np.abs(dp) / r_safe
    out["ramp_tight_mean"] = np.nanmean(tight, axis=1)
    out["ramp_tight_max"] = np.nanmax(tight, axis=1)

    # Binding frequency (proxy): |Δp| within tolerance of R.
    bind_tol = 1e-6
    bind = (r.reshape(-1, 1) > 0.0) & (np.abs(dp) >= (r.reshape(-1, 1) - bind_tol))
    out["bind_freq"] = np.mean(bind, axis=1)

    # Margin opportunity (positive part) and frequency.
    margin_pos = np.maximum(pi.reshape(1, -1) - cost.reshape(-1, 1), 0.0)
    out["margin_pos_mean"] = np.mean(margin_pos, axis=1)
    out["margin_pos_frac"] = np.mean(margin_pos > 0.0, axis=1)

    # Headroom fraction.
    cap_safe = np.where(cap > 0.0, cap, np.nan).reshape(-1, 1)
    headroom = (cap.reshape(-1, 1) - p) / cap_safe
    out["headroom_mean"] = np.nanmean(headroom, axis=1)
    out["headroom_min"] = np.nanmin(headroom, axis=1)

    # Best-response deviation: Δq = p* - p_dispatch.
    if include_best_response:
        if solver is None:
            raise ValueError("include_best_response requires a solver.")
        p_opt = _self_schedule_optimal_dispatch(
            pi_t=pi,
            cost_coef=cost,
            pmax=cap,
            ramp=r,
            gen_init=g0,
            solver=solver,
        )
        dq = p_opt - p
        out["dq_pos_mwh"] = np.sum(np.maximum(dq, 0.0), axis=1) * dt
        out["dq_neg_mwh"] = np.sum(np.maximum(-dq, 0.0), axis=1) * dt
        out["dq_abs_mwh"] = np.sum(np.abs(dq), axis=1) * dt
        out["loc_from_dq"] = np.sum((pi.reshape(1, -1) - cost.reshape(-1, 1)) * dq, axis=1) * dt
    else:
        out["dq_pos_mwh"] = np.full(n_g, np.nan, dtype=float)
        out["dq_neg_mwh"] = np.full(n_g, np.nan, dtype=float)
        out["dq_abs_mwh"] = np.full(n_g, np.nan, dtype=float)
        out["loc_from_dq"] = np.full(n_g, np.nan, dtype=float)

    return pd.DataFrame({"generator": np.arange(1, n_g + 1, dtype=int), **out})


def main() -> int:
    ap = argparse.ArgumentParser(description="Engineered-feature LOC analysis (RP vs LAED).")
    ap.add_argument("--run-root", type=Path, default=Path("runs"), help="Root runs/ folder (default: runs).")
    ap.add_argument("--run-dir", type=Path, default=None, help="Explicit run dir (overrides auto-detect).")
    ap.add_argument("--system", type=str, default="10gen", help="System key from config (default: 10gen).")
    ap.add_argument("--ramp-regime", type=str, default="alphalow", help="Ramp regime name (default: alphalow).")
    ap.add_argument("--horizon", type=int, default=None, help="Optional horizon filter (H).")
    ap.add_argument(
        "--plots",
        action="store_true",
        help="Also save scatter plots with fit lines under <run_dir>/plots/.",
    )
    ap.add_argument(
        "--plot-ytransform",
        type=str,
        default="log1p",
        choices=("none", "log1p", "signed_log1p"),
        help="Transform y for plots (default: log1p). For ΔLOC, log1p auto-falls back to signed log1p.",
    )
    ap.add_argument(
        "--flex-low",
        type=float,
        default=0.1,
        help="Flex bucket cutoff for low flexibility (R/Pmax < flex_low). Default: 0.1.",
    )
    ap.add_argument(
        "--flex-high",
        type=float,
        default=0.2,
        help="Flex bucket cutoff for high flexibility (R/Pmax >= flex_high). Default: 0.2.",
    )
    ap.add_argument(
        "--margin-thresh",
        type=float,
        default=None,
        help="Margin threshold for 'high margin'. Default: median across generators.",
    )
    ap.add_argument(
        "--skip-best-response",
        action="store_true",
        help="Skip self-scheduling solves (won't compute dq_* or loc_from_dq features).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    run_dir = args.run_dir.resolve() if args.run_dir is not None else find_most_recent_run(args.run_root.resolve())

    cfg = load_run_config(run_dir / "config.yaml")
    systems_cfg: dict[str, Any] = cfg.get("systems") or {}
    system_key = str(args.system)
    ramp_regime = str(args.ramp_regime)
    if system_key not in systems_cfg:
        raise SystemExit(f"Run config does not define systems['{system_key}'].")
    case_path = Path(systems_cfg[system_key]["case_path"])
    if not case_path.is_absolute():
        case_path = (repo_root / case_path).resolve()

    gen_df = load_gen_params(case_path)

    model_cfg = cfg.get("model") or {}
    reserve_factor = float(model_cfg.get("reserve_factor", 0.0))

    # Build solver (same config as the run).
    from experiments.utils import create_pyomo_solver

    solver = create_pyomo_solver(cfg.get("solver") or {})

    # Collect scenario folders by reading meta.json so we don't depend on naming conventions.
    scenario_dirs: list[Path] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "loc.csv").exists() or not (child / "meta.json").exists() or not (child / "timeseries.csv").exists():
            continue
        meta = _load_meta(child / "meta.json")
        if str(meta.get("system")) != system_key:
            continue
        if str(meta.get("ramp_regime")) != ramp_regime:
            continue
        if args.horizon is not None and int(meta.get("horizon", -1)) != int(args.horizon):
            continue
        scenario_dirs.append(child)

    if not scenario_dirs:
        raise SystemExit(
            f"No scenarios found in {run_dir} for system={system_key}, ramp_regime={ramp_regime} "
            f"{'(H=' + str(args.horizon) + ')' if args.horizon is not None else ''}."
        )

    rows: list[pd.DataFrame] = []
    n_g = int(gen_df["generator"].max())
    gens = list(range(1, n_g + 1))
    cost_coef = gen_df.sort_values("generator")["cost"].to_numpy(dtype=float)
    cap = gen_df.sort_values("generator")["capacity"].to_numpy(dtype=float)
    ramp_base = gen_df.sort_values("generator")["ramp_limit_base"].to_numpy(dtype=float)

    for sdir in scenario_dirs:
        meta = _load_meta(sdir / "meta.json")
        ts = pd.read_csv(sdir / "timeseries.csv")
        loc = pd.read_csv(sdir / "loc.csv")
        if not {"generator", "loc_rp", "loc_la"}.issubset(set(loc.columns)):
            continue

        dt_hours = float(meta.get("dt_hours", 5.0 / 60.0))
        ramp_multiplier = float(meta.get("ramp_multiplier", 1.0))
        ramp_eff = ramp_base * ramp_multiplier
        flex = np.divide(ramp_eff, cap, out=np.full_like(ramp_eff, np.nan), where=cap > 0.0)

        # Compute Gen_init consistent with the run (depends on load at t=1).
        load_ini = float(ts["demand"].iloc[0]) if "demand" in ts.columns else float("nan")
        gen_init = _compute_gen_init(
            case_path=case_path,
            load_ini=load_ini,
            reserve_factor=reserve_factor,
            solver=solver,
        )

        # Pull dispatch + price series.
        p_ed = np.stack([ts[f"p_ed_g{g}"].to_numpy(dtype=float) for g in gens], axis=0)
        p_laed = np.stack([ts[f"p_laed_g{g}"].to_numpy(dtype=float) for g in gens], axis=0)
        pi_rp = ts["pi_lmp_energy"].to_numpy(dtype=float)
        pi_la = ts["pi_la_energy"].to_numpy(dtype=float)

        include_br = not bool(args.skip_best_response)
        rp_feat = _engineer_features(
            dispatch_by_gen=p_ed,
            pi_t=pi_rp,
            cost_coef=cost_coef,
            capacity=cap,
            ramp=ramp_eff,
            gen_init=gen_init,
            dt_hours=dt_hours,
            solver=solver,
            include_best_response=include_br,
        ).add_prefix("rp_")
        rp_feat = rp_feat.rename(columns={"rp_generator": "generator"})

        la_feat = _engineer_features(
            dispatch_by_gen=p_laed,
            pi_t=pi_la,
            cost_coef=cost_coef,
            capacity=cap,
            ramp=ramp_eff,
            gen_init=gen_init,
            dt_hours=dt_hours,
            solver=solver,
            include_best_response=include_br,
        ).add_prefix("la_")
        la_feat = la_feat.rename(columns={"la_generator": "generator"})

        df = loc.merge(gen_df, on="generator", how="left", validate="many_to_one")
        df["scenario_id"] = str(meta.get("scenario_id", sdir.name))
        df["system"] = system_key
        df["ramp_regime"] = ramp_regime
        df["horizon"] = int(meta.get("horizon", -1))
        df["dt_hours"] = dt_hours
        df["ramp_multiplier"] = ramp_multiplier
        df["ramp_limit_eff"] = ramp_eff[df["generator"].to_numpy(dtype=int) - 1]
        df["flex_ratio"] = flex[df["generator"].to_numpy(dtype=int) - 1]

        df = df.merge(rp_feat, on="generator", how="left")
        df = df.merge(la_feat, on="generator", how="left")

        # Derived targets / normalizations.
        df["delta_loc"] = df["loc_rp"].astype(float) - df["loc_la"].astype(float)
        df["loc_rp_per_mwh"] = df["loc_rp"].astype(float) / df["rp_mwh_dispatch"].replace(0.0, np.nan)
        df["loc_la_per_mwh"] = df["loc_la"].astype(float) / df["la_mwh_dispatch"].replace(0.0, np.nan)

        if include_br:
            df["rp_loc_from_dq_err"] = df["loc_rp"].astype(float) - df["rp_loc_from_dq"].astype(float)
            df["la_loc_from_dq_err"] = df["loc_la"].astype(float) - df["la_loc_from_dq"].astype(float)

        rows.append(df)

    if not rows:
        raise SystemExit("No usable scenario loc.csv files found after filtering.")

    full = pd.concat(rows, ignore_index=True)

    full["loc_la_minus_rp"] = full["loc_la"].astype(float) - full["loc_rp"].astype(float)
    full["loc_rp_minus_la"] = full["loc_rp"].astype(float) - full["loc_la"].astype(float)

    # Differences in engineered features (LA - RP), useful for delta models.
    for base in (
        "mwh_dispatch",
        "ramp_tight_mean",
        "bind_freq",
        "margin_pos_mean",
        "headroom_mean",
        "dq_pos_mwh",
        "dq_neg_mwh",
        "dq_abs_mwh",
    ):
        la_col = f"la_{base}"
        rp_col = f"rp_{base}"
        if la_col in full.columns and rp_col in full.columns:
            full[f"d_{base}"] = full[la_col].astype(float) - full[rp_col].astype(float)

    out_dataset = run_dir / f"loc_features_{system_key}_{ramp_regime}.csv"
    full.to_csv(out_dataset, index=False)

    # ------------------------------------------------------------------
    # Per-generator table + generator-type buckets
    # ------------------------------------------------------------------
    gen_agg_cols: dict[str, str] = {
        "cost": "mean",
        "capacity": "mean",
        "ramp_limit_eff": "mean",
        "flex_ratio": "mean",
        "loc_rp": "mean",
        "loc_la": "mean",
        "loc_la_minus_rp": "mean",
        "rp_margin_pos_mean": "mean",
        "la_margin_pos_mean": "mean",
        "rp_ramp_tight_mean": "mean",
        "la_ramp_tight_mean": "mean",
        "rp_bind_freq": "mean",
        "la_bind_freq": "mean",
    }
    present_agg_cols = {k: v for k, v in gen_agg_cols.items() if k in full.columns}
    gen_summary = full.groupby("generator", as_index=False).agg(present_agg_cols)
    gen_summary = gen_summary.merge(
        full.groupby("generator", as_index=False)["scenario_id"]
        .nunique()
        .rename(columns={"scenario_id": "n_scenarios"}),
        on="generator",
        how="left",
    )

    # Collapsed metrics used in the requested generator table.
    rp_m = gen_summary.get("rp_margin_pos_mean", pd.Series(np.nan, index=gen_summary.index)).astype(float)
    la_m = gen_summary.get("la_margin_pos_mean", pd.Series(np.nan, index=gen_summary.index)).astype(float)
    gen_summary["margin"] = np.nanmean(np.column_stack([rp_m.to_numpy(), la_m.to_numpy()]), axis=1)

    rp_rt = gen_summary.get("rp_ramp_tight_mean", pd.Series(np.nan, index=gen_summary.index)).astype(float)
    la_rt = gen_summary.get("la_ramp_tight_mean", pd.Series(np.nan, index=gen_summary.index)).astype(float)
    gen_summary["ramp_tight_mean"] = np.nanmean(np.column_stack([rp_rt.to_numpy(), la_rt.to_numpy()]), axis=1)

    rp_b = gen_summary.get("rp_bind_freq", pd.Series(np.nan, index=gen_summary.index)).astype(float)
    la_b = gen_summary.get("la_bind_freq", pd.Series(np.nan, index=gen_summary.index)).astype(float)
    gen_summary["bind_freq"] = np.nanmean(np.column_stack([rp_b.to_numpy(), la_b.to_numpy()]), axis=1)

    # Requested columns + friendly names.
    out_gen_table = run_dir / f"loc_generator_table_{system_key}_{ramp_regime}.csv"
    gen_table_cols = [
        "generator",
        "cost",
        "capacity",
        "ramp_limit_eff",
        "flex_ratio",
        "margin",
        "ramp_tight_mean",
        "bind_freq",
        "loc_rp",
        "loc_la",
        "loc_la_minus_rp",
        "n_scenarios",
    ]
    gen_table = gen_summary[[c for c in gen_table_cols if c in gen_summary.columns]].copy()
    gen_table = gen_table.rename(
        columns={
            "ramp_limit_eff": "ramp",
            "flex_ratio": "ramp_over_capacity",
            "loc_rp": "LOC_RP",
            "loc_la": "LOC_LAED",
            "loc_la_minus_rp": "Delta_LOC",
        }
    )

    # Buckets:
    # - Flexibility buckets on R/Pmax: low < 0.1, medium < 0.2, high >= 0.2
    # - Margin buckets: median split of `margin` (strictly above median = high).
    flex_low = float(args.flex_low)
    flex_high = float(args.flex_high)
    flex = gen_table["ramp_over_capacity"].astype(float)
    gen_table["flex_bucket"] = np.where(
        flex < flex_low,
        "low",
        np.where(flex < flex_high, "medium", "high"),
    )
    margin = gen_table["margin"].astype(float)
    if args.margin_thresh is None:
        margin_thresh = float(np.nanmedian(margin.to_numpy(dtype=float)))
    else:
        margin_thresh = float(args.margin_thresh)
    gen_table["margin_bucket"] = np.where(margin > margin_thresh, "high", "low")

    gen_table.to_csv(out_gen_table, index=False)

    # Bucket summaries: average LOC under RP vs LAED.
    def _safe_mean(s: pd.Series) -> float:
        sv = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        return float(np.nanmean(sv))

    bucket_summary = (
        gen_table.groupby(["flex_bucket", "margin_bucket"], observed=True)
        .agg(
            n_gens=("generator", "count"),
            avg_LOC_RP=("LOC_RP", _safe_mean),
            avg_LOC_LAED=("LOC_LAED", _safe_mean),
            avg_Delta_LOC=("Delta_LOC", _safe_mean),  # LA - RP
        )
        .reset_index()
    )
    out_bucket_table = run_dir / f"loc_bucket_summary_{system_key}_{ramp_regime}.csv"
    bucket_summary.to_csv(out_bucket_table, index=False)

    # ------------------------------------------------------------------
    # Report: bivariate comparisons (requested)
    # ------------------------------------------------------------------
    def _bivar_fit(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
        """Return (intercept, slope, r2) for y ~ a + b x."""
        yv = np.asarray(y, dtype=float).reshape(-1)
        xv = np.asarray(x, dtype=float).reshape(-1)
        X1 = np.column_stack([np.ones_like(xv), xv])
        beta, *_ = np.linalg.lstsq(X1, yv, rcond=None)
        a = float(beta[0])
        b = float(beta[1])
        y_hat = X1 @ beta
        sse = float(np.sum((yv - y_hat) ** 2))
        sst = float(np.sum((yv - float(np.mean(yv))) ** 2))
        r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
        return a, b, r2

    def _bivar_block(y_col: str, x_col: str, title: str) -> str:
        cols = ["generator", "scenario_id", y_col, x_col]
        df = full[[c for c in cols if c in full.columns]].copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if df.empty:
            return f"{title}\nNo rows after dropping NaNs."

        y = df[y_col].to_numpy(dtype=float)
        x = df[x_col].to_numpy(dtype=float)
        n = int(df.shape[0])

        if float(np.nanstd(x)) < 1e-12:
            return f"{title}\nRows used: {n}\nPredictor is (near) constant; skipping fit."
        if float(np.nanstd(y)) < 1e-12:
            return f"{title}\nRows used: {n}\nTarget is (near) constant; skipping fit."

        a, b, r2 = _bivar_fit(y=y, x=x)

        lines = [
            title,
            f"Rows used: {n}",
            f"Fit: {y_col} = {a:.6g} + {b:.6g} * {x_col}",
            f"R^2 (in-sample): {r2:.6g}",
        ]

        # Quick quartile means (helps see nonlinearity / outliers).
        if n >= 8 and int(pd.Series(x).nunique()) >= 4:
            try:
                bins = pd.qcut(x, q=4, duplicates="drop")
                q = (
                    pd.DataFrame({x_col: x, y_col: y})
                    .assign(_bin=bins)
                    .groupby("_bin", observed=True)[y_col]
                    .agg(["count", "mean"])
                    .reset_index()
                )
                lines.append("")
                lines.append("Quartile means (by predictor):")
                lines.append(q.to_string(index=False, float_format=lambda v: f"{v: .6g}"))
            except ValueError:
                pass

        return "\n".join(lines) + "\n"

    def _maybe_write_plots() -> list[Path]:
        if not args.plots:
            return []

        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Avoid matplotlib trying to write to ~/.matplotlib (not writable in some environments).
        mpl_cfg = run_dir / ".mplconfig"
        mpl_cfg.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _transform_y_vals(y: np.ndarray, *, allow_negative: bool) -> np.ndarray:
            yv = np.asarray(y, dtype=float).reshape(-1)
            if args.plot_ytransform == "none":
                return yv
            if args.plot_ytransform == "signed_log1p":
                return np.sign(yv) * np.log1p(np.abs(yv))

            # log1p (requested): use signed transform for ΔLOC; for LOC, clip to >= 0.
            if allow_negative:
                return np.sign(yv) * np.log1p(np.abs(yv))
            return np.log1p(np.maximum(yv, 0.0))

        def _ylabel(*, base: str, allow_negative: bool) -> str:
            if args.plot_ytransform == "none":
                return base
            if args.plot_ytransform == "signed_log1p" or allow_negative:
                return f"sign({base})·log1p(|{base}|)"
            return f"log1p({base})"

        def _scatter_series_with_fit(
            ax,
            *,
            x_col: str,
            y_col: str,
            label: str,
            color: str,
            allow_negative_y: bool,
        ) -> None:
            df = full[["generator", "scenario_id", x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if df.empty:
                return

            x = df[x_col].to_numpy(dtype=float)
            y_raw = df[y_col].to_numpy(dtype=float)
            y = _transform_y_vals(y_raw, allow_negative=allow_negative_y)
            ax.scatter(x, y, alpha=0.8, label=label, color=color)

            if float(np.nanstd(x)) >= 1e-12 and float(np.nanstd(y)) >= 1e-12:
                a, b, r2 = _bivar_fit(y=y, x=x)
                xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
                ax.plot(xs, a + b * xs, linewidth=2, color=color, alpha=0.9, label=f"{label} fit R^2={r2:.3g}")

        written: list[Path] = []
        y_suffix = (
            "ylinear"
            if args.plot_ytransform == "none"
            else ("ylog1p" if args.plot_ytransform == "log1p" else "ysignedlog1p")
        )

        # Combined LA vs RP plots (same axes, different x definitions per mechanism).
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        combos = [
            ("ramp_tight_mean", "Ramp tightness (mean |Δp| / R)"),
            ("margin_pos_mean", "Margin opportunity (mean (π - MC)+)"),
            ("bind_freq", "Bind frequency (share |Δp|≈R)"),
        ]
        for ax, (base, pretty) in zip(axes, combos, strict=True):
            x_rp = f"rp_{base}"
            x_la = f"la_{base}"
            if x_rp in full.columns:
                _scatter_series_with_fit(
                    ax,
                    x_col=x_rp,
                    y_col="loc_rp",
                    label="RP",
                    color="tab:blue",
                    allow_negative_y=False,
                )
            if x_la in full.columns:
                _scatter_series_with_fit(
                    ax,
                    x_col=x_la,
                    y_col="loc_la",
                    label="LAED",
                    color="tab:orange",
                    allow_negative_y=False,
                )
            ax.set_title(f"LOC vs {pretty}")
            ax.set_xlabel(pretty)
            ax.set_ylabel(_ylabel(base="LOC", allow_negative=False))
            ax.axhline(0.0, color="black", linewidth=1, alpha=0.25)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        out = plots_dir / f"{system_key}_{ramp_regime}_laed_vs_rp_loc_bivar_{y_suffix}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        written.append(out)

        # Delta plots remain separate (single series).
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (base, pretty) in zip(axes, combos, strict=True):
            x_col = f"d_{base}"
            if x_col not in full.columns:
                ax.set_axis_off()
                continue
            _scatter_series_with_fit(
                ax,
                x_col=x_col,
                y_col="loc_la_minus_rp",
                label="ΔLOC (LA-RP)",
                color="tab:green",
                allow_negative_y=True,
            )
            ax.set_title(f"ΔLOC vs Δ {pretty}")
            ax.set_xlabel(f"Δ {pretty}")
            ax.set_ylabel(_ylabel(base="ΔLOC", allow_negative=True))
            ax.axhline(0.0, color="black", linewidth=1, alpha=0.25)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        out = plots_dir / f"{system_key}_{ramp_regime}_delta_loc_bivar_{y_suffix}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        written.append(out)

        # Bucket comparison: average LOC under RP vs LAED.
        if not bucket_summary.empty:
            flex_order = {"low": 0, "medium": 1, "high": 2}
            margin_order = {"low": 0, "high": 1}
            bs = bucket_summary.copy()
            bs["_o"] = bs["flex_bucket"].map(flex_order).fillna(99) * 10 + bs["margin_bucket"].map(margin_order).fillna(9)
            bs = bs.sort_values("_o").drop(columns=["_o"])

            labels = [
                f"{fb}-flex / {mb}-margin" for fb, mb in zip(bs["flex_bucket"], bs["margin_bucket"], strict=True)
            ]
            loc_rp = bs["avg_LOC_RP"].to_numpy(dtype=float)
            loc_la = bs["avg_LOC_LAED"].to_numpy(dtype=float)
            loc_rp_t = _transform_y_vals(loc_rp, allow_negative=False)
            loc_la_t = _transform_y_vals(loc_la, allow_negative=False)

            x = np.arange(int(bs.shape[0]), dtype=float)
            w = 0.38
            fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
            ax.bar(x - w / 2, loc_rp_t, width=w, label="RP", color="tab:blue", alpha=0.9)
            ax.bar(x + w / 2, loc_la_t, width=w, label="LAED", color="tab:orange", alpha=0.9)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.set_ylabel(_ylabel(base="Avg LOC", allow_negative=False))
            ax.set_title("Average LOC by Generator Type")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(loc="best", fontsize="small")

            # Annotate with raw (untransformed) values for interpretability.
            for xi, v_rp, v_la in zip(x, loc_rp, loc_la, strict=True):
                y_rp = _transform_y_vals(np.array([v_rp]), allow_negative=False)[0]
                y_la = _transform_y_vals(np.array([v_la]), allow_negative=False)[0]
                ax.annotate(
                    f"{v_rp:.0f}",
                    xy=(xi - w / 2, y_rp),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
                ax.annotate(
                    f"{v_la:.0f}",
                    xy=(xi + w / 2, y_la),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            fig.tight_layout()
            out = plots_dir / f"{system_key}_{ramp_regime}_bucket_loc_{y_suffix}.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)
            written.append(out)

        return written

    blocks: list[str] = []
    blocks.append(f"Run: {run_dir}")
    blocks.append(f"System: {system_key}, ramp_regime: {ramp_regime}, scenarios: {len(scenario_dirs)}")
    blocks.append(f"Saved dataset: {out_dataset}")
    blocks.append(f"Saved generator table: {out_gen_table}")
    blocks.append(f"Saved bucket table: {out_bucket_table}")
    blocks.append(
        f"Buckets: flex_low={flex_low:.6g}, flex_high={flex_high:.6g}, "
        f"margin_thresh={'median' if args.margin_thresh is None else str(float(args.margin_thresh))}"
    )
    if not args.skip_best_response:
        blocks.append(
            "Best-response consistency check (loc - loc_from_dq):\n"
            f"  RP  max|err|={float(np.nanmax(np.abs(full.get('rp_loc_from_dq_err', np.nan)))):.6g}\n"
            f"  LA  max|err|={float(np.nanmax(np.abs(full.get('la_loc_from_dq_err', np.nan)))):.6g}"
        )

    blocks.append("Bivariate fits (each is y ~ a + b x)\n")

    blocks.append("RP-LMP LOC (loc_rp)\n")
    blocks.append(_bivar_block("loc_rp", "rp_ramp_tight_mean", "loc_rp vs rp_ramp_tight_mean (ramp tightness)"))
    blocks.append(_bivar_block("loc_rp", "rp_margin_pos_mean", "loc_rp vs rp_margin_pos_mean (margin opportunity)"))
    blocks.append(_bivar_block("loc_rp", "rp_bind_freq", "loc_rp vs rp_bind_freq (bind frequency)"))

    blocks.append("LAED-LMP LOC (loc_la)\n")
    blocks.append(_bivar_block("loc_la", "la_ramp_tight_mean", "loc_la vs la_ramp_tight_mean (ramp tightness)"))
    blocks.append(_bivar_block("loc_la", "la_margin_pos_mean", "loc_la vs la_margin_pos_mean (margin opportunity)"))
    blocks.append(_bivar_block("loc_la", "la_bind_freq", "loc_la vs la_bind_freq (bind frequency)"))

    blocks.append("ΔLOC = loc_la - loc_rp (loc_la_minus_rp)\n")
    blocks.append(
        _bivar_block("loc_la_minus_rp", "d_ramp_tight_mean", "ΔLOC vs d_ramp_tight_mean (Δ ramp tightness)")
    )
    blocks.append(_bivar_block("loc_la_minus_rp", "d_margin_pos_mean", "ΔLOC vs d_margin_pos_mean (Δ margin)"))
    blocks.append(_bivar_block("loc_la_minus_rp", "d_bind_freq", "ΔLOC vs d_bind_freq (Δ bind frequency)"))

    if not bucket_summary.empty:
        blocks.append("Bucket summary (mean across generators; Delta_LOC = LA - RP)\n")
        blocks.append(bucket_summary.to_string(index=False, float_format=lambda v: f"{v: .6g}") + "\n")
        worst = bucket_summary.sort_values("avg_Delta_LOC", ascending=True).iloc[0]
        blocks.append(
            "Most negative avg_Delta_LOC bucket (RP > LAED, i.e. 'suffers' more under RP):\n"
            f"  flex_bucket={worst['flex_bucket']}, margin_bucket={worst['margin_bucket']}, "
            f"avg_Delta_LOC={float(worst['avg_Delta_LOC']):.6g}"
        )

    plot_paths = _maybe_write_plots()
    if plot_paths:
        blocks.append("Saved plots:\n" + "\n".join(f"  {p}" for p in plot_paths))

    out_report = run_dir / f"loc_feature_report_{system_key}_{ramp_regime}.txt"
    out_report.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    print("\n\n".join(blocks))
    print(f"Wrote: {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
