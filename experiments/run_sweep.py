from __future__ import annotations

import argparse
import copy
import contextlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.utils import (
    create_pyomo_solver,
    derive_seed,
    ensure_dir,
    environment_meta,
    exc_to_str,
    initialize_gen_init,
    load_config,
    load_miso_projection,
    set_global_seeds,
    setup_scenario_logger,
    slugify,
    tlmp_marginal_unit_series,
    tlmp_price_paid_by_load,
    volatility_metrics,
    write_json,
    write_summary_csv,
    write_summary_markdown,
    write_text,
    write_timeseries_csv,
)


def _resolve_ramp_multiplier(ramp_regime_cfg: dict[str, Any], system_name: str) -> float:
    """
    Support either:
      - {"multiplier": 0.3}
      - {"multiplier": {"2gen": 0.15, "10gen": 0.3, "default": 0.3}}
      - {"multiplier": 0.3, "multiplier_by_system": {"2gen": 0.15}}
    """
    if not isinstance(ramp_regime_cfg, dict):
        raise TypeError(f"ramp_regime_cfg must be a dict; got {type(ramp_regime_cfg)}")

    base = ramp_regime_cfg.get("multiplier", 1.0)
    if isinstance(base, dict):
        if system_name in base:
            return float(base[system_name])
        if "default" in base:
            return float(base["default"])
        raise KeyError(
            f"ramp_regime_cfg.multiplier is a dict but has no entry for '{system_name}' "
            f"and no 'default'. Keys={sorted(map(str, base.keys()))}"
        )

    out = float(base)
    by_sys = ramp_regime_cfg.get("multiplier_by_system")
    if isinstance(by_sys, dict) and (system_name in by_sys):
        out = float(by_sys[system_name])
    return out


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run pricing-volatility experiment sweeps with reproducible logging.")
    ap.add_argument("--config", default="experiments/config.yaml", help="Config path (YAML or JSON-compatible YAML).")
    ap.add_argument("--out", required=True, help="Output root folder (a unique run subfolder is created per sweep).")
    ap.add_argument(
        "--run-name",
        default=None,
        help="Optional name for this run subfolder (defaults to a timestamp).",
    )
    ap.add_argument(
        "--no-run-subdir",
        action="store_true",
        help="Write directly to --out (no per-run subfolder).",
    )

    ap.add_argument("--systems", nargs="*", default=None, help="Subset of systems to run (e.g. 2gen 10gen).")
    ap.add_argument("--mechanisms", nargs="*", default=None, help="Subset of mechanisms (lmp la tlmp).")
    ap.add_argument(
        "--ramp_regimes",
        nargs="*",
        default=None,
        help="Subset of ramp-scaling regimes from config (e.g. alphalow alphamed alphahigh).",
    )
    ap.add_argument("--horizons", nargs="*", type=int, default=None, help="Subset of look-ahead horizons (e.g. 4).")

    ap.add_argument("--no-console", action="store_true", help="Disable console logging (file logging still enabled).")
    ap.add_argument("--no-plots", action="store_true", help="Disable per-scenario plots.")
    ap.add_argument("--seed", type=int, default=None, help="Override base seed from config.")
    return ap


def _run_single_scenario(
    *,
    out_dir: Path,
    scenario_id: str,
    cfg: dict[str, Any],
    system_name: str,
    system_cfg: dict[str, Any],
    ramp_regime_name: str,
    ramp_regime_cfg: dict[str, Any],
    mechanisms: list[str],
    horizon: int,
    base_seed: int,
    console: bool,
    plots: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run one scenario and write per-scenario artifacts.

    Returns:
      - list of summary rows (one per computed series)
      - ramp diagnostics dict (always returned; may be partial on failure)
    """
    ensure_dir(out_dir)
    scenario_dir = out_dir / scenario_id
    ensure_dir(scenario_dir)
    repo_root = Path(__file__).resolve().parents[1]

    logger = setup_scenario_logger(scenario_dir / "log.txt", scenario_id=scenario_id, console=console)
    t0 = time.perf_counter()
    ramp_diag: dict[str, Any] = {"scenario_id": scenario_id}
    meta: dict[str, Any] = {
        "scenario_id": scenario_id,
        "config": cfg,
        "system": system_name,
        "system_cfg": system_cfg,
        "ramp_regime": ramp_regime_name,
        "ramp_regime_cfg": ramp_regime_cfg,
        "mechanisms_requested": list(mechanisms),
        "horizon": int(horizon),
        "base_seed": int(base_seed),
    }

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
        os.environ.setdefault("MPLBACKEND", "Agg")

        # Derive a deterministic scenario seed (stable across ordering).
        scenario_seed = derive_seed(
            base_seed,
            parts=[
                system_name,
                ramp_regime_name,
                f"H{horizon}",
            ],
        )
        meta["seed"] = int(scenario_seed)
        set_global_seeds(scenario_seed)

        solver_cfg = cfg.get("solver") or {}
        meta["solver"] = solver_cfg
        solver = create_pyomo_solver(solver_cfg)

        # Capture any legacy `print(...)` output from existing solver code into the scenario log.
        # Keep stderr untouched so console logging still works normally.
        with (scenario_dir / "log.txt").open("a", encoding="utf-8") as _stdout_log, contextlib.redirect_stdout(_stdout_log):
            # Load the case data (Pyomo DataPortal).
            from pyomo.environ import DataPortal  # type: ignore

            data = DataPortal()
            case_path = Path(system_cfg["case_path"])
            if not case_path.is_absolute():
                case_path = (repo_root / case_path).resolve()
            data.load(filename=str(case_path))

            # Override horizon.
            data.data()["N_t"][None] = int(horizon)

            # Build base load series.
            load_source = system_cfg.get("load_source") or {"type": "case"}
            if str(load_source.get("type", "case")).lower() == "miso_projection":
                proj_path = Path(load_source["path"])
                if not proj_path.is_absolute():
                    proj_path = (repo_root / proj_path).resolve()
                proj_key = str(load_source.get("key", "2032_Aug"))
                ref_cap = float(load_source.get("ref_cap", 1050.0))
                base_load = load_miso_projection(proj_path, key=proj_key, ref_cap=ref_cap)
                logger.info(f"Loaded MISO projection '{proj_key}' from {proj_path} (T={len(base_load)})")
            else:
                base_load = {int(t): float(v) for t, v in data.data()["Load"].items()}

            # Use a single fixed load trajectory d_t for all runs.
            load = dict(base_load)
            data.data()["Load"] = load
            data.data()["N_T"][None] = int(len(load))

            # Model globals used by laed_rp_analysis.
            import laed_rp_analysis as lra

            model_cfg = cfg.get("model") or {}
            lra.reserve_factor = float(model_cfg.get("reserve_factor", 0.0))
            lra.cost_load = float(model_cfg.get("cost_load", 3500.0))

            # Initialize starting dispatch.
            initialize_gen_init(data, solver, reserve_factor=float(lra.reserve_factor))

            n_g = int(data.data()["N_g"][None])
            n_t = int(data.data()["N_t"][None])
            n_T = int(len(load))
            ramp_multiplier = _resolve_ramp_multiplier(ramp_regime_cfg, system_name)
            meta["ramp_multiplier"] = ramp_multiplier
            meta["N_g"] = n_g
            meta["N_T"] = n_T

            data_ed = copy.deepcopy(data)
            data_laed = copy.deepcopy(data)

            logger.info(
                f"Start scenario: system={system_name}, mechs={sorted(mechanisms)}, "
                f"ramp={ramp_regime_name} (x{ramp_multiplier:g}), H={n_t}, N_g={n_g}, N_T={n_T}"
            )

            # ------------------------------------------------------------------
            # Ramp diagnostics (pre-solve): compare load ramps vs ramp capability
            # ------------------------------------------------------------------
            load_full = np.array([float(load[t]) for t in range(1, n_T + 1)], dtype=float)
            if load_full.size >= 2:
                delta_d = np.diff(load_full)
                max_abs_delta_d = float(np.max(np.abs(delta_d)))
                mean_abs_delta_d = float(np.mean(np.abs(delta_d)))
            else:
                max_abs_delta_d = 0.0
                mean_abs_delta_d = 0.0

            ramp_unscaled = np.array([float(data.data()["Ramp_lim"][g + 1]) for g in range(n_g)], dtype=float)
            ramp_scaled = ramp_unscaled * float(ramp_multiplier)
            sum_ramp_capability = float(np.sum(ramp_scaled))
            max_generator_ramp = float(np.max(ramp_scaled)) if ramp_scaled.size else 0.0
            ramp_stress_ratio = (
                float(max_abs_delta_d / sum_ramp_capability) if sum_ramp_capability > 0 else float("inf")
            )

            ramp_diag.update(
                {
                    "max_abs_delta_d": max_abs_delta_d,
                    "mean_abs_delta_d": mean_abs_delta_d,
                    "sum_ramp_capability": sum_ramp_capability,
                    "max_generator_ramp": max_generator_ramp,
                    "ramp_stress_ratio": ramp_stress_ratio,
                }
            )

            logger.info(
                "Ramp diagnostic (pre): "
                f"max|Δd|={max_abs_delta_d:.6g}, mean|Δd|={mean_abs_delta_d:.6g}, "
                f"sum(αR)={sum_ramp_capability:.6g}, max(αR)={max_generator_ramp:.6g}, ρ={ramp_stress_ratio:.6g}"
            )
            if ramp_stress_ratio < 0.1:
                logger.warning(
                    "WARNING: Demand trajectory does not stress ramp limits. "
                    "TLMP adjustments may be zero (TLMP collapses to LA)."
                )

            # Run and build the scenario time series.
            columns: dict[str, Any] = {}
            metrics_out: dict[str, Any] = {
                "lmp_energy": None,
                "la_energy": None,
                "tlmp_load": None,
                "tlmp_marginal": None,
                "mw_total_rp": None,
                "mw_total_la": None,
                "mw_total_tlmp": None,
                "mw_per_mwh_rp": None,
                "mw_per_mwh_la": None,
                "mw_per_mwh_tlmp": None,
                "loc_total_rp": None,
                "loc_total_la": None,
                "loc_total_tlmp": None,
                "loc_per_mwh_rp": None,
                "loc_per_mwh_la": None,
                "loc_per_mwh_tlmp": None,
            }
            summary_rows: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []

            def _add_series(series_name: str, series: np.ndarray) -> None:
                m = volatility_metrics(series, ddof=1)
                metrics_out[series_name] = m
                summary_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "system": system_name,
                        "ramp_regime": ramp_regime_name,
                        "horizon": int(horizon),
                        "seed": int(scenario_seed),
                        "series": series_name,
                        "n": int(m["n"]),
                        "mean": m["mean"],
                        "sigma": m["sigma"],
                        "V_MA": m["V_MA"],
                        "Delta_max": m["Delta_max"],
                        "TV": m["TV"],
                    }
                )

            # Demand over committed intervals. ED and LAED in this repo both return n_steps = N_T - N_t + 1.
            n_steps = int(n_T - n_t + 1)
            if n_steps < 1:
                raise ValueError(f"Invalid horizon: N_t={n_t} must be <= N_T={n_T}.")
            demand = np.array([float(load[t]) for t in range(1, n_steps + 1)], dtype=float)
            columns["t"] = np.arange(1, n_steps + 1, dtype=int)
            columns["demand"] = demand

            lmp_pi = None
            la_pi = None
            tlmp_load = None
            tlmp_marg = None
            P_ed_commit = None
            P_laed_commit = None
            tlmp_by_gen_commit = None
            shed_ed = None
            shed_laed = None

            if "lmp" in mechanisms:
                try:
                    P_ed, _shed, LMP_ed, TLMP_ed, rup_ed, rupp_ed, rdw_ed, rdwp_ed = lra.ED_no_errors(
                        data_ed, n_g, n_t, n_T, 1.0, ramp_multiplier, solver
                    )
                    P_ed_commit = np.asarray(P_ed, dtype=float)
                    shed_ed = np.asarray(_shed, dtype=float).reshape(-1)
                    # For RPED, incorporate the ramping capability duals into the energy settlement price.
                    # (This corresponds to the "TLMP" returned by `lra.ED_no_errors` / `lra.LMP_calculation`.)
                    lmp_pi = np.asarray(TLMP_ed[0, :], dtype=float)
                    lambda_lmp = np.asarray(LMP_ed[0, :], dtype=float)
                    columns["pi_lmp_energy"] = lmp_pi
                    columns["lambda_lmp"] = lambda_lmp
                    columns["pi_lmp_energy_ramp_adder"] = lmp_pi - lambda_lmp
                    # Also export the ramp product shadow prices (system-level) for diagnostics.
                    columns["pi_rp_ramp_up"] = np.asarray(rupp_ed[0, :], dtype=float)
                    columns["pi_rp_ramp_down"] = np.asarray(rdwp_ed[0, :], dtype=float)
                    for g in range(n_g):
                        columns[f"p_ed_g{g+1}"] = np.asarray(P_ed[g, :], dtype=float)
                    _add_series("lmp_energy", lmp_pi)
                except Exception as e:
                    logger.error("LMP solve failed:\n" + exc_to_str(e))
                    failures.append({"mechanism": "lmp", "error": str(e)})

            if ("la" in mechanisms) or ("tlmp" in mechanisms):
                try:
                    P_laed, _shed, TLMP_laed, LLMP_laed, *_rest = lra.LAED_No_Errors(
                        data_laed, n_g, n_t, n_T, 1.0, ramp_multiplier, solver
                    )
                    P_laed_commit = np.asarray(P_laed, dtype=float)
                    shed_laed = np.asarray(_shed, dtype=float).reshape(-1)
                    la_pi = np.asarray(LLMP_laed[0, :], dtype=float)
                    columns["pi_la_energy"] = la_pi
                    columns["lambda_la"] = la_pi
                    for g in range(n_g):
                        columns[f"p_laed_g{g+1}"] = np.asarray(P_laed[g, :], dtype=float)

                    # --------------------------------------------------------------
                    # Ramp diagnostics (post-solve): ramp usage, binding, TLMP activity
                    # --------------------------------------------------------------
                    gen_init = np.array([float(data.data()["Gen_init"][g + 1]) for g in range(n_g)], dtype=float)
                    # Include the initial transition (Gen_init -> first committed dispatch).
                    delta_p0 = P_laed_commit[:, 0] - gen_init
                    if P_laed_commit.shape[1] >= 2:
                        delta_p_rest = P_laed_commit[:, 1:] - P_laed_commit[:, :-1]
                        delta_p_all = np.concatenate([delta_p0.reshape(n_g, 1), delta_p_rest], axis=1)
                    else:
                        delta_p_all = delta_p0.reshape(n_g, 1)

                    max_ramp_usage = float(np.max(np.abs(delta_p_all))) if delta_p_all.size else 0.0
                    max_ramp_limit = float(np.max(ramp_scaled)) if ramp_scaled.size else 0.0

                    tol = 1e-6
                    slack = ramp_scaled.reshape(n_g, 1) - np.abs(delta_p_all)
                    count_binding_ramps = int(np.sum(slack < tol))
                    denom = int(n_g * delta_p_all.shape[1])
                    fraction_binding_ramps = float(count_binding_ramps / denom) if denom > 0 else 0.0

                    tlmp_by_gen = np.asarray(TLMP_laed, dtype=float)
                    tlmp_adjustment = tlmp_by_gen - la_pi.reshape(1, -1)
                    mean_abs_tlmp_adj = float(np.mean(np.abs(tlmp_adjustment))) if tlmp_adjustment.size else 0.0
                    max_abs_tlmp_adj = float(np.max(np.abs(tlmp_adjustment))) if tlmp_adjustment.size else 0.0

                    ramp_diag.update(
                        {
                            "max_ramp_usage": max_ramp_usage,
                            "max_ramp_limit": max_ramp_limit,
                            "count_binding_ramps": count_binding_ramps,
                            "fraction_binding_ramps": fraction_binding_ramps,
                            "mean_abs_TLMP_adjustment": mean_abs_tlmp_adj,
                            "max_abs_TLMP_adjustment": max_abs_tlmp_adj,
                            "binding_tolerance": tol,
                        }
                    )

                    logger.info(
                        "Ramp diagnostic (post): "
                        f"max|Δp|={max_ramp_usage:.6g}, max(αR)={max_ramp_limit:.6g}, "
                        f"binding={count_binding_ramps}/{denom} ({fraction_binding_ramps:.3%}), "
                        f"max|TLMP-LA|={max_abs_tlmp_adj:.6g}"
                    )
                    if max_abs_tlmp_adj == 0.0:
                        logger.warning("WARNING: TLMP adjustments are zero. Ramp constraints not binding.")

                    if "la" in mechanisms:
                        _add_series("la_energy", la_pi)

                    if "tlmp" in mechanisms:
                        tlmp = np.asarray(TLMP_laed, dtype=float)
                        tlmp_by_gen_commit = tlmp
                        p = np.asarray(P_laed, dtype=float)
                        tlmp_load = tlmp_price_paid_by_load(tlmp_by_gen=tlmp, dispatch_by_gen=p, demand=demand)

                        cap = np.array([float(data.data()["Capacity"][g + 1]) for g in range(n_g)], dtype=float)
                        tlmp_marg = tlmp_marginal_unit_series(
                            tlmp_by_gen=tlmp,
                            dispatch_by_gen=p,
                            capacity_by_gen=cap,
                            fallback=tlmp_load,
                        )

                        columns["pi_tlmp_load"] = tlmp_load
                        columns["pi_tlmp_marginal"] = tlmp_marg
                        _add_series("tlmp_load", tlmp_load)
                        _add_series("tlmp_marginal", tlmp_marg)
                except Exception as e:
                    logger.error("LA/ TLMP solve failed:\n" + exc_to_str(e))
                    failures.append({"mechanism": "la/tlmp", "error": str(e)})

            # Choose pi_energy for the contract: prefer LMP, then LA, then TLMP-load.
            if lmp_pi is not None:
                columns["pi_energy"] = lmp_pi
                # Keep `lambda` as the base energy balance dual when available.
                columns["lambda"] = columns.get("lambda_lmp", lmp_pi)
                meta["pi_energy_source"] = "lmp_energy"
                if P_ed_commit is not None:
                    for g in range(n_g):
                        columns[f"p_g{g+1}"] = P_ed_commit[g, :]
            elif la_pi is not None:
                columns["pi_energy"] = la_pi
                columns["lambda"] = la_pi
                meta["pi_energy_source"] = "la_energy"
                if P_laed_commit is not None:
                    for g in range(n_g):
                        columns[f"p_g{g+1}"] = P_laed_commit[g, :]
            elif tlmp_load is not None:
                columns["pi_energy"] = tlmp_load
                columns["lambda"] = la_pi if la_pi is not None else tlmp_load
                meta["pi_energy_source"] = "tlmp_load"
                if P_laed_commit is not None:
                    for g in range(n_g):
                        columns[f"p_g{g+1}"] = P_laed_commit[g, :]
            else:
                raise RuntimeError("No price series computed (all requested mechanisms failed).")

            if failures:
                meta["failures"] = failures

            meta["n_steps"] = int(np.asarray(columns["t"]).size)

            # ------------------------------------------------------------------
            # Make-whole payments (per generator, per mechanism)
            # ------------------------------------------------------------------
            dt_hours_cfg = cfg.get("dt_hours")
            time_cfg = cfg.get("time") or {}
            if dt_hours_cfg is not None:
                dt_hours = float(dt_hours_cfg)
            elif (time_cfg.get("dt_hours") is not None) and not isinstance(time_cfg.get("dt_hours"), dict):
                dt_hours = float(time_cfg["dt_hours"])
            elif time_cfg.get("dt_minutes") is not None:
                dt_hours = float(time_cfg["dt_minutes"]) / 60.0
            else:
                # Heuristic based on known MISO series lengths.
                dt_hours = 5.0 / 60.0 if int(n_T) == 288 else 1.0

            meta["dt_hours"] = float(dt_hours)

            cost_coef = np.array([float(data.data()["Cost"][g + 1]) for g in range(n_g)], dtype=float)
            demand_mwh = float(np.sum(demand) * dt_hours)

            mw_rows: list[dict[str, Any]] = []

            def _mw_for_price(
                *,
                mechanism: str,
                p_commit: np.ndarray,
                pi: np.ndarray,
                gen_specific_price: bool,
            ) -> tuple[float, float]:
                p_arr = np.asarray(p_commit, dtype=float)
                pi_arr = np.asarray(pi, dtype=float)
                if p_arr.shape != (n_g, n_steps):
                    raise ValueError(f"dispatch shape mismatch for MW: got {p_arr.shape}, expected {(n_g, n_steps)}")
                if gen_specific_price:
                    if pi_arr.shape != (n_g, n_steps):
                        raise ValueError(
                            f"gen-specific price shape mismatch for MW: got {pi_arr.shape}, expected {(n_g, n_steps)}"
                        )
                else:
                    if pi_arr.shape != (n_steps,):
                        raise ValueError(f"price shape mismatch for MW: got {pi_arr.shape}, expected {(n_steps,)}")

                if np.min(p_arr) < -1e-6:
                    raise ValueError(f"Negative dispatch detected for MW under mechanism '{mechanism}'.")

                mw_total = 0.0
                for gi in range(n_g):
                    p_g = p_arr[gi, :]
                    total_cost = float(np.sum(cost_coef[gi] * p_g) * dt_hours)
                    if gen_specific_price:
                        total_rev = float(np.sum(pi_arr[gi, :] * p_g) * dt_hours)
                    else:
                        total_rev = float(np.sum(pi_arr * p_g) * dt_hours)
                    mw = max(0.0, total_cost - total_rev)
                    mw_total += mw
                    mw_rows.append(
                        {
                            "generator": int(gi + 1),
                            "mechanism": str(mechanism),
                            "total_cost": total_cost,
                            "total_revenue": total_rev,
                            "make_whole": mw,
                        }
                    )

                mw_per_mwh = float(mw_total / demand_mwh) if demand_mwh > 0 else float("nan")
                return float(mw_total), float(mw_per_mwh)

            # RP make-whole (RPED energy price includes ramping-dual adjustment)
            if (P_ed_commit is not None) and (lmp_pi is not None):
                mw_total_rp, mw_per_mwh_rp = _mw_for_price(
                    mechanism="rp",
                    p_commit=P_ed_commit,
                    pi=lmp_pi,
                    gen_specific_price=False,
                )
                metrics_out["mw_total_rp"] = mw_total_rp
                metrics_out["mw_per_mwh_rp"] = mw_per_mwh_rp

            # LA make-whole
            if (P_laed_commit is not None) and (la_pi is not None) and ("la" in mechanisms):
                mw_total_la, mw_per_mwh_la = _mw_for_price(
                    mechanism="la",
                    p_commit=P_laed_commit,
                    pi=la_pi,
                    gen_specific_price=False,
                )
                metrics_out["mw_total_la"] = mw_total_la
                metrics_out["mw_per_mwh_la"] = mw_per_mwh_la

            # TLMP make-whole (generator-specific settlement prices)
            if (P_laed_commit is not None) and (tlmp_by_gen_commit is not None) and ("tlmp" in mechanisms):
                mw_total_tlmp, mw_per_mwh_tlmp = _mw_for_price(
                    mechanism="tlmp",
                    p_commit=P_laed_commit,
                    pi=tlmp_by_gen_commit,
                    gen_specific_price=True,
                )
                metrics_out["mw_total_tlmp"] = mw_total_tlmp
                metrics_out["mw_per_mwh_tlmp"] = mw_per_mwh_tlmp

            # Attach MW summary fields to each per-series summary row so they appear in summary_metrics.csv.
            if mw_rows:
                for r in summary_rows:
                    for k_mw in (
                        "mw_total_rp",
                        "mw_total_la",
                        "mw_total_tlmp",
                        "mw_per_mwh_rp",
                        "mw_per_mwh_la",
                        "mw_per_mwh_tlmp",
                    ):
                        if k_mw in metrics_out:
                            r[k_mw] = metrics_out[k_mw]

                # Basic sanity / escalation logging if MW is huge.
                for mech_key in ("rp", "la", "tlmp"):
                    tot = metrics_out.get(f"mw_total_{mech_key}")
                    per = metrics_out.get(f"mw_per_mwh_{mech_key}")
                    if tot is None:
                        continue
                    if (float(tot) > 1e7) or (per is not None and float(per) > 1e3):
                        if mech_key == "rp" and shed_ed is not None:
                            shed_frac = float(np.mean(shed_ed > 1e-6))
                        elif mech_key in ("la", "tlmp") and shed_laed is not None:
                            shed_frac = float(np.mean(shed_laed > 1e-6))
                        else:
                            shed_frac = float("nan")

                        if mech_key == "rp" and lmp_pi is not None:
                            max_price = float(np.max(lmp_pi))
                        elif mech_key in ("la", "tlmp") and la_pi is not None:
                            max_price = float(np.max(la_pi))
                        elif mech_key == "tlmp" and tlmp_by_gen_commit is not None:
                            max_price = float(np.max(tlmp_by_gen_commit))
                        else:
                            max_price = float("nan")

                        logger.warning(
                            f"Make-whole unusually large for {mech_key}: total={tot:.6g}, per_mwh={per}. "
                            f"shed_frac={shed_frac:.3%}, max_price={max_price:.6g}, dt_hours={dt_hours}"
                        )

                write_summary_csv(scenario_dir / "make_whole.csv", mw_rows)

            # ------------------------------------------------------------------
            # Lost opportunity cost (LOC): self-scheduling profit gap
            # ------------------------------------------------------------------
            loc_rows: list[dict[str, Any]] = []
            loc_tol = 1e-6

            pmax = np.array([float(data.data()["Capacity"][g + 1]) for g in range(n_g)], dtype=float)
            gen_init = np.array([float(data.data()["Gen_init"][g + 1]) for g in range(n_g)], dtype=float)

            def _self_schedule_optimal_profit(pi_by_gen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                """
                Solve the self-scheduling problem for all generators at once:

                    max_{p[g,t]} sum_{g,t} (pi[g,t] - c[g]) * p[g,t] * dt_hours

                subject to:
                    0 <= p[g,t] <= Pmax[g]
                    -R[g] <= p[g,t] - p[g,t-1] <= R[g]   (t>=1)
                    -R[g] <= p[g,0] - Gen_init[g] <= R[g]
                """
                pi_mat = np.asarray(pi_by_gen, dtype=float)
                if pi_mat.shape != (n_g, n_steps):
                    raise ValueError(f"pi_by_gen must have shape {(n_g, n_steps)}; got {pi_mat.shape}")
                if not np.isfinite(pi_mat).all():
                    raise ValueError("pi_by_gen contains non-finite values (nan/inf).")

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
                    return m.p[g, t] <= float(pmax[g])

                m.cap = Constraint(m.G, m.T, rule=_cap_rule)

                if n_steps >= 2:
                    m.T_ramp = RangeSet(1, n_steps - 1)

                    def _rup_rule(m, g, t):
                        return m.p[g, t] - m.p[g, t - 1] <= float(ramp_scaled[g])

                    def _rdn_rule(m, g, t):
                        return m.p[g, t - 1] - m.p[g, t] <= float(ramp_scaled[g])

                    m.rup = Constraint(m.G, m.T_ramp, rule=_rup_rule)
                    m.rdn = Constraint(m.G, m.T_ramp, rule=_rdn_rule)

                def _rup0_rule(m, g):
                    return m.p[g, 0] - float(gen_init[g]) <= float(ramp_scaled[g])

                def _rdn0_rule(m, g):
                    return float(gen_init[g]) - m.p[g, 0] <= float(ramp_scaled[g])

                m.rup0 = Constraint(m.G, rule=_rup0_rule)
                m.rdn0 = Constraint(m.G, rule=_rdn0_rule)

                def _obj_rule(m):
                    return float(dt_hours) * sum(
                        (float(pi_mat[g, t]) - float(cost_coef[g])) * m.p[g, t] for g in m.G for t in m.T
                    )

                m.obj = Objective(rule=_obj_rule, sense=maximize)

                res = solver.solve(m, tee=False)
                term = str(res.solver.termination_condition).lower()
                status = str(res.solver.status).lower()
                if term != "optimal":
                    raise RuntimeError(f"LOC self-schedule solve failed: status={status}, term={term}")

                p_opt = np.array([[float(value(m.p[g, t])) for t in m.T] for g in m.G], dtype=float)
                q_opt = np.sum((pi_mat - cost_coef.reshape(-1, 1)) * p_opt, axis=1) * float(dt_hours)
                return q_opt, p_opt

            def _realized_profit(
                *, p_commit: np.ndarray, pi_by_gen: np.ndarray, mechanism: str
            ) -> np.ndarray:
                p_arr = np.asarray(p_commit, dtype=float)
                pi_mat = np.asarray(pi_by_gen, dtype=float)
                if p_arr.shape != (n_g, n_steps):
                    raise ValueError(f"dispatch shape mismatch for LOC: got {p_arr.shape}, expected {(n_g, n_steps)}")
                if pi_mat.shape != (n_g, n_steps):
                    raise ValueError(f"price shape mismatch for LOC: got {pi_mat.shape}, expected {(n_g, n_steps)}")
                if np.min(p_arr) < -1e-6:
                    raise ValueError(f"Negative dispatch detected for LOC under mechanism '{mechanism}'.")
                return np.sum((pi_mat - cost_coef.reshape(-1, 1)) * p_arr, axis=1) * float(dt_hours)

            # Prepare storage (one row per generator).
            loc_by_g: dict[str, np.ndarray] = {}
            q_by_g: dict[str, np.ndarray] = {}
            profit_by_g: dict[str, np.ndarray] = {}

            # RP LOC: RPED dispatch + RP energy prices (with ramping-dual adjustment)
            if (P_ed_commit is not None) and (lmp_pi is not None):
                pi_rp_mat = np.repeat(np.asarray(lmp_pi, dtype=float).reshape(1, -1), n_g, axis=0)
                q_rp, _p_rp_opt = _self_schedule_optimal_profit(pi_rp_mat)
                prof_rp = _realized_profit(p_commit=P_ed_commit, pi_by_gen=pi_rp_mat, mechanism="rp")
                loc_rp = q_rp - prof_rp
                q_by_g["rp"] = q_rp
                profit_by_g["rp"] = prof_rp
                loc_by_g["rp"] = loc_rp

            # LA LOC: LAED dispatch + LA prices
            if (P_laed_commit is not None) and (la_pi is not None) and ("la" in mechanisms):
                pi_la_mat = np.repeat(np.asarray(la_pi, dtype=float).reshape(1, -1), n_g, axis=0)
                q_la, _p_la_opt = _self_schedule_optimal_profit(pi_la_mat)
                prof_la = _realized_profit(p_commit=P_laed_commit, pi_by_gen=pi_la_mat, mechanism="la")
                loc_la = q_la - prof_la
                q_by_g["la"] = q_la
                profit_by_g["la"] = prof_la
                loc_by_g["la"] = loc_la

            # TLMP LOC: LAED dispatch + generator-specific TLMP prices
            if (P_laed_commit is not None) and (tlmp_by_gen_commit is not None) and ("tlmp" in mechanisms):
                pi_tlmp_mat = np.asarray(tlmp_by_gen_commit, dtype=float)
                q_tlmp, _p_tlmp_opt = _self_schedule_optimal_profit(pi_tlmp_mat)
                prof_tlmp = _realized_profit(p_commit=P_laed_commit, pi_by_gen=pi_tlmp_mat, mechanism="tlmp")
                loc_tlmp = q_tlmp - prof_tlmp
                q_by_g["tlmp"] = q_tlmp
                profit_by_g["tlmp"] = prof_tlmp
                loc_by_g["tlmp"] = loc_tlmp

            if loc_by_g:
                for mech, loc_vec in loc_by_g.items():
                    neg = np.where(loc_vec < -loc_tol)[0]
                    if neg.size:
                        # Dump a compact set of diagnostics for debugging.
                        if mech == "rp":
                            pi_dump = np.asarray(lmp_pi, dtype=float) if lmp_pi is not None else np.array([])
                            p_dump = np.asarray(P_ed_commit, dtype=float) if P_ed_commit is not None else np.zeros((n_g, n_steps))
                        elif mech == "la":
                            pi_dump = np.asarray(la_pi, dtype=float) if la_pi is not None else np.array([])
                            p_dump = np.asarray(P_laed_commit, dtype=float) if P_laed_commit is not None else np.zeros((n_g, n_steps))
                        else:
                            pi_dump = np.asarray(tlmp_by_gen_commit, dtype=float).reshape(-1) if tlmp_by_gen_commit is not None else np.array([])
                            p_dump = np.asarray(P_laed_commit, dtype=float) if P_laed_commit is not None else np.zeros((n_g, n_steps))

                        for gi in neg.tolist():
                            logger.error(
                                f"LOC negative beyond tolerance: mech={mech}, g={gi+1}, loc={loc_vec[gi]:.6g}. "
                                f"Pmax={pmax[gi]:.6g}, R={ramp_scaled[gi]:.6g}, "
                                f"pi[min,max]=({float(np.min(pi_dump)) if pi_dump.size else float('nan'):.6g},"
                                f"{float(np.max(pi_dump)) if pi_dump.size else float('nan'):.6g}), "
                                f"p[min,max]=({float(np.min(p_dump[gi,:])):.6g},{float(np.max(p_dump[gi,:])):.6g})"
                            )

                # Export per-generator LOC table (one row per generator with columns per mechanism).
                for gi in range(n_g):
                    row: dict[str, Any] = {"generator": int(gi + 1)}
                    for mech in ("rp", "la", "tlmp"):
                        qv = q_by_g.get(mech)
                        pv = profit_by_g.get(mech)
                        lv = loc_by_g.get(mech)
                        row[f"Q_{mech}"] = float(qv[gi]) if qv is not None else None
                        row[f"profit_{mech}"] = float(pv[gi]) if pv is not None else None
                        row[f"loc_{mech}"] = float(lv[gi]) if lv is not None else None
                    loc_rows.append(row)

                write_summary_csv(scenario_dir / "loc.csv", loc_rows)

                # Totals + normalized per MWh served.
                if "rp" in loc_by_g:
                    loc_total_rp = float(np.sum(np.maximum(0.0, loc_by_g["rp"])))
                    served_mwh_rp = (
                        float(np.sum(np.maximum(0.0, demand - shed_ed)) * dt_hours)
                        if shed_ed is not None
                        else demand_mwh
                    )
                    metrics_out["loc_total_rp"] = loc_total_rp
                    metrics_out["loc_per_mwh_rp"] = float(loc_total_rp / served_mwh_rp) if served_mwh_rp > 0 else float("nan")
                if "la" in loc_by_g:
                    loc_total_la = float(np.sum(np.maximum(0.0, loc_by_g["la"])))
                    served_mwh_la = (
                        float(np.sum(np.maximum(0.0, demand - shed_laed)) * dt_hours)
                        if shed_laed is not None
                        else demand_mwh
                    )
                    metrics_out["loc_total_la"] = loc_total_la
                    metrics_out["loc_per_mwh_la"] = float(loc_total_la / served_mwh_la) if served_mwh_la > 0 else float("nan")
                if "tlmp" in loc_by_g:
                    loc_total_tlmp = float(np.sum(np.maximum(0.0, loc_by_g["tlmp"])))
                    served_mwh_tlmp = (
                        float(np.sum(np.maximum(0.0, demand - shed_laed)) * dt_hours)
                        if shed_laed is not None
                        else demand_mwh
                    )
                    metrics_out["loc_total_tlmp"] = loc_total_tlmp
                    metrics_out["loc_per_mwh_tlmp"] = (
                        float(loc_total_tlmp / served_mwh_tlmp) if served_mwh_tlmp > 0 else float("nan")
                    )

                    # Convex-theory check: TLMP should typically reduce LOC.
                    if loc_total_tlmp > 1e-3:
                        shed_frac = float(np.mean(shed_laed > 1e-6)) if shed_laed is not None else float("nan")
                        logger.warning(
                            "TLMP LOC is nontrivial: "
                            f"loc_total_tlmp={loc_total_tlmp:.6g}, "
                            f"max|TLMP-LA|={float(ramp_diag.get('max_abs_TLMP_adjustment', float('nan'))):.6g}, "
                            f"binding_ramps={int(ramp_diag.get('count_binding_ramps', 0) or 0)}, "
                            f"shed_frac={shed_frac:.3%}"
                        )

                # Attach LOC summary fields to each per-series summary row so they appear in summary_metrics.csv.
                for r in summary_rows:
                    for k_loc in (
                        "loc_total_rp",
                        "loc_total_la",
                        "loc_total_tlmp",
                        "loc_per_mwh_rp",
                        "loc_per_mwh_la",
                        "loc_per_mwh_tlmp",
                    ):
                        if k_loc in metrics_out:
                            r[k_loc] = metrics_out[k_loc]

            # Write artifacts.
            write_timeseries_csv(scenario_dir / "timeseries.csv", columns)
            write_json(scenario_dir / "metrics.json", metrics_out)
            if len(ramp_diag) > 1:
                write_json(scenario_dir / "ramp_diagnostics.json", ramp_diag)

        meta["runtime_s"] = float(time.perf_counter() - t0)
        meta["success"] = True
        meta["env"] = environment_meta(repo_root=repo_root)
        if len(ramp_diag) > 1:
            meta["ramp_diagnostics"] = ramp_diag
        write_json(scenario_dir / "meta.json", meta)

        # Optional plots.
        if plots:
            try:
                import matplotlib.pyplot as plt  # type: ignore

                x = np.asarray(columns["t"], dtype=int)
                plt.figure(figsize=(11, 4))
                to_plot = [
                    # Plot solid first, then patterned lines so overlaps remain visible.
                    ("LMP energy", "pi_lmp_energy", {"color": "C0", "linestyle": "-", "linewidth": 1.1, "alpha": 0.95}),
                    ("LA energy", "pi_la_energy", {"color": "C1", "linestyle": "--", "linewidth": 1.1, "alpha": 0.95}),
                    ("TLMP load", "pi_tlmp_load", {"color": "C2", "linestyle": ":", "linewidth": 1.3, "alpha": 0.95}),
                    ("TLMP marginal", "pi_tlmp_marginal", {"color": "C3", "linestyle": "-.", "linewidth": 1.1, "alpha": 0.95}),
                ]
                for label, key, style in to_plot:
                    if key in columns:
                        plt.plot(x, np.asarray(columns[key], dtype=float), label=label, **style)
                plt.xlabel("Time (5-min intervals)")
                plt.ylabel("Price ($/MWh)")
                n_g_title = int(meta.get("N_g", 0))
                ramp_mult_title = float(meta.get("ramp_multiplier", float("nan")))
                plt.title(f"Energy Price ({n_g_title} Generators {ramp_mult_title:g} Ramp Factor)")
                plt.grid(True, alpha=0.25)
                plt.legend()
                plt.tight_layout()
                plt.savefig(scenario_dir / "prices.png", dpi=200)
                plt.close()
            except Exception as e:
                logger.info(f"Plotting skipped/failed: {e}")

        logger.info(f"Scenario completed in {meta['runtime_s']:.3f}s")
        ramp_diag["success"] = True
        return summary_rows, ramp_diag

    except Exception as e:
        meta["runtime_s"] = float(time.perf_counter() - t0)
        meta["success"] = False
        meta["error"] = str(e)
        meta["traceback"] = exc_to_str(e)
        meta["env"] = environment_meta(repo_root=repo_root)
        write_json(scenario_dir / "meta.json", meta)
        write_text(scenario_dir / "FAILED.txt", meta["traceback"])
        logger.error("Scenario failed:\n" + meta["traceback"])
        ramp_diag["success"] = False
        if len(ramp_diag) > 1:
            try:
                write_json(scenario_dir / "ramp_diagnostics.json", ramp_diag)
            except Exception:
                pass
        return [], ramp_diag


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    out_root = Path(args.out)
    ensure_dir(out_root)
    if args.no_run_subdir:
        out_dir = out_root
    else:
        run_name = slugify(str(args.run_name)) if args.run_name else time.strftime("%Y-%m-%d_%H%M%S")
        out_dir = out_root / run_name
        if out_dir.exists():
            for k in range(1, 10_000):
                candidate = out_root / f"{run_name}-{k}"
                if not candidate.exists():
                    out_dir = candidate
                    break
            else:
                raise RuntimeError(f"Failed to find an unused run folder under {out_root} for base '{run_name}'.")
        ensure_dir(out_dir)

    # Persist the exact config file used for this run for reproducibility.
    # (We intentionally copy raw text instead of re-serializing `cfg`.)
    write_text(out_dir / "config.yaml", cfg_path.read_text(encoding="utf-8"))

    base_seed = int(args.seed if args.seed is not None else cfg.get("seed", 123))
    systems_cfg: dict[str, Any] = cfg.get("systems") or {}
    ramp_regimes_cfg: dict[str, Any] = cfg.get("ramp_regimes") or {}
    mechanisms_cfg: list[str] = list(cfg.get("mechanisms") or [])
    horizons_cfg: list[int] = list((cfg.get("lookahead") or {}).get("horizons") or [4])

    systems = args.systems if args.systems is not None else list(systems_cfg.keys())
    ramp_regimes = args.ramp_regimes if args.ramp_regimes is not None else list(ramp_regimes_cfg.keys())
    mechanisms = args.mechanisms if args.mechanisms is not None else mechanisms_cfg
    horizons = args.horizons if args.horizons is not None else horizons_cfg

    # Normalize/validate mechanisms early.
    mechanisms = [str(m).lower() for m in mechanisms]
    allowed_mechs = {"lmp", "la", "tlmp"}
    unknown_mechs = sorted(set(mechanisms) - allowed_mechs)
    if unknown_mechs:
        raise SystemExit(f"Unknown mechanisms: {unknown_mechs}. Allowed: {sorted(allowed_mechs)}")

    plots = bool(cfg.get("plots", True)) and (not args.no_plots)
    console = not args.no_console

    # Sweep.
    summary_rows: list[dict[str, Any]] = []
    any_ramp_diag = False
    any_ramp_scarcity = False
    for system_name in systems:
        if system_name not in systems_cfg:
            raise SystemExit(f"Unknown system '{system_name}'. Available: {sorted(systems_cfg.keys())}")
        for ramp_regime_name in ramp_regimes:
            if ramp_regime_name not in ramp_regimes_cfg:
                raise SystemExit(
                    f"Unknown ramp_regime '{ramp_regime_name}'. Available: {sorted(ramp_regimes_cfg.keys())}"
                )
            for horizon in horizons:
                scenario_id = slugify(
                    "__".join(
                        [
                            system_name,
                            ramp_regime_name,
                            f"H{int(horizon)}",
                            f"seed{int(base_seed)}",
                        ]
                    )
                )
                rows, ramp_diag = _run_single_scenario(
                    out_dir=out_dir,
                    scenario_id=scenario_id,
                    cfg=cfg,
                    system_name=system_name,
                    system_cfg=systems_cfg[system_name],
                    ramp_regime_name=ramp_regime_name,
                    ramp_regime_cfg=ramp_regimes_cfg[ramp_regime_name],
                    mechanisms=mechanisms,
                    horizon=int(horizon),
                    base_seed=base_seed,
                    console=console,
                    plots=plots,
                )
                summary_rows.extend(rows)
                if ramp_diag:
                    # A scenario has ramp diagnostics if the pre-solve fields are present.
                    if "ramp_stress_ratio" in ramp_diag:
                        any_ramp_diag = True
                    # Scarcity if any binding ramp transition OR any non-zero TLMP adjustment is observed.
                    if (ramp_diag.get("count_binding_ramps", 0) or 0) > 0:
                        any_ramp_scarcity = True
                    elif float(ramp_diag.get("max_abs_TLMP_adjustment", 0.0) or 0.0) > 0.0:
                        any_ramp_scarcity = True

    # Global summaries.
    summary_csv = out_dir / "summary_metrics.csv"
    write_summary_csv(summary_csv, summary_rows)

    md_cols = ["system", "ramp_regime", "horizon", "series", "mean", "sigma", "V_MA", "Delta_max", "TV"]
    write_summary_markdown(out_dir / "summary_metrics.md", summary_rows, columns=md_cols)
    if any_ramp_diag and (not any_ramp_scarcity):
        print(
            "RAMP DIAGNOSTIC: No ramp scarcity detected.\n"
            "Consider reducing ramp factors or increasing load ramp rate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
