#!/usr/bin/env python3
"""
Sweep battery size and quantify how it changes the *committed* rolling dispatch.

This script is meant to answer: "If I make the battery bigger/smaller, how does the
actual (committed) LAED dispatch change?"

It:
  1) loads a Pyomo .dat case (defaults to toy_data_storage.dat)
  2) overwrites demand using MISO_Projection.json (defaults to 2032_Aug) scaled to avg demand
  3) runs LAED rolling simulation for a grid of battery sizes
  4) writes a summary CSV + an .npz bundle of time series, plus a few plots

Notes:
  - This uses the LAED implementation in laed_rp_with_storage.py, which commits t=1 each window.
    Therefore the returned time series length is n_steps = N_T - N_t + 1 (not N_T).
  - This file focuses on LAED. (RP/ED is implemented separately via ED_with_error.)
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

# Avoid matplotlib cache warnings on macOS when $HOME isn't writable in some environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib.pyplot as plt  # noqa: E402

from pyomo.environ import DataPortal, SolverFactory  # noqa: E402

import laed_rp_with_storage as lrs  # noqa: E402


STORAGE_KEYS = (
    "N_s",
    "E_cap",
    "P_ch_cap",
    "P_dis_cap",
    "SoC_init",
    "SoC_min",
    "SoC_max",
    "charge_cost",
    "discharge_cost",
    "eta_c",
    "eta_d",
    "delta_t",
)


def _scalar_param(base: dict, name: str, default=None):
    """Read a scalar param from a DataPortal.data() dict (handles {None: v} or v)."""
    if name not in base:
        return default
    val = base[name]
    if isinstance(val, dict):
        return val.get(None, default)
    return val


def strip_storage(base: dict) -> dict:
    b = copy.deepcopy(base)
    for k in STORAGE_KEYS:
        b.pop(k, None)
    return b


def _initial_dispatch(cost: dict, capacity: dict, load: float) -> dict:
    """
    Linear-cost economic dispatch to initialize Gen_init to a feasible starting point.
    """
    load = float(load)
    order = sorted(cost.keys(), key=lambda g: float(cost[g]))
    remaining = load
    dispatch = {int(g): 0.0 for g in cost.keys()}

    for g in order:
        g_int = int(g)
        cap = float(capacity[g_int])
        p = min(cap, max(0.0, remaining))
        dispatch[g_int] = float(p)
        remaining -= p
        if remaining <= 1e-9:
            break

    if remaining > 1e-6:
        total_cap = sum(float(capacity[int(g)]) for g in cost.keys())
        raise ValueError(f"Initial load {load} exceeds total capacity {total_cap}.")
    return dispatch


def _parse_list_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip() != ""]


def _infer_step_hours(N_T: int) -> float:
    # Common case in this repo: 288 points == 24 hours at 5-minute resolution.
    if int(N_T) == 288:
        return 1.0 / 12.0
    return 1.0


def _set_battery_params(
    base: dict,
    *,
    p_mw: float,
    duration_hr: float,
    soc_init_frac: float,
    soc_min_frac: float,
    eta_c: float,
    eta_d: float,
    delta_t: float,
    charge_cost: float,
    discharge_cost: float,
) -> dict:
    """
    Return a deep-copied base dict with a single battery sized by (power, duration).

    If p_mw <= 0, returns a storage-stripped base (no storage).
    """
    b = strip_storage(base)

    p_mw = float(p_mw)
    if p_mw <= 0.0:
        return b

    duration_hr = float(duration_hr)
    if duration_hr <= 0.0:
        raise ValueError("duration_hr must be positive")

    soc_init_frac = float(soc_init_frac)
    soc_min_frac = float(soc_min_frac)
    if not (0.0 <= soc_min_frac <= soc_init_frac <= 1.0):
        raise ValueError("Require 0 <= soc_min_frac <= soc_init_frac <= 1")

    e_mwh = p_mw * duration_hr
    soc_max = e_mwh
    soc_min = soc_min_frac * soc_max
    soc_init = soc_init_frac * soc_max

    b["N_s"] = {None: 1}
    b["E_cap"] = {1: float(e_mwh)}
    b["P_ch_cap"] = {1: float(p_mw)}
    b["P_dis_cap"] = {1: float(p_mw)}
    b["SoC_max"] = {1: float(soc_max)}
    b["SoC_min"] = {1: float(soc_min)}
    b["SoC_init"] = {1: float(soc_init)}
    b["charge_cost"] = {1: float(charge_cost)}
    b["discharge_cost"] = {1: float(discharge_cost)}
    b["eta_c"] = {None: float(eta_c)}
    b["eta_d"] = {None: float(eta_d)}
    b["delta_t"] = {None: float(delta_t)}
    return b


@dataclass
class DictData:
    _base: dict

    def data(self) -> dict:
        return self._base


def run_laed_dispatch(
    base: dict,
    *,
    N_g: int,
    N_t: int,
    N_T: int,
    load_factor: float,
    ramp_factor: float,
    sigma_rel: float,
    rho: float,
    seed: int,
    solver,
):
    rng = np.random.default_rng(int(seed))
    P, Shed, *_rest, SoC, Pch, Pdis = lrs.LAED_with_error(
        DictData(base),
        int(N_g),
        int(N_t),
        int(N_T),
        float(load_factor),
        float(ramp_factor),
        solver,
        float(sigma_rel),
        float(rho),
        rng,
        error_type="gaussian",
    )
    return np.asarray(P, dtype=float), np.asarray(Shed, dtype=float), np.asarray(SoC, dtype=float), np.asarray(Pch, dtype=float), np.asarray(Pdis, dtype=float)


def _dispatch_diff_metrics(P: np.ndarray, P0: np.ndarray) -> dict:
    # P and P0 shapes: (N_g, n_steps)
    d = P - P0
    metrics = {
        "dispatch_L1": float(np.sum(np.abs(d))),
        "dispatch_L2": float(np.sqrt(np.sum(d * d))),
        "dispatch_Linf": float(np.max(np.abs(d))),
    }
    for gi in range(P.shape[0]):
        dg = d[gi]
        metrics[f"g{gi+1}_L1"] = float(np.sum(np.abs(dg)))
        metrics[f"g{gi+1}_Linf"] = float(np.max(np.abs(dg)))
    return metrics


def _summarize_run(
    *,
    P: np.ndarray,
    Shed: np.ndarray,
    SoC: np.ndarray,
    Pch: np.ndarray,
    Pdis: np.ndarray,
    step_hours: float,
) -> dict:
    # P shape: (N_g, n_steps); storage arrays shape (N_s, n_steps) or (0, n_steps)
    out: dict[str, float] = {}

    out["shed_sum"] = float(np.sum(Shed))
    out["shed_mean"] = float(np.mean(Shed))
    out["shed_max"] = float(np.max(Shed))

    for gi in range(P.shape[0]):
        Pg = P[gi]
        out[f"g{gi+1}_mean_MW"] = float(np.mean(Pg))
        out[f"g{gi+1}_max_MW"] = float(np.max(Pg))
        out[f"g{gi+1}_energy_MWh"] = float(np.sum(Pg) * step_hours)

        # Physical committed ramp (between committed steps)
        if Pg.size >= 2:
            ramp = np.diff(Pg)
            out[f"g{gi+1}_ramp_up_max_MW_per_step"] = float(np.max(ramp))
            out[f"g{gi+1}_ramp_down_min_MW_per_step"] = float(np.min(ramp))
        else:
            out[f"g{gi+1}_ramp_up_max_MW_per_step"] = 0.0
            out[f"g{gi+1}_ramp_down_min_MW_per_step"] = 0.0

    if SoC.size == 0:
        out["batt_present"] = 0.0
        return out

    out["batt_present"] = 1.0
    soc = SoC[0]
    pch = Pch[0]
    pdis = Pdis[0]
    pnet = pdis - pch

    out["batt_soc_min_MWh"] = float(np.min(soc))
    out["batt_soc_max_MWh"] = float(np.max(soc))
    out["batt_pnet_max_MW"] = float(np.max(pnet))
    out["batt_pnet_min_MW"] = float(np.min(pnet))

    out["batt_discharge_energy_MWh"] = float(np.sum(pdis) * step_hours)
    out["batt_charge_energy_MWh"] = float(np.sum(pch) * step_hours)
    out["batt_throughput_MWh"] = float((np.sum(pdis) + np.sum(pch)) * step_hours)
    return out


def _maybe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _maybe_remove_heatmaps(out_dir: Path) -> None:
    # Keep output dirs tidy if older runs left heatmaps around.
    for p in out_dir.glob("heatmap_gen2_dispatch_*.png"):
        try:
            p.unlink()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="toy_data_storage.dat", help="Pyomo .dat case file")
    ap.add_argument("--solver", default="gurobi_direct", help="Pyomo solver name (e.g. gurobi_direct)")
    ap.add_argument("--laed-window", type=int, default=13, help="LAED window size N_t (default: 13)")
    ap.add_argument(
        "--laed-windows",
        default=None,
        help="Comma-separated LAED window sizes N_t (5-minute intervals, incl. current). Example: 13,25. Overrides --laed-window.",
    )
    ap.add_argument("--horizon", type=int, default=None, help="Physical horizon N_T (default: projection length)")
    ap.add_argument("--projection-file", default="MISO_Projection.json", help="JSON file with demand projection")
    ap.add_argument("--projection-key", default="2032_Aug", help="Top-level key in projection JSON (e.g. 2032_Aug)")
    ap.add_argument(
        "--avg-demand",
        type=float,
        default=None,
        help="(Deprecated) Single average demand in MW. If set, overrides --avg-demands.",
    )
    ap.add_argument(
        "--avg-demands",
        default="300,350,400",
        help="Comma-separated list of average demands in MW to overlay on the same plots (default: 300,350,400).",
    )
    ap.add_argument("--load-factor", type=float, default=1.0)
    ap.add_argument("--ramp-factor", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=0.05, help="Relative forecast error scale (Gaussian)")
    ap.add_argument("--rho", type=float, default=0.99, help="Inter-window correlation parameter in [0,1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost-load", type=float, default=1e10, help="Penalty for load shedding (default: 1e10)")
    ap.add_argument("--reserve-factor", type=float, default=0.0)

    ap.add_argument("--p-fracs", default="0,0.01,0.05,0.1,0.15,0.2", help="Comma-separated battery power fractions of avg demand")
    ap.add_argument("--duration-hr", type=float, default=3.0, help="Battery duration in hours (energy = P * duration)")
    ap.add_argument("--soc-init-frac", type=float, default=0.5, help="Initial SoC fraction of E_cap")
    ap.add_argument("--soc-min-frac", type=float, default=0.1, help="Minimum SoC fraction of E_cap")
    ap.add_argument(
        "--storage-cost",
        type=float,
        default=25.0,
        help="Storage throughput bid/cost applied to both charging and discharging (per MW per interval).",
    )
    ap.add_argument("--eta-c", type=float, default=None, help="Charge efficiency override (default: case file)")
    ap.add_argument("--eta-d", type=float, default=None, help="Discharge efficiency override (default: case file)")
    ap.add_argument("--delta-t", type=float, default=None, help="Time step in hours for SoC dynamics (default: infer; 288->1/12)")

    ap.add_argument("--out-dir", default="battery_size_dispatch_results", help="Output directory")
    ap.add_argument("--no-show", action="store_true", help="Do not show plots")
    args = ap.parse_args()

    # Set global knobs used by laed_rp_with_storage.py objective/constraints.
    lrs.cost_load = float(args.cost_load)
    lrs.reserve_factor = float(args.reserve_factor)

    data = DataPortal()
    data.load(filename=args.case)
    base_case = copy.deepcopy(data.data())

    # Override demand and horizon length using the projection file.
    with open(args.projection_file, "r") as f:
        interpolated_data = json.load(f)

    proj = interpolated_data[args.projection_key]
    if args.avg_demand is not None:
        avg_demands = [float(args.avg_demand)]
    else:
        avg_demands = _parse_list_floats(args.avg_demands)
    if not avg_demands:
        raise ValueError("No avg_demands provided.")
    avg_demands = sorted(set(float(x) for x in avg_demands))

    N_T_projection = int(len(proj))
    N_T = int(args.horizon) if args.horizon is not None else int(N_T_projection)
    if N_T > N_T_projection:
        raise ValueError(f"--horizon N_T={N_T} exceeds projection length={N_T_projection}")

    N_g = int(_scalar_param(base_case, "N_g"))
    if N_g <= 0:
        raise ValueError(f"Invalid N_g={N_g}")

    # Storage dynamics timestep (hours). Default: infer from horizon length (288 -> 5 min).
    step_hours = float(args.delta_t) if args.delta_t is not None else _infer_step_hours(N_T)
    if step_hours <= 0:
        raise ValueError(f"Invalid step_hours={step_hours}")

    # Determine which LAED look-ahead windows to run.
    if args.laed_windows is not None:
        laed_windows = [int(x) for x in _parse_list_floats(args.laed_windows)]
    else:
        laed_windows = [int(args.laed_window)]

    laed_windows = sorted(set(int(x) for x in laed_windows))
    for N_t in laed_windows:
        if not (1 <= int(N_t) <= int(N_T)):
            raise ValueError(f"Invalid LAED window N_t={N_t} for horizon N_T={N_T}.")
        if int(N_t) < 13 and int(_scalar_param(base_case, "N_s", default=0) or 0) > 0:
            print(f"WARNING: LAED window N_t={N_t} is short with storage enabled. Try N_t>=13 for storage.")

    eta_c = float(args.eta_c) if args.eta_c is not None else float(_scalar_param(base_case, "eta_c", default=0.95))
    eta_d = float(args.eta_d) if args.eta_d is not None else float(_scalar_param(base_case, "eta_d", default=0.95))

    solver = SolverFactory(args.solver)
    if solver is None:
        raise RuntimeError(f"Could not create solver '{args.solver}'.")
    if hasattr(solver, "options") and "gurobi" in str(args.solver).lower():
        solver.options["OutputFlag"] = 0

    p_fracs = _parse_list_floats(args.p_fracs)
    if not p_fracs:
        raise ValueError("No p_fracs provided.")
    p_fracs = sorted(set(float(x) for x in p_fracs))

    out_dir = Path(args.out_dir)
    _maybe_makedirs(out_dir)
    _maybe_remove_heatmaps(out_dir)

    # Run sweeps for each average demand.
    p_frac_arr = np.array(p_fracs, dtype=float)
    all_rows: list[dict] = []
    sweeps = []

    for avg_demand in avg_demands:
        avg_demand = float(avg_demand)
        load_scale = avg_demand / (sum(proj.values()) / len(proj))
        Load = {int(k): float(v) * load_scale for k, v in proj.items() if int(k) <= N_T}

        base = copy.deepcopy(base_case)
        base["Load"] = Load
        base["N_T"] = {None: int(N_T)}

        # Initialize Gen_init to a feasible starting dispatch at this load level.
        load_ini = float(base["Load"][1])
        base["Gen_init"] = _initial_dispatch(base["Cost"], base["Capacity"], load_ini)

        base_no_storage = strip_storage(base)

        for N_t in laed_windows:
            lookahead_hr = float(int(N_t) - 1) * float(step_hours)

            # Baseline: no storage (for diff metrics), horizon-dependent.
            P0, Shed0, SoC0, Pch0, Pdis0 = run_laed_dispatch(
                base_no_storage,
                N_g=N_g,
                N_t=int(N_t),
                N_T=N_T,
                load_factor=float(args.load_factor),
                ramp_factor=float(args.ramp_factor),
                sigma_rel=float(args.sigma),
                rho=float(args.rho),
                seed=int(args.seed),
                solver=solver,
            )

            runs = []
            for frac in p_fracs:
                p_mw = float(frac) * float(avg_demand)
                b = _set_battery_params(
                    base_no_storage,
                    p_mw=p_mw,
                    duration_hr=float(args.duration_hr),
                    soc_init_frac=float(args.soc_init_frac),
                    soc_min_frac=float(args.soc_min_frac),
                eta_c=eta_c,
                eta_d=eta_d,
                delta_t=step_hours,
                charge_cost=float(args.storage_cost),
                discharge_cost=float(args.storage_cost),
            )
                P, Shed, SoC, Pch, Pdis = run_laed_dispatch(
                    b,
                    N_g=N_g,
                    N_t=int(N_t),
                    N_T=N_T,
                    load_factor=float(args.load_factor),
                    ramp_factor=float(args.ramp_factor),
                    sigma_rel=float(args.sigma),
                    rho=float(args.rho),
                    seed=int(args.seed),
                    solver=solver,
                )
                metrics = _summarize_run(P=P, Shed=Shed, SoC=SoC, Pch=Pch, Pdis=Pdis, step_hours=step_hours)
                diff = _dispatch_diff_metrics(P, P0)

                e_mwh = p_mw * float(args.duration_hr) if p_mw > 0.0 else 0.0
                row = {
                    "avg_demand_MW": float(avg_demand),
                    "N_t": int(N_t),
                    "lookahead_hr": float(lookahead_hr),
                    "p_frac": float(frac),
                    "p_mw": float(p_mw),
                    "duration_hr": float(args.duration_hr),
                    "e_mwh": float(e_mwh),
                    **metrics,
                    **diff,
                }
                runs.append((row, P, Shed, SoC, Pch, Pdis))
                all_rows.append(row)

            # Save arrays in an .npz bundle for ad-hoc analysis.
            p_mw_arr = np.array([r[0]["p_mw"] for r in runs], dtype=float)
            e_mwh_arr = np.array([r[0]["e_mwh"] for r in runs], dtype=float)
            P_arr = np.stack([r[1] for r in runs], axis=0)  # (n_sizes, N_g, n_steps)
            Shed_arr = np.stack([r[2] for r in runs], axis=0)  # (n_sizes, n_steps)

            # Single-battery sweep: store SoC/Pch/Pdis as (n_sizes, n_steps) with NaN when absent.
            n_sizes = len(runs)
            n_steps = int(P0.shape[1])
            SoC_sweep = np.full((n_sizes, n_steps), np.nan, dtype=float)
            Pch_sweep = np.full((n_sizes, n_steps), np.nan, dtype=float)
            Pdis_sweep = np.full((n_sizes, n_steps), np.nan, dtype=float)
            for i, (_row, _P, _Shed, SoC_i, Pch_i, Pdis_i) in enumerate(runs):
                if SoC_i.size == 0:
                    continue
                # Expect shape (1, n_steps) for the constructed single battery.
                SoC_sweep[i, :] = SoC_i[0, :]
                Pch_sweep[i, :] = Pch_i[0, :]
                Pdis_sweep[i, :] = Pdis_i[0, :]

            tag = f"avg{int(round(avg_demand))}_nt{int(N_t)}"
            npz_path = out_dir / f"dispatch_sweep_{tag}.npz"
            np.savez(
                npz_path,
                avg_demand_MW=np.array([avg_demand], dtype=float),
                N_t=np.array([int(N_t)], dtype=int),
                lookahead_hr=np.array([lookahead_hr], dtype=float),
                p_frac=p_frac_arr,
                p_mw=p_mw_arr,
                e_mwh=e_mwh_arr,
                P=P_arr,
                Shed=Shed_arr,
                SoC=SoC_sweep,
                Pch=Pch_sweep,
                Pdis=Pdis_sweep,
                P_baseline=P0,
                Shed_baseline=Shed0,
                step_hours=np.array([step_hours], dtype=float),
            )

            sweeps.append(
                {
                    "avg_demand_MW": avg_demand,
                    "N_t": int(N_t),
                    "lookahead_hr": float(lookahead_hr),
                    "runs": runs,
                    "P_arr": P_arr,
                    "npz_path": npz_path,
                }
            )

    # Write summary CSV.
    csv_path = out_dir / "summary.csv"
    fieldnames: list[str] = []
    for row in all_rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with csv_path.open("w", newline="") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    # ---- Plots ----
    # Metrics vs battery size
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].set_ylabel("Total load shed (MW)")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Gen2 mean dispatch (MW)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Gen 2 L1-Norm (with & w/o storage)")
    axes[2].set_xlabel("Battery Penetration(fraction of avg demand)")
    axes[2].grid(True, alpha=0.3)

    for s in sweeps:
        avg_demand = float(s["avg_demand_MW"])
        N_t = int(s.get("N_t", -1))
        runs = s["runs"]
        shed_sum = np.array([r[0]["shed_sum"] for r in runs], dtype=float)
        g2_mean = np.array([r[0].get("g2_mean_MW", np.nan) for r in runs], dtype=float)
        g2_L1 = np.array([r[0].get("g2_L1", np.nan) for r in runs], dtype=float)
        label = f"Avg load {avg_demand:.0f} MW, N_t={N_t}"

        axes[0].plot(p_frac_arr, shed_sum, marker="o", label=label)
        axes[1].plot(p_frac_arr, g2_mean, marker="o", label=label)
        axes[2].plot(p_frac_arr, g2_L1, marker="o", label=label)

    axes[0].legend()

    # Make the x-axis show the actual battery sizes tested (instead of "nice" tick spacing).
    def _fmt_frac(x: float) -> str:
        s = f"{x:.3f}"
        s = s.rstrip("0").rstrip(".")
        return s if s else "0"

    xt = [float(x) for x in p_frac_arr]
    axes[2].set_xticks(xt)
    axes[2].set_xticklabels([_fmt_frac(x) for x in xt])

    fig.suptitle("Battery Size Sensitivity (LAED rolling dispatch)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "metrics_vs_size.png", dpi=200)
    if not args.no_show:
        plt.show()
    plt.close(fig)

    print(f"Wrote: {csv_path}")
    for s in sweeps:
        print(f"Wrote: {s['npz_path']}")
    print(f"Wrote: {out_dir / 'metrics_vs_size.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
