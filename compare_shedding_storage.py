#!/usr/bin/env python3
"""
Compare load shedding with vs without storage (Gaussian load forecast error).

Runs both:
  - RP/ED (10-minute ramp product) via ED_with_error (forces N_t=2)
  - LAED via LAED_with_error (user-chosen N_t)

For a fair comparison, the with-storage and no-storage runs use the same RNG seed
within each formulation.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json

import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import DataPortal, SolverFactory

import laed_rp_with_storage as lrs


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
    """Return a deep-copied base dict with all storage params removed."""
    b = copy.deepcopy(base)
    for k in STORAGE_KEYS:
        b.pop(k, None)
    return b


def _initial_dispatch(cost: dict, capacity: dict, load: float) -> dict:
    """
    Simple economic dispatch for linear costs:
    fill cheapest generators up to capacity until meeting load.

    This is used only to set Gen_init so the first rolling window is feasible
    under ramp constraints.
    """
    load = float(load)
    if load < 0:
        raise ValueError("Load must be nonnegative for initialization.")

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


@dataclass
class DictData:
    """Minimal shim so ED_with_error/LAED_with_error can accept a plain dict."""

    _base: dict

    def data(self) -> dict:
        return self._base


def solve_rp_shedding(
    base: dict,
    *,
    N_g: int,
    N_T: int,
    load_factor: float,
    ramp_factor: float,
    sigma_rel: float,
    rho: float,
    seed: int,
    solver,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    _, Shed, *_ = lrs.ED_with_error(
        DictData(base),
        N_g,
        2,  # RP/ED model is implemented as a 10-minute ramp product
        N_T,
        load_factor,
        ramp_factor,
        solver,
        sigma_rel,
        rho,
        rng,
        error_type="gaussian",
    )
    return np.asarray(Shed, dtype=float)


def solve_laed_shedding(
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
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    _, Shed, *_ = lrs.LAED_with_error(
        DictData(base),
        N_g,
        N_t,
        N_T,
        load_factor,
        ramp_factor,
        solver,
        sigma_rel,
        rho,
        rng,
        error_type="gaussian",
    )
    return np.asarray(Shed, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="toy_data_storage.dat", help="Pyomo .dat case file")
    ap.add_argument("--solver", default="gurobi_direct", help="Pyomo solver name (e.g. gurobi_direct)")
    # MISO RTD/LAED is typically modeled with ~1 hour look-ahead at 5-minute resolution
    # => N_t ~= 13 (current interval + 12 future intervals). Shorter windows can make
    # storage look worse in a receding-horizon simulation because the optimizer will
    # use storage to defer slow-ramping generation, then run out of energy when the
    # peak is just beyond the look-ahead boundary.
    ap.add_argument("--laed-window", type=int, default=13, help="LAED window size N_t (default: 13)")
    ap.add_argument("--horizon", type=int, default=None, help="Physical horizon N_T (defaults to case param N_T)")
    ap.add_argument("--projection-file", default="MISO_Projection.json", help="JSON file with demand projection")
    ap.add_argument("--projection-key", default="2032_Aug", help="Top-level key in projection JSON (e.g. 2032_Aug)")
    ap.add_argument("--avg-demand", type=float, default=350.0, help="Scale projection to this average demand")
    ap.add_argument("--load-factor", type=float, default=1.0)
    ap.add_argument("--ramp-factor", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=0.05, help="Relative forecast error scale (Gaussian)")
    ap.add_argument("--rho", type=float, default=0.99, help="Inter-window correlation parameter in [0,1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost-load", type=float, default=1e10, help="Penalty for load shedding (default: 1e10)")
    ap.add_argument("--no-show", action="store_true", help="Do not show plots")
    ap.add_argument("--out", default=None, help="If set, save plot to this path")
    args = ap.parse_args()

    # Ensure the objective uses the intended load-shedding penalty.
    lrs.cost_load = float(args.cost_load)

    data = DataPortal()
    data.load(filename=args.case)
    base = data.data()

    # Override demand and horizon length using the projection file.
    # This intentionally ignores any N_T / Load time series embedded in the .dat.
    file_path = args.projection_file
    with open(file_path, "r") as f:
        interpolated_data = json.load(f)

    avg_demand = float(args.avg_demand)

    Aug_2032_ori = interpolated_data[args.projection_key]
    load_scale_2032 = avg_demand / (sum(Aug_2032_ori.values()) / len(Aug_2032_ori))
    Aug_2032 = {int(key): float(value) * load_scale_2032 for key, value in Aug_2032_ori.items()}

    base["Load"] = Aug_2032
    base["N_T"] = {None: len(Aug_2032_ori)}
    # Projection is 288 points at 5-minute resolution (24h); ensure storage SoC dynamics use correct dt (hours).
    if int(len(Aug_2032_ori)) == 288:
        base["delta_t"] = {None: 1.0 / 12.0}

    N_g = int(_scalar_param(base, "N_g"))
    if N_g <= 0:
        raise ValueError(f"Invalid N_g={N_g}")

    # IMPORTANT: initialize Gen_init to a feasible starting dispatch. Many of the
    # .dat files set Gen_init = 0, and both RP and LAED impose ramp constraints
    # from Gen_init/Gen_prev at the first commit interval; that will otherwise
    # force load shedding at k=1 even when there is plenty of capacity.
    if "Cost" not in base or "Capacity" not in base:
        raise ValueError("Case file must provide Cost and Capacity to initialize Gen_init.")
    load_ini = float(base["Load"][1])
    gen_init = _initial_dispatch(base["Cost"], base["Capacity"], load_ini)
    base["Gen_init"] = gen_init

    N_T_projection = len(Aug_2032_ori)
    N_T = int(args.horizon) if args.horizon is not None else int(N_T_projection)

    if "Load" not in base or len(base["Load"]) < N_T:
        raise ValueError(f"Case has only {len(base.get('Load', {}))} Load points but horizon N_T={N_T}.")

    N_t_case = _scalar_param(base, "N_t", default=None)
    N_t_laed = int(args.laed_window) if args.laed_window is not None else int(N_t_case)
    if not (1 <= N_t_laed <= N_T):
        raise ValueError(f"Invalid LAED N_t={N_t_laed} for horizon N_T={N_T}.")
    if N_t_laed < 13 and _scalar_param(base, "N_s", default=0) > 0:
        print(f"WARNING: LAED window N_t={N_t_laed} is short with storage enabled. Try --laed-window 13.")

    solver = SolverFactory(args.solver)
    if solver is None:
        raise RuntimeError(f"Could not create solver '{args.solver}'.")
    if hasattr(solver, "options") and "gurobi" in str(args.solver).lower():
        solver.options["OutputFlag"] = 0

    base_no_storage = strip_storage(base)

    # Solve RP with and without storage (same seed => same forecast error path).
    Shed_RP_with = solve_rp_shedding(
        base,
        N_g=N_g,
        N_T=N_T,
        load_factor=float(args.load_factor),
        ramp_factor=float(args.ramp_factor),
        sigma_rel=float(args.sigma),
        rho=float(args.rho),
        seed=int(args.seed),
        solver=solver,
    )
    Shed_RP_without = solve_rp_shedding(
        base_no_storage,
        N_g=N_g,
        N_T=N_T,
        load_factor=float(args.load_factor),
        ramp_factor=float(args.ramp_factor),
        sigma_rel=float(args.sigma),
        rho=float(args.rho),
        seed=int(args.seed),
        solver=solver,
    )

    # Solve LAED with and without storage (same seed => same forecast error path).
    Shed_LAED_with = solve_laed_shedding(
        base,
        N_g=N_g,
        N_t=N_t_laed,
        N_T=N_T,
        load_factor=float(args.load_factor),
        ramp_factor=float(args.ramp_factor),
        sigma_rel=float(args.sigma),
        rho=float(args.rho),
        seed=int(args.seed),
        solver=solver,
    )
    Shed_LAED_without = solve_laed_shedding(
        base_no_storage,
        N_g=N_g,
        N_t=N_t_laed,
        N_T=N_T,
        load_factor=float(args.load_factor),
        ramp_factor=float(args.ramp_factor),
        sigma_rel=float(args.sigma),
        rho=float(args.rho),
        seed=int(args.seed),
        solver=solver,
    )

    def _summ(name: str, arr: np.ndarray) -> str:
        return f"{name}: sum={float(np.sum(arr)):.6g}, mean={float(np.mean(arr)):.6g}, max={float(np.max(arr)):.6g}"

    print("Load shedding comparison (Gaussian error)")
    print(f"case={args.case}, solver={args.solver}, N_T={N_T}, N_t_LAED={N_t_laed}, seed={args.seed}")
    print(f"load_factor={args.load_factor}, ramp_factor={args.ramp_factor}, sigma={args.sigma}, rho={args.rho}, cost_load={args.cost_load}")
    print("")
    print(_summ("RP with storage", Shed_RP_with))
    print(_summ("RP w/o storage", Shed_RP_without))
    print(_summ("LAED with storage", Shed_LAED_with))
    print(_summ("LAED w/o storage", Shed_LAED_without))
    print("")
    print(f"RP delta (w/o - with): {float(np.sum(Shed_RP_without) - np.sum(Shed_RP_with)):.6g}")
    print(f"LAED delta (w/o - with): {float(np.sum(Shed_LAED_without) - np.sum(Shed_LAED_with)):.6g}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    t_rp = np.arange(1, Shed_RP_with.size + 1)
    axes[0].plot(t_rp, Shed_RP_with, label="RP with storage")
    axes[0].plot(t_rp, Shed_RP_without, label="RP w/o storage", linestyle="--")
    axes[0].set_title("RP/ED (N_t=2) load shedding")
    axes[0].set_xlabel("Rolling step k")
    axes[0].set_ylabel("Loadshed at commit interval")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    t_laed = np.arange(1, Shed_LAED_with.size + 1)
    axes[1].plot(t_laed, Shed_LAED_with, label="LAED with storage")
    axes[1].plot(t_laed, Shed_LAED_without, label="LAED w/o storage", linestyle="--")
    axes[1].set_title(f"LAED (N_t={N_t_laed}) load shedding")
    axes[1].set_xlabel("Rolling step k")
    axes[1].set_ylabel("Loadshed at commit interval")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=200)
        print(f"\nSaved plot to: {args.out}")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
