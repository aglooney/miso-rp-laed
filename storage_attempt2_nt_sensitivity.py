import os
import tempfile

# Force a non-interactive backend so this script can run headless (e.g. in CI / terminals).
# Users can override by setting MPLBACKEND externally.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import (
    AbstractModel,
    Constraint,
    DataPortal,
    NonNegativeIntegers,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    SolverFactory,
    Var,
    minimize,
)

import storage_attempt2 as sa2


def _parse_int_list(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _load_scaled_projection(*, projection_file: str, projection_key: str, avg_demand_mw: float) -> dict[int, float]:
    with open(projection_file, "r") as f:
        interpolated_data = json.load(f)
    if projection_key not in interpolated_data:
        raise KeyError(f"projection_key '{projection_key}' not in {projection_file}")

    series = interpolated_data[projection_key]
    # JSON keys may be strings; normalize to int keys.
    series_int = {int(k): float(v) for k, v in series.items()}
    mean = float(sum(series_int.values()) / len(series_int))
    if mean <= 0.0:
        raise ValueError("Projection mean demand must be positive")
    scale = float(avg_demand_mw) / mean
    return {k: v * scale for k, v in series_int.items()}


def _ensure_gen_init(data: DataPortal, solver) -> None:
    """
    Match the initialization logic in storage_attempt2.py:
      - For N_g==2: set Gen_init to cover Load[1] with Gen1 then Gen2.
      - Else: solve a 1-period ED to set Gen_init.
    """
    base = data.data()
    N_g = int(base["N_g"][None])

    if N_g == 2:
        load_1 = float(base["Load"][1])
        cap = base["Capacity"]
        g1 = min(float(cap[1]), load_1)
        g2 = min(float(cap[2]), max(0.0, load_1 - g1))
        base["Gen_init"] = {1: g1, 2: g2}
        return

    load_1 = float(base["Load"][1])
    model_ini = AbstractModel()
    model_ini.N_g = Param(within=NonNegativeIntegers)  # Number of Generators
    model_ini.G = RangeSet(1, model_ini.N_g)
    model_ini.Cost = Param(model_ini.G)
    model_ini.Capacity = Param(model_ini.G)
    model_ini.reserve_single = Param()

    model_ini.P = Var(model_ini.G, within=NonNegativeReals)
    model_ini.Reserve = Var(model_ini.G, within=NonNegativeReals)

    def _obj(m):
        return sum(m.Cost[g] * m.P[g] for g in m.G)

    model_ini.obj = Objective(rule=_obj, sense=minimize)

    def _balance(m):
        return sum(m.P[g] for g in m.G) == load_1

    model_ini.power_balance_constraint = Constraint(rule=_balance)

    def _cap(m, g):
        return m.P[g] + m.Reserve[g] <= m.Capacity[g]

    model_ini.capacity_constraint = Constraint(model_ini.G, rule=_cap)

    def _reserve(m):
        return sum(m.Reserve[g] for g in m.G) >= sa2.reserve_factor * load_1

    model_ini.reserve_constraint = Constraint(rule=_reserve)

    def _reserve_single(m, g):
        return m.Reserve[g] <= m.reserve_single * m.Capacity[g]

    model_ini.reserve_single_constraint = Constraint(model_ini.G, rule=_reserve_single)

    inst = model_ini.create_instance(data)
    solver.solve(inst, tee=False)
    base["Gen_init"] = {g: float(inst.P[g].value) for g in inst.G}


def _total_shed_mwh(shed_mw: np.ndarray, step_hours: float) -> float:
    return float(np.sum(np.asarray(shed_mw, dtype=float)) * float(step_hours))


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep LAED look-ahead window (N_t) for storage_attempt2 and plot total load shed.")
    ap.add_argument("--case", default="toy_data_storage.dat", help="Pyomo .dat case file")
    ap.add_argument("--solver", default="gurobi_direct", help="Pyomo solver name (default: gurobi_direct)")
    ap.add_argument("--nts", default="5,7,9,11,13,15,17,19,21,23,25", help="Comma-separated LAED window sizes (N_t) to test, e.g. 5,13,25")

    ap.add_argument("--projection-file", default="MISO_Projection.json", help="Demand projection JSON file")
    ap.add_argument("--projection-key", default="2032_Aug", help="Key inside projection JSON (default: 2032_Aug)")
    ap.add_argument("--avg-demand", type=float, default=350.0, help="Scale projection to this average demand (MW)")
    ap.add_argument("--n-intervals", type=int, default=0, help="If >0, truncate to the first N intervals (for quick tests)")

    ap.add_argument("--load-factor", type=float, default=1.0)
    ap.add_argument("--ramp-factor", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=0.0, help="Forecast error sigma_rel (0 => no forecast error)")
    ap.add_argument("--rho", type=float, default=0.95, help="Forecast error correlation rho")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--cost-load", type=float, default=1e10, help="Load-shed penalty ($/MWh)")
    ap.add_argument("--reserve-factor", type=float, default=0.0)
    ap.add_argument("--out", default="nt_vs_loadshed_storage_attempt2.png", help="Output plot path")
    ap.add_argument("--no-show", action="store_true", help="Do not show plot window")
    args = ap.parse_args()

    # Reduce noise when solving many MILPs (duals unavailable for MIP).
    logging.getLogger("pyomo.solvers").setLevel(logging.ERROR)

    # Set globals used by storage_attempt2's model definitions.
    sa2.cost_load = float(args.cost_load)
    sa2.reserve_factor = float(args.reserve_factor)

    nts = _parse_int_list(args.nts)
    if not nts:
        raise SystemExit("No N_t values provided via --nts")
    if any(n <= 0 for n in nts):
        raise SystemExit("All N_t must be positive integers")

    # Load case data.
    data = DataPortal()
    data.load(filename=args.case)

    # Replace load with scaled projection and ensure delta_t is 5 minutes.
    load_proj = _load_scaled_projection(
        projection_file=args.projection_file,
        projection_key=args.projection_key,
        avg_demand_mw=float(args.avg_demand),
    )
    if int(args.n_intervals) > 0:
        n_int = int(args.n_intervals)
        load_proj = {k: load_proj[k] for k in range(1, n_int + 1)}
    data.data()["Load"] = load_proj
    data.data()["delta_t"][None] = 1.0 / 12.0

    N_g = int(data.data()["N_g"][None])
    N_T = len(load_proj)
    if any(int(n_t) > N_T for n_t in nts):
        raise SystemExit(f"All N_t must be <= N_T={N_T}. Got --nts={args.nts!r}")

    solver = SolverFactory(args.solver)
    # Common gurobi_direct option; harmless for other solvers.
    try:
        solver.options["OutputFlag"] = 0
    except Exception:
        pass

    _ensure_gen_init(data, solver)

    step_hours = float(data.data()["delta_t"][None])

    # Compute a single RP baseline (10-minute ramp product; N_t=2).
    rng_rp = np.random.default_rng(int(args.seed))
    _P_rp, shed_rp, *_rest = sa2.ED_with_error(
        copy.deepcopy(data),
        N_g,
        2,
        N_T,
        float(args.load_factor),
        float(args.ramp_factor),
        solver,
        float(args.sigma),
        float(args.rho),
        rng_rp,
        error_type="gaussian",
    )
    rp_total_mwh = _total_shed_mwh(shed_rp, step_hours)

    laed_totals_mwh: list[float] = []
    for n_t in nts:
        rng_laed = np.random.default_rng(int(args.seed))
        _P_laed, shed_laed, *_rest = sa2.LAED_with_error(
            copy.deepcopy(data),
            N_g,
            int(n_t),
            N_T,
            float(args.load_factor),
            float(args.ramp_factor),
            solver,
            float(args.sigma),
            float(args.rho),
            rng_laed,
            error_type="gaussian",
        )
        laed_totals_mwh.append(_total_shed_mwh(shed_laed, step_hours))

    # ---- Plot ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(nts, laed_totals_mwh, marker="o", label="LAED total load shed")
    ax.axhline(rp_total_mwh, linestyle="--", color="k", alpha=0.6, label="RP total load shed (N_t=2)")
    ax.set_xlabel("LAED look-ahead window length (N_t, 5-min intervals)")
    ax.set_ylabel("Total load shed (MWh)")
    ax.set_title("Storage Attempt 2: Total Load Shed vs LAED Look-Ahead Window")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(sorted(set(int(x) for x in nts)))
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if not args.no_show:
        plt.show()
    plt.close(fig)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
