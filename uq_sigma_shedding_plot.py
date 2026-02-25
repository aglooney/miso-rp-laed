import argparse
import json
import os
import time

# Set matplotlib env early because laed_rp_random_error imports pyplot at import time.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
# Default to a non-interactive backend for reliability in headless runs.
# Users can override by exporting MPLBACKEND before invoking this script.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
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
    value,
)

import laed_rp_analysis as lra
import laed_rp_random_error as lre


def _parse_csv_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_strings(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def apply_aug_2032_projection(data: DataPortal, projection_path: str, ref_cap: float) -> None:
    """Scale the 2032_Aug demand trajectory in MISO_Projection.json to the requested average load."""
    with open(projection_path, "r") as f:
        interpolated = json.load(f)
    aug_2032_ori = interpolated["2032_Aug"]
    scale = float(ref_cap) / (sum(aug_2032_ori.values()) / len(aug_2032_ori))
    aug_2032 = {int(k): float(v) * scale for k, v in aug_2032_ori.items()}
    data.data()["Load"] = aug_2032
    data.data()["N_T"][None] = len(aug_2032)


def initialize_gen_init(data: DataPortal, solver) -> None:
    """Initialize Gen_init by solving a single-interval ED (or a simple 2-gen heuristic)."""
    base = data.data()
    n_g = int(base["N_g"][None])
    load_ini = float(base["Load"][1])

    if n_g == 2:
        gen1_ini = min(float(base["Capacity"][1]), load_ini)
        base["Gen_init"] = {
            1: gen1_ini,
            2: min(float(base["Capacity"][2]), load_ini - gen1_ini),
        }
        return

    reserve_single = base["reserve_single"] if isinstance(base["reserve_single"], (int, float)) else base["reserve_single"][None]

    model_ini = AbstractModel()
    model_ini.N_g = Param(within=NonNegativeIntegers)
    model_ini.G = RangeSet(1, model_ini.N_g)
    model_ini.Cost = Param(model_ini.G)
    model_ini.Capacity = Param(model_ini.G)
    model_ini.reserve_single = Param()

    model_ini.P = Var(model_ini.G, within=NonNegativeReals)
    model_ini.Reserve = Var(model_ini.G, within=NonNegativeReals)

    def objective_rule(m):
        return sum(m.Cost[g] * m.P[g] for g in m.G)

    model_ini.obj = Objective(rule=objective_rule, sense=minimize)

    def power_balance_rule(m):
        return sum(m.P[g] for g in m.G) == load_ini

    model_ini.power_balance_constraint = Constraint(rule=power_balance_rule)

    def capacity_rule(m, g):
        return m.P[g] + m.Reserve[g] <= m.Capacity[g]

    model_ini.capacity_constraint = Constraint(model_ini.G, rule=capacity_rule)

    def reserve_rule(m):
        return sum(m.Reserve[g] for g in m.G) >= float(lra.reserve_factor) * load_ini

    model_ini.reserve_constraint = Constraint(rule=reserve_rule)

    def reserve_single_rule(m, g):
        return m.Reserve[g] <= m.reserve_single * m.Capacity[g]

    model_ini.reserve_single_constraint = Constraint(model_ini.G, rule=reserve_single_rule)

    ed_ini = model_ini.create_instance(data)
    solver.solve(ed_ini, tee=False)
    base["Gen_init"] = {g: float(ed_ini.P[g].value) for g in ed_ini.G}


def build_scaled_inputs(data: DataPortal, load_factor: float, ramp_factor: float):
    base = data.data()
    ramp_single = base["ramp_single"] if isinstance(base["ramp_single"], (int, float)) else base["ramp_single"][None]
    reserve_single = base["reserve_single"] if isinstance(base["reserve_single"], (int, float)) else base["reserve_single"][None]
    return {
        "N_g": int(base["N_g"][None]),
        "N_T": int(base["N_T"][None]),
        "Cost": {int(g): float(base["Cost"][g]) for g in base["Cost"]},
        "Capacity": {int(g): float(base["Capacity"][g]) for g in base["Capacity"]},
        "Ramp_lim": {int(g): float(base["Ramp_lim"][g]) * float(ramp_factor) for g in base["Ramp_lim"]},
        "Load_true": {int(t): float(base["Load"][t]) * float(load_factor) for t in base["Load"]},
        "Gen_init": {int(g): float(base["Gen_init"][g]) * float(load_factor) for g in base["Gen_init"]},
        "ramp_single": float(ramp_single),
        "reserve_single": float(reserve_single),
    }


def build_forecast_load_window(
    *,
    load_true,
    err_state,
    t0: int,
    horizon: int,
    sigma_rel: float,
    rho: float,
    rng: np.random.Generator,
    error_type: str,
    error_kwargs: dict,
    clip_nonnegative: bool = True,
):
    """
    Match laed_rp_random_error.py forecast-error process:
    - lead h=0 is perfectly observed (error=0)
    - Std grows as sigma_rel * |Load_true| * sqrt(h)
    - correlated revisions across rolling windows via persistent err_state by physical time index
    """
    load_window = {}
    err_state[t0] = 0.0

    for tt in range(1, int(horizon) + 1):
        t_phys = int(t0) + tt - 1
        h = tt - 1
        if h == 0:
            e = 0.0
            err_state[t_phys] = 0.0
        else:
            std_h = float(sigma_rel) * abs(float(load_true[t_phys])) * np.sqrt(h)
            e = lre.correlated_error_update(
                prev_err=float(err_state[t_phys]),
                std_h=std_h,
                rho=float(rho),
                rng=rng,
                error_type=error_type,
                **error_kwargs,
            )
            err_state[t_phys] = float(e)

        Lhat = float(load_true[t_phys]) + float(e)
        if clip_nonnegative:
            Lhat = max(0.0, Lhat)
        load_window[tt] = float(Lhat)

    return load_window


def build_forecast_load_window_simple_gaussian(
    *,
    load_true,
    t0: int,
    horizon: int,
    sigma_rel: float,
    rng: np.random.Generator,
    clip_nonnegative: bool = True,
):
    """
    Very simple forecast-error model:
    - lead h=0 is perfectly observed (error=0)
    - for h>0, draw iid Gaussian errors with Std = sigma_rel * |Load_true[t_phys]|
    - no correlation across windows and no lead-time scaling
    """
    load_window = {}
    for tt in range(1, int(horizon) + 1):
        t_phys = int(t0) + tt - 1
        h = tt - 1
        if h == 0:
            e = 0.0
        else:
            std = float(sigma_rel) * abs(float(load_true[t_phys]))
            e = float(std * rng.standard_normal())

        Lhat = float(load_true[t_phys]) + float(e)
        if clip_nonnegative:
            Lhat = max(0.0, Lhat)
        load_window[tt] = float(Lhat)
    return load_window


def solve_rp_window_shed_and_dispatch(*, scaled, rp_n_t: int, gen_prev, load_window, solver):
    model = lra.rped_opt_model()
    inst_data = {
        None: {
            "N_g": {None: scaled["N_g"]},
            "N_t": {None: int(rp_n_t)},
            "Cost": scaled["Cost"],
            "Capacity": scaled["Capacity"],
            "Ramp_lim": scaled["Ramp_lim"],
            "Gen_prev": gen_prev,
            "Load": load_window,
            "ramp_single": {None: scaled["ramp_single"]},
            "reserve_single": {None: scaled["reserve_single"]},
        }
    }
    inst = model.create_instance(data=inst_data)
    res = solver.solve(inst, tee=False, load_solutions=False)
    term = str(res.solver.termination_condition).lower()
    status = str(res.solver.status).lower()
    if term != "optimal" or status != "ok":
        raise RuntimeError(f"RP infeasible/unknown: status={status}, term={term}")
    inst.solutions.load_from(res)

    dispatch = {g: float(value(inst.P[g])) for g in range(1, scaled["N_g"] + 1)}
    shed_commit = float(value(inst.Loadshed[1]))
    return dispatch, shed_commit


def solve_laed_window_shed_and_dispatch(*, scaled, laed_n_t: int, gen_init_current, load_window, solver):
    model = lra.laed_opt_model()
    inst_data = {
        None: {
            "N_g": {None: scaled["N_g"]},
            "N_t": {None: int(laed_n_t)},
            "N_T": {None: scaled["N_T"]},
            "Cost": scaled["Cost"],
            "Capacity": scaled["Capacity"],
            "Ramp_lim": scaled["Ramp_lim"],
            "Load": load_window,
            "Gen_init": gen_init_current,
            "reserve_single": {None: scaled["reserve_single"]},
        }
    }
    inst = model.create_instance(data=inst_data)
    res = solver.solve(inst, tee=False, load_solutions=False)
    term = str(res.solver.termination_condition).lower()
    status = str(res.solver.status).lower()
    if term != "optimal" or status != "ok":
        raise RuntimeError(f"LAED infeasible/unknown: status={status}, term={term}")
    inst.solutions.load_from(res)

    dispatch = {g: float(value(inst.P[g, 1])) for g in range(1, scaled["N_g"] + 1)}
    shed_commit = float(value(inst.Loadshed[1]))
    return dispatch, shed_commit

def simulate_committed_shedding(
    *,
    scaled,
    laed_n_t: int,
    rp_n_t: int,
    n_steps: int,
    solver,
    sigma_rel: float,
    rho: float,
    seed: int,
    error_type: str,
    nu: float,
    forecast_model: str,
):
    forecast_model = str(forecast_model).strip().lower()
    if forecast_model not in ("correlated", "simple_gaussian"):
        raise ValueError("forecast_model must be 'correlated' or 'simple_gaussian'")

    # Each rolling optimization has its own forecast process tied to its look-ahead horizon.
    rng_rp = np.random.default_rng(int(seed))
    rng_laed = np.random.default_rng(int(seed))
    et = error_type.lower()
    err_kwargs = {"nu": float(nu)} if et in ("student-t", "student_t", "t") else {}

    # Persistent error state per physical time index (1..N_T) (only used for correlated model)
    err_state_rp = {t: 0.0 for t in range(1, int(scaled["N_T"]) + 1)}
    err_state_laed = {t: 0.0 for t in range(1, int(scaled["N_T"]) + 1)}

    demand = np.array([float(scaled["Load_true"][t]) for t in range(1, n_steps + 1)], dtype=float)
    rp_shed = np.full(n_steps, np.nan, dtype=float)
    laed_shed = np.full(n_steps, np.nan, dtype=float)

    gen_prev_rp = dict(scaled["Gen_init"])
    gen_prev_laed = dict(scaled["Gen_init"])

    rp_fail = 0
    laed_fail = 0

    for k in range(n_steps):
        t0 = k + 1

        if forecast_model == "correlated":
            load_rp = build_forecast_load_window(
                load_true=scaled["Load_true"],
                err_state=err_state_rp,
                t0=t0,
                horizon=int(rp_n_t),
                sigma_rel=float(sigma_rel),
                rho=float(rho),
                rng=rng_rp,
                error_type=error_type,
                error_kwargs=err_kwargs,
                clip_nonnegative=True,
            )
        else:
            load_rp = build_forecast_load_window_simple_gaussian(
                load_true=scaled["Load_true"],
                t0=t0,
                horizon=int(rp_n_t),
                sigma_rel=float(sigma_rel),
                rng=rng_rp,
                clip_nonnegative=True,
            )

        try:
            dispatch_rp, shed_rp_k = solve_rp_window_shed_and_dispatch(
                scaled=scaled, rp_n_t=int(rp_n_t), gen_prev=gen_prev_rp, load_window=load_rp, solver=solver
            )
            rp_shed[k] = float(shed_rp_k)
            gen_prev_rp = dispatch_rp
        except Exception:
            rp_fail += 1
            # Keep prior dispatch as the best available warm start for the next step.

        try:
            if forecast_model == "correlated":
                load_laed = build_forecast_load_window(
                    load_true=scaled["Load_true"],
                    err_state=err_state_laed,
                    t0=t0,
                    horizon=int(laed_n_t),
                    sigma_rel=float(sigma_rel),
                    rho=float(rho),
                    rng=rng_laed,
                    error_type=error_type,
                    error_kwargs=err_kwargs,
                    clip_nonnegative=True,
                )
            else:
                load_laed = build_forecast_load_window_simple_gaussian(
                    load_true=scaled["Load_true"],
                    t0=t0,
                    horizon=int(laed_n_t),
                    sigma_rel=float(sigma_rel),
                    rng=rng_laed,
                    clip_nonnegative=True,
                )
            dispatch_laed, shed_laed_k = solve_laed_window_shed_and_dispatch(
                scaled=scaled,
                laed_n_t=int(laed_n_t),
                gen_init_current=gen_prev_laed,
                load_window=load_laed,
                solver=solver,
            )
            laed_shed[k] = float(shed_laed_k)
            gen_prev_laed = dispatch_laed
        except Exception:
            laed_fail += 1

    return demand, rp_shed, laed_shed, rp_fail, laed_fail


def main():
    ap = argparse.ArgumentParser(
        description="Plot demand trajectory + committed load shedding across sigma levels (first-look UQ)."
    )
    ap.add_argument("--case", default="toy_data.dat")
    ap.add_argument("--projection", default="MISO_Projection.json")
    ap.add_argument("--ref-cap", type=float, default=350.0)
    ap.add_argument("--n-t", type=int, default=13, help="LAED rolling look-ahead window length.")
    ap.add_argument("--rp-n-t", type=int, default=2, help="RPED horizon length.")
    ap.add_argument("--ramp-factor", type=float, default=0.1, help="Ramp limit multiplier.")
    ap.add_argument("--load-factor", type=float, default=1.0, help="Load multiplier.")
    ap.add_argument("--cost-load", type=float, default=1e10, help="Load shedding penalty.")
    ap.add_argument("--sigmas", default="0.0,0.002,0.005,0.01,0.05", help="Comma-separated sigma values.")
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=109)
    ap.add_argument(
        "--seeds",
        default="42,43,44,45,46",
        help="Comma-separated seeds. If provided, overrides --seed and averages results across seeds.",
    )
    ap.add_argument("--error-type", default="gaussian", choices=["gaussian", "laplace", "student-t"])
    ap.add_argument("--nu", type=float, default=4.0, help="Student-t dof (only used if --error-type=student-t).")
    ap.add_argument("--df", dest="nu", type=float, help=argparse.SUPPRESS)  # backwards compatible alias
    ap.add_argument(
        "--forecast-models",
        default="correlated,simple_gaussian",
        help="Comma-separated forecast models: correlated, simple_gaussian",
    )
    ap.add_argument("--max-steps", type=int, default=0, help="Limit number of committed intervals (0 = full).")
    ap.add_argument("--out", default="uq_sigma_shedding.png")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    sigmas = _parse_csv_floats(args.sigmas)
    if not sigmas:
        raise SystemExit("No sigmas provided.")
    seeds = _parse_csv_ints(args.seeds) if str(args.seeds).strip() else [int(args.seed)]
    if not seeds:
        raise SystemExit("No seeds provided.")
    forecast_models = [m.lower() for m in _parse_csv_strings(args.forecast_models)]
    if not forecast_models:
        raise SystemExit("No forecast models provided.")
    allowed_models = {"correlated", "simple_gaussian"}
    for m in forecast_models:
        if m not in allowed_models:
            raise SystemExit(f"Unknown forecast model '{m}'. Use: correlated, simple_gaussian")

    solver = SolverFactory("gurobi_direct")
    solver.options["OutputFlag"] = 0
    lra.reserve_factor = 0.0
    lra.cost_load = float(args.cost_load)

    data = DataPortal()
    data.load(filename=args.case)
    apply_aug_2032_projection(data, args.projection, args.ref_cap)
    initialize_gen_init(data, solver)

    scaled = build_scaled_inputs(data, args.load_factor, args.ramp_factor)
    n_steps_full = int(scaled["N_T"]) - int(args.n_t) + 1
    n_steps = int(args.max_steps) if int(args.max_steps) > 0 else n_steps_full
    n_steps = max(1, min(n_steps, n_steps_full))
    x = np.arange(1, n_steps + 1)

    demand = None
    rp_shed_by_model = {fm: {} for fm in forecast_models}
    laed_shed_by_model = {fm: {} for fm in forecast_models}
    failures = []  # (model, sigma, rp_fail_steps_total, laed_fail_steps_total)

    t_start = time.perf_counter()
    for fm in forecast_models:
        for s in sigmas:
            rp_runs = []
            laed_runs = []
            rp_fail_tot = 0
            laed_fail_tot = 0
            for seed in seeds:
                d, rp_s, laed_s, rp_fail, laed_fail = simulate_committed_shedding(
                    scaled=scaled,
                    laed_n_t=int(args.n_t),
                    rp_n_t=int(args.rp_n_t),
                    n_steps=n_steps,
                    solver=solver,
                    sigma_rel=float(s),
                    rho=float(args.rho),
                    seed=int(seed),
                    error_type=str(args.error_type),
                    nu=float(args.nu),
                    forecast_model=fm,
                )
                if demand is None:
                    demand = d
                rp_runs.append(rp_s)
                laed_runs.append(laed_s)
                rp_fail_tot += int(rp_fail)
                laed_fail_tot += int(laed_fail)

            rp_mat = np.stack(rp_runs, axis=0)  # (n_seeds, n_steps)
            laed_mat = np.stack(laed_runs, axis=0)
            rp_mean = np.nanmean(rp_mat, axis=0)
            laed_mean = np.nanmean(laed_mat, axis=0)

            rp_shed_by_model[fm][s] = rp_mean
            laed_shed_by_model[fm][s] = laed_mean
            failures.append((fm, s, rp_fail_tot, laed_fail_tot))

    runtime = time.perf_counter() - t_start

    import matplotlib.pyplot as plt

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sigmas)))

    ncols = len(forecast_models)
    fig, axes = plt.subplots(3, ncols, figsize=(6 * ncols + 6, 9), sharex=True)
    axes = np.asarray(axes)
    if ncols == 1:
        axes = axes.reshape(3, 1)

    # Global y-limit across ALL shedding panels (all models, all sigmas).
    shed_ymax = 0.0
    for fm in forecast_models:
        for s in sigmas:
            for arr in (laed_shed_by_model[fm][s], rp_shed_by_model[fm][s]):
                a = np.asarray(arr, dtype=float).ravel()
                a = a[np.isfinite(a)]
                if a.size:
                    shed_ymax = max(shed_ymax, float(np.max(a)))
    shed_ymax = 1.0 if shed_ymax <= 0.0 else 1.05 * shed_ymax

    demand_handle = None
    for j, fm in enumerate(forecast_models):
        ax0 = axes[0, j]
        ax1 = axes[1, j]
        ax2 = axes[2, j]

        if j == 0:
            (demand_handle,) = ax0.plot(x, demand, color="black", linewidth=2.0, label="Demand (realized)")
        else:
            ax0.plot(x, demand, color="black", linewidth=2.0)
        ax0.set_title(f"Forecast model: {fm}")
        ax0.grid(True, alpha=0.25)

        for c, s in zip(colors, sigmas):
            ax1.plot(x, laed_shed_by_model[fm][s], color=c, linewidth=1.8, label=f"sigma={s:g}")
        ax1.grid(True, alpha=0.25)
        ax1.set_ylim(0.0, shed_ymax)

        for c, s in zip(colors, sigmas):
            ax2.plot(x, rp_shed_by_model[fm][s], color=c, linewidth=1.8, label=f"sigma={s:g}")
        ax2.grid(True, alpha=0.25)
        ax2.set_ylim(0.0, shed_ymax)
        ax2.set_xlabel("Time (5-min intervals)")

        if j == 0:
            ax0.set_ylabel("Demand (MW)")
            ax1.set_ylabel("LAED load shed (MW)")
            ax2.set_ylabel("RP load shed (MW)")

    # Shared legend (demand + sigma lines) from first column
    sigma_handles, sigma_labels = axes[1, 0].get_legend_handles_labels()
    if demand_handle is not None:
        handles = [demand_handle] + sigma_handles
        labels = ["Demand (realized)"] + sigma_labels
    else:
        handles = sigma_handles
        labels = sigma_labels
    fig.legend(handles, labels, ncol=min(len(labels), 5), fontsize=9, loc="upper right")

    title_bits = [f"avg over {len(seeds)} seeds", f"laed_n_t={args.n_t}", f"rp_n_t={args.rp_n_t}"]
    if "correlated" in forecast_models:
        title_bits.append(f"rho={args.rho:g}")
        title_bits.append(f"innovation={args.error_type}")
    fig.suptitle(f"Committed Load Shedding vs Sigma ({'; '.join(title_bits)})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

    print(
        f"Runtime: {runtime:.2f}s for {len(forecast_models)} models x {len(sigmas)} sigmas x {len(seeds)} seeds x {n_steps} steps"
    )
    print(f"Seeds: {','.join(str(x) for x in seeds)}")
    print("Forecast models: " + ",".join(forecast_models))
    print(
        "model, sigma, rp_fail_steps_total, laed_fail_steps_total, rp_shed_sum_mean, laed_shed_sum_mean, rp_shed_max_mean, laed_shed_max_mean"
    )
    for fm, s, rp_fail_tot, laed_fail_tot in failures:
        rp_s = rp_shed_by_model[fm][s]
        laed_s = laed_shed_by_model[fm][s]
        rp_sum = float(np.nansum(rp_s))
        laed_sum = float(np.nansum(laed_s))
        rp_max = float(np.nanmax(rp_s)) if np.isfinite(np.nanmax(rp_s)) else float("nan")
        laed_max = float(np.nanmax(laed_s)) if np.isfinite(np.nanmax(laed_s)) else float("nan")
        print(f"{fm}, {s:g}, {rp_fail_tot}, {laed_fail_tot}, {rp_sum:.6f}, {laed_sum:.6f}, {rp_max:.6f}, {laed_max:.6f}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
