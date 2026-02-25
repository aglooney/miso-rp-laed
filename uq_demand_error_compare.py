import argparse
import csv
import os
import time
from dataclasses import dataclass

# Set matplotlib env early because laed_rp_analysis imports pyplot at import time.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from pyomo.environ import DataPortal, SolverFactory

import laed_rp_analysis as lra
import price_vol_compare as pvc


def _parse_csv_list(s: str, cast):
    if s is None:
        return []
    parts = [p.strip() for p in s.split(",")]
    out = []
    for p in parts:
        if not p:
            continue
        out.append(cast(p))
    return out


def _error_label(error_type: str, df: float) -> str:
    et = error_type.lower()
    if et == "student-t":
        return f"student-t(df={df:g})"
    return et


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class RunSpec:
    error_type: str
    sigma: float
    rho: float
    seed: int
    df: float


def run_one(
    *,
    scaled,
    laed_n_t: int,
    rp_n_t: int,
    n_steps: int,
    solver,
    spec: RunSpec,
):
    t0 = time.perf_counter()
    err_kwargs = {"df": float(spec.df)} if spec.error_type.lower() == "student-t" else {}

    rp_lmp, laed_lmp, laed_tlmp, pmp_price, cmp_price, diag = pvc.solve_prices_with_forecast_error(
        scaled=scaled,
        laed_n_t=int(laed_n_t),
        rp_n_t=int(rp_n_t),
        n_steps=int(n_steps),
        solver=solver,
        sigma_rel=float(spec.sigma),
        rho=float(spec.rho),
        seed=int(spec.seed),
        error_type=str(spec.error_type),
        error_kwargs=err_kwargs,
    )
    runtime_s = time.perf_counter() - t0

    series = {
        "RP_LMP": np.asarray(rp_lmp, dtype=float),
        "LAED_LMP": np.asarray(laed_lmp, dtype=float),
        "PMP": np.asarray(pmp_price, dtype=float),
        "CMP": np.asarray(cmp_price, dtype=float),
        "LAED_TLMP_G1": np.asarray(laed_tlmp[0, :], dtype=float),
        "LAED_TLMP_G2": np.asarray(laed_tlmp[1, :], dtype=float),
    }

    metrics = {
        "runtime_s": float(runtime_s),
        "ramp_bind_count": int(diag.get("ramp_bind_count", 0)),
        "rp_shed_sum": _safe_float(np.sum(diag.get("rp_shed", 0.0))),
        "laed_shed_sum": _safe_float(np.sum(diag.get("laed_shed", 0.0))),
        "pmp_shed_sum": _safe_float(np.sum(diag.get("pmp_shed", 0.0))),
        "cmp_shed_sum": _safe_float(np.sum(diag.get("cmp_shed", 0.0))),
        "rp_shed_max": _safe_float(np.max(diag.get("rp_shed", 0.0))),
        "laed_shed_max": _safe_float(np.max(diag.get("laed_shed", 0.0))),
        "pmp_shed_max": _safe_float(np.max(diag.get("pmp_shed", 0.0))),
        "cmp_shed_max": _safe_float(np.max(diag.get("cmp_shed", 0.0))),
    }

    # If a series ever hits the load-shed penalty, price comparisons are typically dominated by that.
    penalty = float(getattr(lra, "cost_load", 0.0))
    penalty_thresh = 0.5 * penalty if penalty > 0 else float("inf")

    for name, arr in series.items():
        st = pvc.series_stats(arr)
        for k, v in st.items():
            metrics[f"{name}.{k}"] = float(v)
        metrics[f"{name}.penalty_frac"] = float(np.mean(arr >= penalty_thresh)) if np.isfinite(penalty_thresh) else 0.0

    return metrics


def summarize(runs_ok, group_keys, metric_keys):
    # Group by (error_label, sigma, rho) to aggregate across seeds.
    groups = {}
    for r in runs_ok:
        g = tuple(r[k] for k in group_keys)
        groups.setdefault(g, []).append(r)

    out = []
    for g, rows in groups.items():
        row0 = rows[0]
        base = {k: row0[k] for k in group_keys}
        base["n"] = len(rows)

        for mk in metric_keys:
            vals = np.array([_safe_float(rr.get(mk, float("nan"))) for rr in rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                base[f"{mk}.mean"] = float("nan")
                base[f"{mk}.std"] = float("nan")
            elif vals.size == 1:
                base[f"{mk}.mean"] = float(vals[0])
                base[f"{mk}.std"] = 0.0
            else:
                base[f"{mk}.mean"] = float(np.mean(vals))
                base[f"{mk}.std"] = float(np.std(vals, ddof=1))

        out.append(base)

    # Stable order: sigma, rho, error_label
    out.sort(key=lambda d: (float(d["sigma"]), float(d["rho"]), str(d["error_label"])))
    return out


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    # Use union of keys so failed runs (with an "error" field) still write cleanly.
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_summary(out_dir, summaries, series_for_volatility):
    import matplotlib.pyplot as plt

    # Facet by (sigma, rho)
    sigmas = sorted({float(s["sigma"]) for s in summaries})
    rhos = sorted({float(s["rho"]) for s in summaries})

    for sigma in sigmas:
        for rho in rhos:
            sub = [s for s in summaries if float(s["sigma"]) == sigma and float(s["rho"]) == rho]
            if not sub:
                continue

            err_labels = [s["error_label"] for s in sub]
            x = np.arange(len(err_labels), dtype=int)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Volatility (std of deltas)
            for name, key in series_for_volatility:
                y = np.array([_safe_float(s.get(f"{key}.mean", float("nan"))) for s in sub], dtype=float)
                yerr = np.array([_safe_float(s.get(f"{key}.std", 0.0)) for s in sub], dtype=float)
                ax1.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.6, capsize=3, label=name)

            ax1.set_ylabel("Std(Δprice) ($/MWh per interval)")
            ax1.set_title(f"Price Volatility by Error Quantification (sigma={sigma:g}, rho={rho:g})")
            ax1.grid(True, alpha=0.25)
            ax1.legend(ncol=2, fontsize=9)

            # Total committed load shed (sum over committed intervals)
            shed_keys = [
                ("RP", "rp_shed_sum"),
                ("LAED", "laed_shed_sum"),
                ("PMP", "pmp_shed_sum"),
                ("CMP", "cmp_shed_sum"),
            ]
            for name, key in shed_keys:
                y = np.array([_safe_float(s.get(f"{key}.mean", float("nan"))) for s in sub], dtype=float)
                yerr = np.array([_safe_float(s.get(f"{key}.std", 0.0)) for s in sub], dtype=float)
                ax2.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.6, capsize=3, label=name)

            ax2.set_ylabel("Total committed load shed (MW)")
            ax2.set_xlabel("Error quantification type")
            ax2.set_xticks(x)
            ax2.set_xticklabels(err_labels, rotation=25, ha="right")
            ax2.grid(True, alpha=0.25)
            ax2.legend(ncol=4, fontsize=9)

            fig.tight_layout()
            out_path = os.path.join(out_dir, f"uq_compare_sigma{sigma:g}_rho{rho:g}.png")
            fig.savefig(out_path, dpi=200)
            plt.close(fig)


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    os.environ.setdefault("MPLBACKEND", "Agg")

    ap = argparse.ArgumentParser(description="Compare demand uncertainty quantification types (no new error models).")
    ap.add_argument("--case", default="toy_data.dat")
    ap.add_argument("--projection", default="MISO_Projection.json")
    ap.add_argument("--ref-cap", type=float, default=350.0)
    ap.add_argument("--n-t", type=int, default=13, help="LAED rolling look-ahead window length.")
    ap.add_argument("--rp-n-t", type=int, default=2, help="RPED horizon length.")
    ap.add_argument("--ramp-factor", type=float, default=1.0, help="Ramp limit multiplier.")
    ap.add_argument("--load-factor", type=float, default=1.0, help="Load multiplier.")
    ap.add_argument("--cost-load", type=float, default=1e10, help="Load shedding penalty.")
    ap.add_argument("--error-types", default="gaussian,laplace,student-t")
    ap.add_argument("--sigmas", default="0.005", help="Comma-separated sigma values (relative).")
    ap.add_argument("--rhos", default="0.9", help="Comma-separated rho values.")
    ap.add_argument("--df", type=float, default=3.0, help="Student-t dof (only used if error-type=student-t).")
    ap.add_argument("--seeds", default="42", help="Comma-separated RNG seeds.")
    ap.add_argument("--max-steps", type=int, default=0, help="Limit number of committed intervals (0 = full).")
    ap.add_argument("--out-dir", default="uq_results")
    args = ap.parse_args()

    error_types = _parse_csv_list(args.error_types, str)
    sigmas = _parse_csv_list(args.sigmas, float)
    rhos = _parse_csv_list(args.rhos, float)
    seeds = _parse_csv_list(args.seeds, int)

    solver = SolverFactory("gurobi_direct")
    solver.options["OutputFlag"] = 0
    lra.reserve_factor = 0
    lra.cost_load = float(args.cost_load)

    data = DataPortal()
    data.load(filename=args.case)
    pvc.apply_aug_2032_projection(data, args.projection, args.ref_cap)
    data.data()["N_t"][None] = int(args.n_t)
    pvc.initialize_gen_init(data, solver)

    scaled = pvc.build_scaled_inputs(data, args.load_factor, args.ramp_factor)
    n_t_total = int(scaled["N_T"])
    n_steps_full = n_t_total - int(args.n_t) + 1
    n_steps = int(args.max_steps) if int(args.max_steps) > 0 else n_steps_full
    n_steps = max(1, min(n_steps, n_steps_full))

    os.makedirs(args.out_dir, exist_ok=True)

    runs = []
    total = len(error_types) * len(sigmas) * len(rhos) * len(seeds)
    idx = 0
    for et in error_types:
        for sigma in sigmas:
            for rho in rhos:
                for seed in seeds:
                    idx += 1
                    spec = RunSpec(error_type=et, sigma=sigma, rho=rho, seed=seed, df=float(args.df))
                    base_row = {
                        "status": "ok",
                        "error_type": et.lower(),
                        "error_label": _error_label(et, float(args.df)),
                        "sigma": float(sigma),
                        "rho": float(rho),
                        "seed": int(seed),
                        "df": float(args.df),
                        "n_t": int(args.n_t),
                        "rp_n_t": int(args.rp_n_t),
                        "ramp_factor": float(args.ramp_factor),
                        "ref_cap": float(args.ref_cap),
                        "n_steps": int(n_steps),
                    }
                    print(f"[{idx}/{total}] {base_row['error_label']} sigma={sigma:g} rho={rho:g} seed={seed}")
                    try:
                        metrics = run_one(
                            scaled=scaled,
                            laed_n_t=int(args.n_t),
                            rp_n_t=int(args.rp_n_t),
                            n_steps=n_steps,
                            solver=solver,
                            spec=spec,
                        )
                        base_row.update(metrics)
                    except Exception as e:
                        base_row["status"] = "fail"
                        base_row["error"] = str(e)
                    runs.append(base_row)

    runs_path = os.path.join(args.out_dir, "uq_runs.csv")
    write_csv(runs_path, runs)
    print(f"Wrote runs: {runs_path}")

    runs_ok = [r for r in runs if r.get("status") == "ok"]
    if not runs_ok:
        print("No successful runs; skipping summaries/plots.")
        return 2

    # Summary metrics to aggregate across seeds.
    metric_keys = [
        "runtime_s",
        "ramp_bind_count",
        "rp_shed_sum",
        "laed_shed_sum",
        "pmp_shed_sum",
        "cmp_shed_sum",
        # volatility (Std of deltas)
        "RP_LMP.std_delta",
        "LAED_LMP.std_delta",
        "PMP.std_delta",
        "CMP.std_delta",
        "LAED_TLMP_G1.std_delta",
        "LAED_TLMP_G2.std_delta",
        # tail volatility
        "RP_LMP.p95_abs_delta",
        "LAED_LMP.p95_abs_delta",
        "PMP.p95_abs_delta",
        "CMP.p95_abs_delta",
        "LAED_TLMP_G1.p95_abs_delta",
        "LAED_TLMP_G2.p95_abs_delta",
        # penalty hit rate
        "RP_LMP.penalty_frac",
        "LAED_LMP.penalty_frac",
        "PMP.penalty_frac",
        "CMP.penalty_frac",
        "LAED_TLMP_G1.penalty_frac",
        "LAED_TLMP_G2.penalty_frac",
    ]

    group_keys = ["error_label", "sigma", "rho"]
    summaries = summarize(runs_ok, group_keys=group_keys, metric_keys=metric_keys)
    summary_path = os.path.join(args.out_dir, "uq_summary.csv")
    write_csv(summary_path, summaries)
    print(f"Wrote summary: {summary_path}")

    # Plot summary.
    series_for_volatility = [
        ("RP LMP", "RP_LMP.std_delta"),
        ("LAED LMP", "LAED_LMP.std_delta"),
        ("PMP", "PMP.std_delta"),
        ("CMP", "CMP.std_delta"),
        ("LAED TLMP G1", "LAED_TLMP_G1.std_delta"),
        ("LAED TLMP G2", "LAED_TLMP_G2.std_delta"),
    ]
    plot_summary(args.out_dir, summaries, series_for_volatility=series_for_volatility)
    print(f"Wrote plots to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
