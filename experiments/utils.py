from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Failed to parse config. Install PyYAML (`pip install pyyaml`) or use JSON-compatible YAML."
            ) from e
    return yaml.safe_load(text)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def utc_timestamp() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def git_commit_hash(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def git_is_dirty(repo_root: Path) -> bool | None:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root)
        return bool(out.decode("utf-8").strip())
    except Exception:
        return None


def environment_meta(repo_root: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp_utc": utc_timestamp(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "git": {
            "commit": git_commit_hash(repo_root),
            "dirty": git_is_dirty(repo_root),
        },
    }
    try:
        import pyomo  # type: ignore

        meta["pyomo_version"] = getattr(pyomo, "__version__", None)
    except Exception:
        meta["pyomo_version"] = None
    try:
        import gurobipy  # type: ignore

        meta["gurobipy_version"] = getattr(gurobipy, "__version__", None)
        try:
            meta["gurobi_version"] = getattr(gurobipy, "gurobi", None).version()
        except Exception:
            meta["gurobi_version"] = None
    except Exception:
        meta["gurobipy_version"] = None
        meta["gurobi_version"] = None
    return meta


class _ScenarioIdFilter(logging.Filter):
    def __init__(self, scenario_id: str):
        super().__init__()
        self._scenario_id = scenario_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "scenario_id"):
            record.scenario_id = self._scenario_id  # type: ignore[attr-defined]
        return True


def setup_scenario_logger(log_path: Path, scenario_id: str, console: bool = True) -> logging.Logger:
    logger = logging.getLogger(f"experiments.{scenario_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers so repeated runs in the same interpreter don't duplicate output.
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(scenario_id)s] %(message)s")
    sid_filter = _ScenarioIdFilter(scenario_id)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.addFilter(sid_filter)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.addFilter(sid_filter)
        logger.addHandler(ch)

    return logger


def slugify(s: str) -> str:
    out = []
    for ch in s.strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("-")
    return "".join(out)


def derive_seed(base_seed: int, parts: Iterable[str]) -> int:
    h = hashlib.sha256()
    h.update(str(int(base_seed)).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big") % (2**32)


def set_global_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))


def volatility_metrics(pi_t: Iterable[float], ddof: int = 1) -> dict[str, Any]:
    pi = np.asarray(list(pi_t), dtype=float).reshape(-1)
    pi = pi[np.isfinite(pi)]
    n = int(pi.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sigma": float("nan"),
            "V_MA": float("nan"),
            "Delta_max": float("nan"),
            "TV": float("nan"),
        }
    delta = np.diff(pi)
    sigma = float(np.std(pi, ddof=ddof)) if n > 1 else 0.0
    if delta.size == 0:
        v_ma = 0.0
        delta_max = 0.0
        tv = 0.0
    else:
        abs_delta = np.abs(delta)
        v_ma = float(np.mean(abs_delta))
        delta_max = float(np.max(abs_delta))
        tv = float(np.sum(abs_delta))
    return {
        "n": n,
        "mean": float(np.mean(pi)),
        "sigma": sigma,
        "V_MA": v_ma,
        "Delta_max": delta_max,
        "TV": tv,
    }


def apply_load_profile(
    base_load: dict[int, float],
    profile_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> dict[int, float]:
    kind = str(profile_cfg.get("kind", "baseline")).lower()
    keys = sorted(base_load.keys())
    series = np.array([float(base_load[t]) for t in keys], dtype=float)

    if kind in ("baseline", "smooth"):
        return {t: float(v) for t, v in zip(keys, series)}

    if kind != "aggressive":
        raise ValueError(f"Unknown load profile kind: {kind}")

    noise_std_rel = float(profile_cfg.get("noise_std_rel", 0.0))
    spike_prob = float(profile_cfg.get("spike_prob", 0.0))
    spike_scale = profile_cfg.get("spike_scale", [1.2, 1.6])
    delta_scale = float(profile_cfg.get("delta_scale", 1.0))

    series2 = series.copy()

    if delta_scale != 1.0 and series2.size > 0:
        series2 = series2[0] + delta_scale * (series2 - series2[0])

    if noise_std_rel > 0:
        series2 = series2 * (1.0 + rng.normal(0.0, noise_std_rel, size=series2.size))

    if spike_prob > 0:
        spikes = rng.random(series2.size) < spike_prob
        if np.any(spikes):
            lo, hi = float(spike_scale[0]), float(spike_scale[1])
            scales = rng.uniform(lo, hi, size=series2.size)
            series2[spikes] = series2[spikes] * scales[spikes]

    series2 = np.maximum(series2, 0.0)
    return {t: float(v) for t, v in zip(keys, series2)}


def load_miso_projection(path: Path, key: str, ref_cap: float) -> dict[int, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    series_raw = raw[key]
    items = sorted((int(k), float(v)) for k, v in series_raw.items())
    vals = np.array([v for _k, v in items], dtype=float)
    scale = float(ref_cap) / float(np.mean(vals))
    # Reindex to 1..T to match Pyomo RangeSet indexing conventions in this repo.
    return {i + 1: float(v * scale) for i, (_k, v) in enumerate(items)}


def tlmp_price_paid_by_load(
    tlmp_by_gen: np.ndarray,
    dispatch_by_gen: np.ndarray,
    demand: np.ndarray,
) -> np.ndarray:
    tlmp = np.asarray(tlmp_by_gen, dtype=float)
    p = np.asarray(dispatch_by_gen, dtype=float)
    d = np.asarray(demand, dtype=float).reshape(-1)
    if tlmp.shape != p.shape:
        raise ValueError(f"TLMP and dispatch must have the same shape; got {tlmp.shape} vs {p.shape}")
    if tlmp.shape[1] != d.size:
        raise ValueError(f"demand must have length T={tlmp.shape[1]}; got {d.size}")
    out = np.zeros(d.size, dtype=float)
    for t in range(d.size):
        if d[t] <= 0:
            out[t] = float("nan")
        else:
            out[t] = float(np.sum((p[:, t] / d[t]) * tlmp[:, t]))
    return out


def tlmp_marginal_unit_series(
    tlmp_by_gen: np.ndarray,
    dispatch_by_gen: np.ndarray,
    capacity_by_gen: np.ndarray,
    fallback: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    tlmp = np.asarray(tlmp_by_gen, dtype=float)
    p = np.asarray(dispatch_by_gen, dtype=float)
    cap = np.asarray(capacity_by_gen, dtype=float).reshape(-1)
    if cap.size != tlmp.shape[0]:
        raise ValueError(f"capacity length must be N_g={tlmp.shape[0]}; got {cap.size}")
    out = np.full(tlmp.shape[1], np.nan, dtype=float)
    for t in range(tlmp.shape[1]):
        candidates = [
            g
            for g in range(tlmp.shape[0])
            if (p[g, t] > eps) and (p[g, t] < cap[g] - eps)
        ]
        if candidates:
            g_star = max(candidates, key=lambda g: p[g, t])
            out[t] = float(tlmp[g_star, t])
        elif fallback is not None:
            out[t] = float(fallback[t])
    return out


def write_timeseries_csv(path: Path, columns: dict[str, Iterable[Any]]) -> None:
    # Prefer pandas when present (already in repo requirements), but keep a csv fallback.
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(columns)
        df.to_csv(path, index=False)
        return
    except Exception:
        pass

    import csv

    keys = list(columns.keys())
    rows = zip(*(list(columns[k]) for k in keys))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_summary_markdown(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            if not np.isfinite(v):
                return "nan"
            return f"{v:.6g}"
        return str(v)

    header = "| " + " | ".join(columns) + " |\n"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |\n"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |\n")
    path.write_text("".join(lines), encoding="utf-8")


def exc_to_str(e: BaseException) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))


def create_pyomo_solver(solver_cfg: dict[str, Any]):
    """Lazy-import Pyomo and construct a solver from config."""
    from pyomo.environ import SolverFactory  # type: ignore

    name = str(solver_cfg.get("name", "gurobi_direct"))
    solver = SolverFactory(name)
    for k, v in (solver_cfg.get("options") or {}).items():
        solver.options[k] = v
    return solver


def initialize_gen_init(data, solver, reserve_factor: float) -> None:
    """Initialize Gen_init consistent with existing scripts (see price_vol_compare.initialize_gen_init)."""
    from pyomo.environ import (  # type: ignore
        AbstractModel,
        Constraint,
        NonNegativeIntegers,
        NonNegativeReals,
        Objective,
        Param,
        RangeSet,
        Var,
        minimize,
    )

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

    model_ini = AbstractModel()
    model_ini.N_g = Param(within=NonNegativeIntegers)
    model_ini.G = RangeSet(1, model_ini.N_g)
    model_ini.Cost = Param(model_ini.G)
    model_ini.Capacity = Param(model_ini.G)
    model_ini.reserve_single = Param()
    model_ini.P = Var(model_ini.G, within=NonNegativeReals)
    model_ini.Reserve = Var(model_ini.G, within=NonNegativeReals)

    def objective_rule(model):
        return sum(model.Cost[g] * model.P[g] for g in model.G)

    model_ini.obj = Objective(rule=objective_rule, sense=minimize)

    def power_balance_rule(model):
        return sum(model.P[g] for g in model.G) == load_ini

    model_ini.power_balance_constraint = Constraint(rule=power_balance_rule)

    def capacity_rule(model, g):
        return model.P[g] + model.Reserve[g] <= model.Capacity[g]

    model_ini.capacity_constraint = Constraint(model_ini.G, rule=capacity_rule)

    def reserve_rule(model):
        return sum(model.Reserve[g] for g in model.G) >= float(reserve_factor) * load_ini

    model_ini.reserve_constraint = Constraint(rule=reserve_rule)

    def reserve_single_rule(model, g):
        return model.Reserve[g] <= model.reserve_single * model.Capacity[g]

    model_ini.reserve_single_constraint = Constraint(model_ini.G, rule=reserve_single_rule)

    ed_ini = model_ini.create_instance(data)
    solver.solve(ed_ini, tee=False)
    base["Gen_init"] = {g: float(ed_ini.P[g].value) for g in ed_ini.G}
