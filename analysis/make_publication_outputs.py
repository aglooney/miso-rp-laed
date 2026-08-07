"""Create compact tables, figures, and inferential summaries for main.tex."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "revision"
IMAGES = ROOT / "images"


def _fmt_money(value: float) -> str:
    return f"{value:,.0f}"


def _bootstrap_ci(
    values: np.ndarray, *, statistic=np.median, seed: int = 299, draws: int = 5000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True)
    stats = np.apply_along_axis(statistic, 1, samples)
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def _group_summary(frame: pd.DataFrame) -> dict[str, float]:
    clean = frame[
        (frame["emergency_rp_mwh"] <= 1e-6)
        & (frame["emergency_la_mwh"] <= 1e-6)
    ]
    delta = clean["loc_rp"] - clean["loc_la"]
    return {
        "n": len(clean),
        "n_total": len(frame),
        "rp": float(np.median(clean["loc_rp"])),
        "la": float(np.median(clean["loc_la"])),
        "share": float(np.mean(delta > 1e-6)),
        "ties": float(np.mean(np.abs(delta) <= 1e-6)),
        "median_relief": float(np.median(delta)),
        "relief_q1": float(np.quantile(delta, 0.25)),
        "relief_q3": float(np.quantile(delta, 0.75)),
    }


def _write_ablation_table(aggregate: pd.DataFrame) -> dict[str, float]:
    feature = aggregate[
        (aggregate["month"] == 10)
        & (aggregate["day"] == 15)
        & (aggregate["study"] == "ramp_requirement")
        & np.isclose(aggregate["ramp_multiplier"], 0.1)
        & np.isclose(aggregate["upward_adder_fraction"], 0.16)
    ].iloc[0]
    horizon = aggregate[
        (aggregate["month"] == 10)
        & (aggregate["day"] == 15)
        & (aggregate["study"] == "horizon")
    ].set_index("horizon")
    rows = [
        ("RP dispatch", "energy only", feature["loc_rp_energy"]),
        ("same RP dispatch", "energy + RCP", feature["loc_rp"]),
        ("single interval", "uniform LMP", horizon.loc[1, "loc_la"]),
        ("13 intervals", "uniform LA-LMP", horizon.loc[13, "loc_la"]),
        ("same LA dispatch", "TLMP", feature["loc_tlmp"]),
    ]
    lines = [
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Dispatch/ablation & Settlement & LOC (\$) \\",
        r"\midrule",
    ]
    lines.extend(f"{a} & {b} & {c:,.0f} \\\\" for a, b, c in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUTPUT / "ablation_table.tex").write_text("\n".join(lines) + "\n")
    return {
        "feature_loc_rp_energy": float(feature["loc_rp_energy"]),
        "feature_loc_rp": float(feature["loc_rp"]),
        "feature_loc_la": float(feature["loc_la"]),
        "feature_loc_tlmp": float(feature["loc_tlmp"]),
        "feature_rcup_share": float(feature["rcup_active_share"]),
        "feature_max_rcup": float(feature["max_rcup"]),
    }


def _write_robustness_table(aggregate: pd.DataFrame) -> dict[str, dict[str, float]]:
    groups: list[tuple[str, pd.DataFrame]] = []
    ramp = aggregate[
        (aggregate["study"] == "ramp_requirement")
        & np.isclose(aggregate["upward_adder_fraction"], 0.16)
    ]
    for alpha in (0.2, 0.4, 0.8):
        groups.append((rf"$\alpha={alpha:.1f}$", ramp[np.isclose(ramp["ramp_multiplier"], alpha)]))
    forecast = aggregate[aggregate["study"] == "forecast_error"]
    groups.append(
        (
            r"AR(1) error, 5\%",
            forecast[np.isclose(forecast["sigma_rel"], 0.05)],
        )
    )
    for alpha in (0.2, 0.4, 0.8):
        groups.append(
            (
                rf"RTS-93, $\alpha={alpha:.1f}$",
                aggregate[
                    (aggregate["study"] == "external93")
                    & np.isclose(aggregate["ramp_multiplier"], alpha)
                ],
            )
        )
    groups.append(
        (
            r"RTS-93 AR(1), 5\%",
            aggregate[
                (aggregate["study"] == "external93_forecast_error")
                & np.isclose(aggregate["sigma_rel"], 0.05)
            ],
        )
    )
    summaries = {label: _group_summary(frame) for label, frame in groups}
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Case & No shed/spill & RP$>$LA & Med. $\Delta$ & IQR $\Delta$ \\",
        r"\midrule",
    ]
    for label, summary in summaries.items():
        lines.append(
            f"{label} & {summary['n']:.0f}/{summary['n_total']:.0f} & "
            f"{100 * summary['share']:.1f}\\% & "
            f"{_fmt_money(summary['median_relief'])} & "
            f"[{_fmt_money(summary['relief_q1'])}, "
            f"{_fmt_money(summary['relief_q3'])}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUTPUT / "robustness_table.tex").write_text("\n".join(lines) + "\n")
    return summaries


def _write_incidence_table(generator: pd.DataFrame) -> dict[str, float]:
    featured = generator[
        (generator["system"] == "10gen")
        & (generator["month"] == 10)
        & (generator["day"] == 15)
        & (generator["study"] == "ramp_requirement")
        & np.isclose(generator["ramp_multiplier"], 0.1)
        & np.isclose(generator["upward_adder_fraction"], 0.16)
    ].copy()
    featured["generator_num"] = featured["generator"].astype(int)
    featured = featured.sort_values("generator_num")
    nonzero = featured[
        (featured["loc_rp"] > 0.005) | (featured["loc_la"] > 0.005)
    ]
    lines = [
        r"\begin{tabular}{crrr}",
        r"\toprule",
        r"Gen. & RP-LMP & LA-LMP & Relief \\",
        r"\midrule",
    ]
    for row in nonzero.itertuples(index=False):
        relief = row.loc_rp - row.loc_la
        lines.append(
            f"{int(row.generator_num)} & {row.loc_rp:,.0f} & "
            f"{row.loc_la:,.0f} & {relief:,.0f} \\\\"
        )
    total_rp = float(featured["loc_rp"].sum())
    total_la = float(featured["loc_la"].sum())
    lines.extend(
        [
            r"\midrule",
            f"Total & {total_rp:,.0f} & {total_la:,.0f} & "
            f"{total_rp - total_la:,.0f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (OUTPUT / "incidence_table.tex").write_text("\n".join(lines) + "\n")
    largest = featured.loc[featured["loc_rp"].idxmax()]
    return {
        "feature_largest_generator": int(largest["generator_num"]),
        "feature_largest_share": float(largest["loc_rp"] / total_rp),
        "feature_nonzero_generators": int(len(nonzero)),
    }


def _standardize_within_system(frame: pd.DataFrame, column: str) -> pd.Series:
    def zscore(series: pd.Series) -> pd.Series:
        std = float(series.std(ddof=0))
        if std <= 1e-12:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - float(series.mean())) / std

    return frame.groupby("system", group_keys=False)[column].apply(zscore)


def _fit_regressions(
    generator: pd.DataFrame, aggregate: pd.DataFrame
) -> dict[str, dict[str, float]]:
    selected = generator[
        (
            (generator["system"] == "10gen")
            & (generator["study"] == "ramp_requirement")
            & np.isclose(generator["ramp_multiplier"], 0.4)
            & np.isclose(generator["upward_adder_fraction"], 0.16)
        )
        | (
            (generator["system"] == "RTS93")
            & (generator["study"] == "external93")
            & np.isclose(generator["ramp_multiplier"], 0.3)
            & np.isclose(generator["upward_adder_fraction"], 0.16)
        )
    ].copy()
    identifiers = [
        "system",
        "study",
        "month",
        "day",
        "horizon",
        "ramp_multiplier",
        "upward_adder_fraction",
        "forecast_type",
        "sigma_rel",
        "seed",
        "shortage_penalty",
    ]
    emergency = aggregate[
        identifiers + ["emergency_rp_mwh", "emergency_la_mwh"]
    ]
    selected = selected.merge(emergency, on=identifiers, how="left")
    selected = selected[
        (selected["emergency_rp_mwh"] <= 1e-6)
        & (selected["emergency_la_mwh"] <= 1e-6)
    ].copy()
    selected["log_flex"] = np.log(np.maximum(selected["flexibility"], 1e-9))
    selected["log_capacity"] = np.log(selected["capacity"])
    selected["loc_per_mw"] = selected["loc_rp"] / selected["capacity"]
    selected["relief_per_mw"] = (
        selected["loc_rp"] - selected["loc_la"]
    ) / selected["capacity"]
    selected["y_loc"] = np.log1p(np.maximum(selected["loc_per_mw"], 0.0))
    selected["y_relief"] = np.sign(selected["relief_per_mw"]) * np.log1p(
        np.abs(selected["relief_per_mw"])
    )
    predictors = {
        "rp_bind": "Bind frequency",
        "log_flex": "Log flexibility",
        "log_capacity": "Log capacity",
        "cost": "Marginal cost",
        "rp_utilization": "Utilization",
    }
    for column in predictors:
        selected[f"z_{column}"] = _standardize_within_system(selected, column)
    selected["is_rts"] = (selected["system"] == "RTS93").astype(float)
    x_columns = [f"z_{column}" for column in predictors] + ["is_rts"]
    x = sm.add_constant(selected[x_columns], has_constant="add")
    clusters = selected["system"].astype(str) + ":" + selected["generator"].astype(str)
    models = {
        "LOC/MW": sm.OLS(selected["y_loc"], x).fit(
            cov_type="cluster", cov_kwds={"groups": clusters}
        ),
        "RP-LA relief/MW": sm.OLS(selected["y_relief"], x).fit(
            cov_type="cluster", cov_kwds={"groups": clusters}
        ),
    }

    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"& $\log(1+\mathrm{RP\ LOC/MW})$ & $s(\mathrm{relief/MW})$ \\",
        r"\midrule",
    ]
    output: dict[str, dict[str, float]] = {}
    for column, label in predictors.items():
        key = f"z_{column}"
        cells = []
        output[label] = {}
        for model_name, model in models.items():
            coef = float(model.params[key])
            se = float(model.bse[key])
            pvalue = float(model.pvalues[key])
            stars = "***" if pvalue < 0.01 else "**" if pvalue < 0.05 else "*" if pvalue < 0.1 else ""
            cells.append(f"{coef:.3f}{stars} ({se:.3f})")
            output[label][f"{model_name}_coef"] = coef
            output[label][f"{model_name}_se"] = se
            output[label][f"{model_name}_p"] = pvalue
        lines.append(f"{label} & {cells[0]} & {cells[1]} \\\\")
    lines.append(r"\midrule")
    lines.append(
        f"Observations & {int(models['LOC/MW'].nobs)} & {int(models['RP-LA relief/MW'].nobs)} \\\\"
    )
    lines.append(
        f"$R^2$ & {models['LOC/MW'].rsquared:.3f} & {models['RP-LA relief/MW'].rsquared:.3f} \\\\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUTPUT / "regression_table.tex").write_text("\n".join(lines) + "\n")
    output["model"] = {
        "n": int(models["LOC/MW"].nobs),
        "clusters": int(clusters.nunique()),
        "loc_r2": float(models["LOC/MW"].rsquared),
        "relief_r2": float(models["RP-LA relief/MW"].rsquared),
    }
    selected.to_csv(OUTPUT / "regression_sample.csv", index=False)
    return output


def _make_figure(aggregate: pd.DataFrame, generator: pd.DataFrame) -> None:
    horizon_cases = aggregate[aggregate["study"] == "external93_horizon"].copy()
    horizon_cases["clean"] = (
        (horizon_cases["emergency_rp_mwh"] <= 1e-6)
        & (horizon_cases["emergency_la_mwh"] <= 1e-6)
    )
    common_days = (
        horizon_cases.groupby(["month", "day"])["clean"]
        .all()
        .loc[lambda values: values]
        .index
    )
    common_index = pd.MultiIndex.from_arrays(
        [horizon_cases["month"], horizon_cases["day"]]
    )
    horizon_cases = horizon_cases[common_index.isin(common_days)]
    horizon = (
        horizon_cases
        .groupby("horizon")
        .agg(rp=("loc_rp", "median"), la=("loc_la", "median"), tlmp=("loc_tlmp", "median"))
        .reset_index()
    )
    featured = generator[
        (generator["system"] == "10gen")
        & (generator["month"] == 10)
        & (generator["day"] == 15)
        & np.isclose(generator["ramp_multiplier"], 0.1)
        & np.isclose(generator["upward_adder_fraction"], 0.16)
    ].copy()
    featured["generator_num"] = featured["generator"].astype(int)
    featured = featured.sort_values("generator_num")

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35))
    ax = axes[0]
    ax.plot(horizon["horizon"], horizon["rp"] / 1e6, "o-", label="RP-LMP")
    ax.plot(horizon["horizon"], horizon["la"] / 1e6, "s-", label="LA-LMP")
    ax.plot(horizon["horizon"], horizon["tlmp"] / 1e6, "^-", label="TLMP")
    ax.set_xticks([1, 3, 7, 13])
    ax.set_xlabel("Look-ahead intervals")
    ax.set_ylabel("Median daily LOC ($ million)")
    ax.set_title("(a) Horizon ablation, 10 RTS days")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    x = np.arange(len(featured))
    width = 0.38
    ax.bar(x - width / 2, featured["loc_rp"], width, label="RP-LMP")
    ax.bar(x + width / 2, featured["loc_la"], width, label="LA-LMP")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(x)
    ax.set_xticklabels(featured["generator"])
    ax.set_xlabel("Generator")
    ax.set_ylabel("LOC ($, symlog)")
    ax.set_title("(b) Clean active-RCP day")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(w_pad=1.4)
    IMAGES.mkdir(exist_ok=True)
    fig.savefig(IMAGES / "revision_summary.pdf", bbox_inches="tight")
    fig.savefig(IMAGES / "revision_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    aggregate = pd.read_csv(OUTPUT / "aggregate_results.csv")
    generator = pd.read_csv(OUTPUT / "generator_panel.csv")
    summary: dict[str, object] = {}
    summary.update(_write_ablation_table(aggregate))
    summary["robustness"] = _write_robustness_table(aggregate)
    summary.update(_write_incidence_table(generator))
    summary["regression"] = _fit_regressions(generator, aggregate)
    _make_figure(aggregate, generator)

    severe = aggregate[
        (aggregate["study"] == "ramp_requirement")
        & np.isclose(aggregate["ramp_multiplier"], 0.1)
        & np.isclose(aggregate["upward_adder_fraction"], 0.16)
    ]
    relief = severe["loc_rp"] - severe["loc_la"]
    ci = _bootstrap_ci(relief.to_numpy())
    summary["severe_28day"] = {
        "n": len(severe),
        "share_rp_gt_la": float(np.mean(relief > 1e-6)),
        "median_relief": float(np.median(relief)),
        "median_relief_ci": list(ci),
        "median_rp": float(np.median(severe["loc_rp"])),
        "median_la": float(np.median(severe["loc_la"])),
        "emergency_any_share": float(
            np.mean(
                (severe["emergency_rp_mwh"] > 1e-6)
                | (severe["emergency_la_mwh"] > 1e-6)
            )
        ),
    }
    (OUTPUT / "publication_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
