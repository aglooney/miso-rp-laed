# Lost Opportunity Costs Under Ramp Stress

This repository contains the reproducible convex market models and experiments
used in `main_revised.tex`. The revision study compares:

- single-interval energy dispatch with a stylized ramp-capability product,
- rolling look-ahead economic dispatch settled at a uniform LA-LMP, and
- the same look-ahead dispatch settled at generator-specific TLMP.

The paper intentionally treats the ramp product as a transparent,
MISO-inspired benchmark rather than a reproduction of any ISO market engine.

## Reproduce the revision

Python 3.10 or newer is recommended. The implementation uses SciPy/HiGHS and
does not require a commercial solver.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/GridMod/RTS-GMLC data/RTS-GMLC
git -C data/RTS-GMLC sparse-checkout set \
  RTS_Data/SourceData RTS_Data/timeseries_data_files
PYTHONPATH=analysis .venv/bin/python analysis/test_market_models.py
PYTHONPATH=analysis .venv/bin/python analysis/run_revision_study.py
.venv/bin/python analysis/make_publication_outputs.py
PYTHONPATH=analysis .venv/bin/python analysis/validate_publication.py
```

The reported study used RTS-GMLC commit
`3ece0d3725c844056132393ee252b3083dd4eab4`. Aggregate results, the
generator-day panel, regression sample, and generated LaTeX tables are under
`outputs/revision/`. The final publication figures are
`images/10gen_dispatch_window_rp_vs_la.pdf` and
`images/rts_dispatchable_laed_vs_rp_loc_bivar_ylinear.pdf`.

To compile the manuscript with a local TeX installation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main_revised.tex
```

## Study coverage

The pipeline contains 1,120 aggregate simulation cases and 3,708
generator-level result rows. Its study coverage includes:

- 28 multi-season RTS-GMLC net-load days;
- four ramp multipliers and four stylized uncertainty-buffer levels;
- look-ahead horizons of 1, 3, 7, and 13 five-minute intervals;
- controlled forecast errors of 0%, 1%, 3%, and 5%, generated from a
  persistent AR(1) delivery-error path with coherent rolling revisions on
  both the 10- and 93-generator systems;
- a 93-unit convex thermal-plus-hydro RTS check on 12 days at two ramp
  multipliers, plus four-horizon and controlled-error multi-day ablations.

An exploratory 28-case raw day-ahead forecast check and an additional 24-case
73-unit thermal specification check are retained in the output data but are
not used for the paper's reported comparisons.

Emergency under- and over-generation variables are retained and reported so
that physically stressed cases cannot be mistaken for ordinary feasible
dispatch. Minimum-output, commitment, startup, network, and nonconvex cost
constraints are deliberately omitted; these are limitations of the study, not
features of an ISO implementation.

## Legacy scripts

The root-level Pyomo/Gurobi scripts are earlier exploratory models and are not
used to produce the revised paper. They remain in the repository for
provenance. The publication pipeline is the code under `analysis/`.
