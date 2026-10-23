# Regional holidays and DST-aligned price lags

Research of 2026-10-23, on the training pipeline: do fuller holiday features (state-weighted
regional holidays, bridge days, Dec 24/31, Luxembourg) or price lags at the same local
wall-clock hour across a DST switch improve the DE-LU model? The holidays did not, the lags
became part of feature schema 2. Summary in [`../README.md`](../README.md), full output in
`results.txt`.

| File | What |
|---|---|
| `build.py` | the training dataset once, plus the candidate columns (`holiday_table`, `dst_lags`), cached |
| `wf.py` | monthly walk-forward of one variant (`base`, `base_seed1`, `hol`, `dst`, `both`), median model |
| `evaluate.py` | the variants on identical rows: per horizon, on the affected subsets, per month, with day-bootstrap CIs |

Outputs go to `research/data/holidays_dst/` (git-ignored). Run from the repo root:

```sh
.venv/bin/python research/holidays_dst/build.py
.venv/bin/python research/holidays_dst/wf.py dst 600     # each variant, about 40 minutes
.venv/bin/python research/holidays_dst/evaluate.py
```

`results.txt` was produced on feature schema 1 (v0.1.2). Since schema 2 the baseline
features already have the wall-clock lags (`dst_lags` is what `core.features` implements,
`tests/training/test_dataset.py` checks it) and relative price levels, so the `dst`
variant reproduces only on the v0.1.2 code.
