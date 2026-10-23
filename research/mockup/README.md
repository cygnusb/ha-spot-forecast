# Card mock-up (reference)

The approved design for the Plotly cards: variant A ("Verlauf") out of three mock-ups
(A Verlauf, B Tageskacheln, C Kalender). Variant A shows one continuous time axis:
known day-ahead prices on the left, the forecast on the right with the hatched P90
peak-risk band, nights shaded, "now" dotted; next to it a review chart that compares
the forecast made x days ahead with the actual price. `shot_a.png` is a screenshot
of variant A. All numbers are real backtest data for DE-LU (forecast state of
2026-09-20 10:00, review until 2026-10-20).

Files:

- `template.html`: the page, with the chart code inline and a `/*DATA*/`
  placeholder for the data. This is what `build.py` uses.
- `main.js`: a working copy of the inline script from shortly before the last edits
  to `template.html` (day labels "Ø … max …", no y-range refit on the review buttons).
  `shot_a.png` was taken from that state.
- `export.py`: writes `data.json` from the research backtest predictions.
- `build.py`: puts the data from 2026-04-01 on into the template and writes
  `spot-entwuerfe.html` and `preview.html`.

`data.json` and the built pages are large and regenerable, so they are not committed.

## Rebuild

Uses the research environment and data from `research/README.md`.

```sh
cd research
.venv/bin/python backtest.py --tag default               # data/predictions_default.parquet (median)
.venv/bin/python backtest.py --quantile 0.9 --tag p90    # data/predictions_p90.parquet
.venv/bin/python mockup/export.py mockup/data.json
cd mockup && ../.venv/bin/python build.py                # open spot-entwuerfe.html
```

`export.py` reads both prediction files from `data/` (relative to the working
directory, hence `cd research`), adds the rolling conformal offset to the P90 (trailing
56 days' 90 % quantile of `price − P90` per horizon, as in `research/README.md`) and
writes hourly series in ct/kWh, Europe/Berlin time: the actual price and, per horizon
D+1 … D+6, the forecast and the calibrated P90.
