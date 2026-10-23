# Dashboard cards

Spot Forecast ships no Lovelace card of its own. It ships two dashboards for
the [Plotly Graph Card](https://github.com/dbuezas/lovelace-plotly-graph-card)
(`custom:plotly-graph`), written for and tested on **v4.0.2**.

1. Install *Plotly Graph Card* from HACS (category *Dashboard*) and reload
   the browser.
2. On a dashboard: *Add card* → *Manual*, and paste one of the YAML files.

Both cards read only the integration's fixed entity ids, so they work
unchanged on any instance language. Their labels are German; the comments in
the YAML are English and point at every label.

## Forecast card

[`examples/plotly_forecast.yaml`](https://github.com/cygnusb/ha-spot-forecast/blob/main/examples/plotly_forecast.yaml)

![The forecast card: known prices as a black step line, then six forecast days in blue with a hatched red peak-risk band above and a hatched green low-price band below, the dearest and cheapest hours of each day marked, and a peak warning above each forecast day.](https://raw.githubusercontent.com/cygnusb/ha-spot-forecast/main/assets/screenshots/forecast_light.png)

One time axis from yesterday to the end of the forecast:

* **known prices** (yesterday from the statistics, today and tomorrow from
  `prices`) as a solid step line in the theme's text colour;
* the **forecast** (expected price) as a blue step line;
* the **peak risk** hatched red from the expected price up to P90;
* the **low-price chance** hatched green from P10 up to the expected price,
  so likely negative hours reach below the zero line;
* per day the **4 dearest (▲) and 4 cheapest (▼) hours**;
* above each day its weekday, date and maximum, and "⚠ bis NN" in red when
  the peak risk crosses the threshold of the
  [peak-risk sensor](entities.md#peak-risk);
* nights shaded from `sun.sun`, and a dotted "jetzt" line where the forecast
  begins.

Prices are shown in ct/kWh (EUR/kWh × 100) with German number formatting.

## Look-back card

[`examples/plotly_lookback.yaml`](https://github.com/cygnusb/ha-spot-forecast/blob/main/examples/plotly_lookback.yaml)

![The look-back card: the actual price as a dark line and the forecast made two days ahead in orange, the gap between them filled red, with buttons for the horizon and the range and the mean error above the plot.](https://raw.githubusercontent.com/cygnusb/ha-spot-forecast/main/assets/screenshots/lookback_light.png)

How good were past forecasts on your install? The card draws the actual price
against the forecast made *h* days before, from the
[long-term statistics](entities.md#long-term-statistics):

* the buttons on the left choose the horizon, **D+1 … D+6**, or **alle** (all
  six lines, no fill); with one horizon, the gap to the actual price is filled
  red;
* the buttons on the right choose the range: **7 T, 30 T, 3 M, 6 M**;
* above the plot: the mean absolute error of the shown horizon over the shown
  range, and the error in the hours above the peak threshold.

It has content from the first day, thanks to the
[backfill](configuration.md#first-start-and-the-backfill). The horizon choice
holds until the page is reloaded and is shared by all look-back cards on the
page.

## How the screenshots are made

The images in the README, on the website and on this page are rendered by
[`tools/gen_card_screenshots.py`](https://github.com/cygnusb/ha-spot-forecast/blob/main/tools/gen_card_screenshots.py):
it mounts the shipped YAML in the released Plotly Graph Card bundle, against
states and statistics that the integration's own code computes from the
bundled model and research data
([`tools/card_scenarios.py`](https://github.com/cygnusb/ha-spot-forecast/blob/main/tools/card_scenarios.py)).
The moment shown, 14 October 2026, lies after the model's training cutoff, so
the forecast in it is out of sample, as it would be on a real install.
