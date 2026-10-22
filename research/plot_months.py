"""Monthly charts: actual day-ahead price vs. forecasts issued 2..6 days ahead.

Reads data/predictions_<tag>.parquet from backtest.py, writes plots/<YYYY-MM>.png and
plots/all_months.pdf.

Usage: python plot_months.py [--tag default]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

HERE = Path(__file__).parent
TZ = "Europe/Berlin"
HORIZONS = [2, 3, 4, 5, 6]

# Reference palette (dataviz skill), light mode; validated as an adjacent 5-slot set.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ACTUAL = "#52514e"
SERIES = dict(zip(HORIZONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
FILL = "#e34948"
MONTHS_DE = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September",
             "Oktober", "November", "Dezember"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
    }
)


def month_figure(p: pd.DataFrame, month: pd.Period) -> plt.Figure:
    m = p[p["month"] == month]
    actual = m.groupby("time")["price"].first().sort_index()

    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(len(HORIZONS) + 1, 2, height_ratios=[1] * len(HORIZONS) + [1.5], hspace=0.55, top=0.94, bottom=0.04, left=0.06, right=0.98)
    mae_all = (m["pred"] - m["price"]).abs().mean()
    fig.suptitle(
        f"DE-LU Day-Ahead {MONTHS_DE[month.month - 1]} {month.year} – Börsenpreis vs. Prognose 2–6 Tage vorher",
        x=0.06, ha="left", fontsize=14, color=INK, y=0.997,
    )
    fig.text(
        0.06, 0.974,
        f"ct/kWh, Stundenmittel, Ortszeit · Ø Abweichung D+2…D+6: {mae_all:.2f} ct/kWh · "
        "rote Fläche = Abweichung Prognose ↔ Börsenpreis",
        ha="left", fontsize=9, color=INK_2,
    )

    top = None
    for i, h in enumerate(HORIZONS):
        ax = fig.add_subplot(gs[i, :], sharex=top, sharey=top)
        top = top or ax
        f = m[m["h"] == h].set_index("time")["pred"].sort_index()
        a = actual.reindex(f.index)
        ax.fill_between(f.index, a, f, color=FILL, alpha=0.18, linewidth=0, step=None)
        ax.plot(actual.index, actual.values, color=ACTUAL, linewidth=1.1, label="Börsenpreis")
        ax.plot(f.index, f.values, color=SERIES[h], linewidth=1.1, label=f"Prognose D+{h}")
        mae = (f - a).abs().mean()
        ax.set_title(f"Prognose {h} Tage vorher (D+{h}) · Ø Abweichung {mae:.2f} ct/kWh")
        ax.set_ylabel("ct/kWh")
        ax.axhline(0, color=INK_2, linewidth=0.6)
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2, tz=TZ))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.", tz=TZ))
        ax.set_xlim(actual.index.min(), actual.index.max())

    # mean 24h profile
    ax = fig.add_subplot(gs[-1, 0])
    prof_a = m.groupby("hour")["price"].mean()
    ax.plot(prof_a.index, prof_a.values, color=INK, linewidth=2.2, label="Börsenpreis")
    for h in HORIZONS:
        g = m[m["h"] == h].groupby("hour")["pred"].mean()
        ax.plot(g.index, g.values, color=SERIES[h], linewidth=1.3, label=f"D+{h}")
    ax.set_title("Mittleres Tagesprofil (24 h)")
    ax.set_xlabel("Uhrzeit")
    ax.set_ylabel("ct/kWh")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(0, 23)
    ax.legend(loc="upper center", ncol=6)

    # mean absolute deviation by hour of day
    ax = fig.add_subplot(gs[-1, 1])
    for h in HORIZONS:
        g = m[m["h"] == h]
        dev = (g["pred"] - g["price"]).abs().groupby(g["hour"]).mean()
        ax.plot(dev.index, dev.values, color=SERIES[h], linewidth=1.5, label=f"D+{h}")
    ax.set_title("Mittlere Abweichung nach Uhrzeit")
    ax.set_xlabel("Uhrzeit")
    ax.set_ylabel("ct/kWh")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(0, 23)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", ncol=5)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    p = pd.read_parquet(HERE / "data" / f"predictions_{args.tag}.parquet")
    p = p[p["h"].isin(HORIZONS)].copy()
    p.index.name = None
    p["time"] = p.index.tz_convert(TZ)
    p["hour"] = p["time"].dt.hour
    p["month"] = p["target_date"].dt.to_period("M")
    p[["price", "pred"]] = p[["price", "pred"]] / 10  # EUR/MWh -> ct/kWh

    out = HERE / "plots"
    out.mkdir(exist_ok=True)
    with PdfPages(out / "all_months.pdf") as pdf:
        for month in sorted(p["month"].unique()):
            fig = month_figure(p, month)
            fig.savefig(out / f"{month}.png", dpi=110)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"plots/{month}.png")


if __name__ == "__main__":
    main()
