"""Real-time backtest: at every vintage, nowcast every quarter whose GDP is not yet published (the previous
quarter while its release is pending, the current quarter, and optionally the next), using only that vintage.

Each nowcast is scored against two targets: the first release (what markets react to) and the latest value
(the best estimate of the truth). Results are grouped by months remaining until GDP is first published.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _quarter_start(ts):
    return pd.Timestamp(ts).to_period("Q").start_time


def run(vintages, models, lag_q=2, col="gdp", horizon_quarters=(-1, 0)):
    dates = sorted(vintages)
    first_release = {}
    for v in dates:                       # the first vintage in which each quarter appears
        q = vintages[v][1][col].dropna()
        for t, val in q.items():
            first_release.setdefault(t, val)
    latest = vintages[dates[-1]][1][col]
    rows = []
    for v in dates:
        m, q = vintages[v]
        for h in horizon_quarters:
            tq = _quarter_start(v) + pd.DateOffset(months=3 * h)
            if tq in q[col].dropna().index:
                continue                  # already published at this vintage
            release = tq + pd.DateOffset(months=3 + lag_q)
            months_to_release = (release.year - v.year) * 12 + release.month - v.month
            for name, f in models.items():
                rows.append({"vintage": v, "quarter": tq, "months_to_release": months_to_release, "model": name,
                             "nowcast": f(m, q, tq, col)})
    df = pd.DataFrame(rows)
    df["first_release"] = df.quarter.map(first_release)
    df["latest"] = df.quarter.map(latest)
    return df.dropna(subset=["first_release"])


def dm(e1, e2, h=1):
    """Diebold-Mariano (HLN-corrected) on squared errors; negative = model 1 better."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    if n < 8:
        return np.nan, np.nan
    db = d.mean()
    g = [np.mean((d[k:] - db) * (d[:n - k] - db)) for k in range(h)]
    var = (g[0] + 2 * sum(g[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    s = db / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(s), float(2 * stats.t.sf(abs(s), n - 1))


def score(df, target="latest", bench="AR(1)"):
    df = df.assign(err=df.nowcast - df[target])
    rows = []
    for (mtr, name), g in df.groupby(["months_to_release", "model"]):
        b = df[(df.months_to_release == mtr) & (df.model == bench)].set_index("quarter").err
        gg = g.set_index("quarter").err
        common = gg.index.intersection(b.index)
        s, p = dm(gg[common], b[common]) if name != bench else (np.nan, np.nan)
        rows.append({"months_to_release": mtr, "model": name, "n": len(g), "rmse": float(np.sqrt((g.err ** 2).mean())),
                     "rmse_vs_ar": float(np.sqrt((gg[common] ** 2).mean()) / np.sqrt((b[common] ** 2).mean())),
                     "dm_p": p})
    return pd.DataFrame(rows)
