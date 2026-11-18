"""Real-time data: what was published, as of each date.

A *vintage* is the data set exactly as it stood on one date: each series ends where its latest release
ended (a ragged edge), and values are as first published or revised up to that date. Nowcasting honestly
means using only vintage v to nowcast at date v.

Three sources fill the same structure, {vintage date: (monthly DataFrame, quarterly DataFrame)}:
  1. A folder of real vintages, e.g. the LCDMA real-time database (Fortin-Gagnon, Leroux, Stevanovic and
     Surprenant, 2022, Canadian Journal of Economics), one monthly and one quarterly CSV per vintage.
  2. Pseudo-real-time from one current data set: each vintage is cut at that date minus each series'
     publication lag. This gets the ragged edge right but ignores revisions (documented as such).
  3. Simulated vintages with a known factor structure, lags and revisions (for tests and the demo).
"""
import glob
import os
import re

import numpy as np
import pandas as pd

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}


def _read_panel(path):
    """Read a CSV with a date column first. Rows that are not dates (e.g. a transformation-code row, as in
    FRED-MD-style files) are returned separately."""
    raw = pd.read_csv(path)
    first = raw.columns[0]
    dates = pd.to_datetime(raw[first], errors="coerce")
    meta = raw[dates.isna()].set_index(first)
    df = raw[dates.notna()].copy()
    df.index = pd.DatetimeIndex(dates[dates.notna()]).to_period("M").to_timestamp()
    df = df.drop(columns=first).apply(pd.to_numeric, errors="coerce")
    return df, meta


def from_folder(folder, monthly_glob="*LCDMA_[A-Z]*.csv", quarterly_glob="*LCDMA_Q_*.csv"):
    """Real vintages saved as files named like LCDMA_March_2026.csv / LCDMA_Q_March_2026.csv.

    The LCDMA file layout is read generically (date column first, one column per series, optional code row);
    check the first loaded vintage against the source before relying on it.
    """
    def vdate(p):
        m = re.search(r"_([A-Za-z]+)_(\d{4})", os.path.basename(p))
        return pd.Timestamp(int(m.group(2)), MONTHS[m.group(1).lower()], 1) if m else None

    mon = {vdate(p): p for p in glob.glob(os.path.join(folder, monthly_glob)) if "_Q_" not in p}
    qua = {vdate(p): p for p in glob.glob(os.path.join(folder, quarterly_glob))}
    out = {}
    for v in sorted(set(mon) & set(qua)):
        out[v] = (_read_panel(mon[v])[0], _read_panel(qua[v])[0])
    return out


def pseudo_real_time(monthly, quarterly, lags_m, lag_q, start, end, freq="MS"):
    """Cut the latest data at each vintage date: a series with lag L (months) observed at date v ends at
    the month v - L - 1. Quarterly GDP for quarter q is published lag_q months after the quarter ends."""
    out = {}
    for v in pd.date_range(start, end, freq=freq):
        m = monthly.copy()
        for c in m:
            last = v - pd.DateOffset(months=int(lags_m.get(c, 1)) + 1)
            m.loc[m.index > last, c] = np.nan
        q = quarterly.copy()
        q_end = q.index + pd.DateOffset(months=3)         # first day after the quarter
        q.loc[q_end + pd.DateOffset(months=lag_q) > v] = np.nan
        out[v] = (m.loc[:v], q.loc[:v])
    return out


def simulate(n_series=12, start="1995-01-01", end="2025-12-01", vint_start="2010-01-01", seed=0,
             revision_sd=0.15):
    """Simulated monthly panel with 2 AR(1) factors, quarterly GDP growth loading on the quarterly average of
    factor 1, publication lags of 0-2 months, GDP released 2 months after the quarter, and revisions that
    shrink with age. Returns (vintages, truth) where truth is final quarterly GDP growth."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="MS")
    T = len(idx)
    F = np.zeros((T, 2))
    for t in range(1, T):
        F[t] = [0.8, 0.5] * F[t - 1] + rng.normal(0, 1, 2)
    lam = rng.uniform(0.4, 1.0, (n_series, 2)) * np.sign(rng.normal(size=(n_series, 2)))
    lam[: n_series // 2, 0] = np.abs(lam[: n_series // 2, 0])
    X = F @ lam.T + rng.normal(0, 0.8, (T, n_series))
    names = [f"x{i:02d}" for i in range(n_series)]
    monthly = pd.DataFrame(X, index=idx, columns=names)
    fq = pd.DataFrame(F, index=idx).resample("QS").mean()
    gdp = 0.5 + 0.45 * fq[0] + rng.normal(0, 0.25, len(fq))
    truth = pd.DataFrame({"gdp": gdp.values}, index=fq.index)
    lags = {c: int(rng.integers(0, 3)) for c in names}
    vint = pseudo_real_time(monthly, truth, lags, 2, vint_start, end)
    # revisions: a GDP value k months after its first release carries noise sd * 0.7^k
    for v, (m, q) in vint.items():
        first_pub = q.index + pd.DateOffset(months=5)
        age = np.clip(np.asarray((v - first_pub).days) // 30, 0, None)
        noise = rng.normal(0, 1, len(q)) * revision_sd * (0.7 ** age)
        vint[v] = (m, q.assign(gdp=q.gdp + np.where(q.gdp.notna(), noise, 0)))
    return vint, truth, lags
