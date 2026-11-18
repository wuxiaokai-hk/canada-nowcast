"""Nowcasting models. Each takes one vintage (monthly panel, quarterly target) and a target quarter, and
returns a forecast of that quarter's GDP growth using only what the vintage contains.

- ar:     AR(1) on released quarterly GDP growth (the benchmark)
- bridge: one bridge equation per monthly indicator (ragged edge filled by the indicator's own AR(1),
          months averaged to the quarter, GDP regressed on it), averaged across indicators
- pca:    factors from the standardized monthly panel (missing values filled by EM-style PCA iterations),
          averaged to quarters, GDP regressed on them
- dfm:    mixed-frequency dynamic factor model (Banbura and Modugno, 2014) via statsmodels DynamicFactorMQ,
          the method central banks use for this job
"""
import warnings

import numpy as np
import pandas as pd


def _quarter(ts):
    return pd.Timestamp(ts).to_period("Q").start_time


def ar(m, q, target, col="gdp"):
    y = q[col].dropna()
    if len(y) < 8:
        return np.nan
    b = np.polyfit(y.values[:-1], y.values[1:], 1)
    last, v = y.index[-1], y.values[-1]
    steps = (target.to_period("Q") - last.to_period("Q")).n
    for _ in range(steps):
        v = b[0] * v + b[1]
    return float(v)


def _fill_ar(s, until):
    """Extend a monthly series to `until` with its own AR(1) forecast."""
    s = s.dropna()
    b = np.polyfit(s.values[:-1], s.values[1:], 1)
    out, v, t = s.copy(), s.values[-1], s.index[-1]
    while t < until:
        t = t + pd.DateOffset(months=1)
        v = b[0] * v + b[1]
        out.loc[t] = v
    return out


def bridge(m, q, target, col="gdp"):
    end = target + pd.DateOffset(months=2)
    y = q[col].dropna()
    preds = []
    for c in m:
        if m[c].dropna().size < 36:
            continue
        x = _fill_ar(m[c], end).resample("QS").mean()
        d = pd.concat([y, x.rename("x")], axis=1).dropna()
        if len(d) < 12:
            continue
        b = np.polyfit(d.x, d[col], 1)
        preds.append(b[0] * x.loc[target] + b[1])
    return float(np.mean(preds)) if preds else np.nan


def _pca_fill(X, k=2, iters=30):
    """Fill missing cells of a standardized panel by iterating a rank-k PCA reconstruction."""
    Z = X.copy()
    mask = Z.isna()
    Z = Z.fillna(0.0)
    for _ in range(iters):
        U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
        R = (U[:, :k] * S[:k]) @ Vt[:k]
        Z = Z.where(~mask, pd.DataFrame(R, index=Z.index, columns=Z.columns))
    U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
    return pd.DataFrame(U[:, :k] * S[:k], index=Z.index)


def pca(m, q, target, col="gdp", k=2):
    end = target + pd.DateOffset(months=2)
    mm = m.reindex(pd.date_range(m.index[0], end, freq="MS"))
    mm = mm.loc[:, mm.notna().sum() > 36]
    Z = (mm - mm.mean()) / mm.std()
    f = _pca_fill(Z, k).resample("QS").mean()
    d = f.join(q[col]).dropna()
    A = np.c_[np.ones(len(d)), d[list(range(k))].values]
    b = np.linalg.lstsq(A, d[col].values, rcond=None)[0]
    return float(np.r_[1, f.loc[target].values] @ b)


class DFM:
    """Mixed-frequency DFM; parameters re-estimated every `refit` calls and reused in between (as in practice)."""

    def __init__(self, factors=1, refit=3):
        self.factors, self.refit, self.calls, self.params = factors, refit, 0, None

    def __call__(self, m, q, target, col="gdp"):
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
        end = target + pd.DateOffset(months=2)
        mm = m.reindex(pd.date_range(m.index[0], end, freq="MS"))
        mm = mm.loc[:, mm.notna().sum() > 36]
        qq = q[[col]].copy()
        qq.index = qq.index.to_period("Q")
        mm.index = mm.index.to_period("M")
        qq = qq.reindex(pd.period_range(qq.index[0], pd.Period(target, "Q"), freq="Q"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = DynamicFactorMQ(mm, endog_quarterly=qq, factors=self.factors, factor_orders=1,
                                  idiosyncratic_ar1=True, standardize=True)
            if self.params is None or self.calls % self.refit == 0:
                res = mod.fit(disp=False, maxiter=200)
                self.params = res.params
            else:
                res = mod.smooth(self.params)
        self.calls += 1
        # The quarterly series sits in the third month of its quarter; its smoothed/predicted value there is
        # the nowcast for the target quarter.
        yhat = res.predict(start=mm.index[0], end=mm.index[-1])
        v = yhat[col].iloc[-1]
        return float(v)


def all_models(dfm=True):
    out = {"AR(1)": ar, "Bridge equations": bridge, "PCA factors": pca}
    if dfm:
        out["Dynamic factor model"] = DFM()
    return out
