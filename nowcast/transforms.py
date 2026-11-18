"""Make series stationary. Real-time databases such as FRED-MD (and, in the same style, LCDMA) ship a
transformation code per series; without codes, log-difference positive series and difference the rest."""
import numpy as np
import pandas as pd

CODES = {1: "level", 2: "diff", 3: "diff2", 4: "log", 5: "dlog", 6: "dlog2", 7: "dpct"}


def apply_code(s, code):
    code = int(code)
    if code == 1:
        return s
    if code == 2:
        return s.diff()
    if code == 3:
        return s.diff().diff()
    if code == 4:
        return np.log(s)
    if code == 5:
        return 100 * np.log(s).diff()
    if code == 6:
        return 100 * np.log(s).diff().diff()
    if code == 7:
        return 100 * (s / s.shift(1) - 1).diff()
    raise ValueError(f"unknown transformation code {code}")


def stationary(panel, codes=None):
    out = {}
    for c in panel:
        s = panel[c]
        if codes is not None and c in codes and pd.notna(codes[c]):
            out[c] = apply_code(s, codes[c])
        elif (s.dropna() > 0).all():
            out[c] = 100 * np.log(s).diff()
        else:
            out[c] = s.diff()
    return pd.DataFrame(out, index=panel.index)


def quarterly_growth(level, annualized=True):
    g = np.log(level).diff() * 100
    return g * 4 if annualized else g
