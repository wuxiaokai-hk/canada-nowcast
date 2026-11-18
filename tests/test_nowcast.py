import numpy as np
import pandas as pd
import pytest

from nowcast.backtest import run, score
from nowcast.models import ar, bridge, pca
from nowcast.transforms import apply_code, stationary
from nowcast.vintages import pseudo_real_time, from_folder, simulate


def test_pseudo_real_time_has_no_look_ahead():
    idx = pd.date_range("2015-01-01", "2020-12-01", freq="MS")
    m = pd.DataFrame({"a": np.arange(len(idx), dtype=float), "b": 1.0}, index=idx)
    q = pd.DataFrame({"gdp": 1.0}, index=pd.date_range("2015-01-01", "2020-10-01", freq="QS"))
    v = pd.Timestamp("2019-06-01")
    mv, qv = pseudo_real_time(m, q, {"a": 0, "b": 2}, 2, v, v)[v]
    assert mv.a.last_valid_index() == pd.Timestamp("2019-05-01")      # lag 0: last month is May
    assert mv.b.last_valid_index() == pd.Timestamp("2019-03-01")      # lag 2: last month is March
    assert qv.gdp.last_valid_index() == pd.Timestamp("2019-01-01")    # 2019Q1 is out 2 months after March
    v0 = pd.Timestamp("2019-05-01")
    assert pseudo_real_time(m, q, {"a": 0}, 2, v0, v0)[v0][1].gdp.last_valid_index() == pd.Timestamp("2018-10-01")


def test_from_folder_reads_lcdma_layout(tmp_path):
    """The real layout: LCDMA_<Month>_<Year>/CAN_MD.csv and LCDMA_Q_<Month>_<Year>/CAN_QD.csv, with an unnamed
    row-number column before Date, NA for missing values, and macOS metadata folders to ignore."""
    m_dir = tmp_path / "LCDMA_March_2026"
    q_dir = tmp_path / "LCDMA_Q_March_2026"
    junk = tmp_path / "__MACOSX" / "LCDMA_March_2026"
    for d in (m_dir, q_dir, junk):
        d.mkdir(parents=True)
    (m_dir / "CAN_MD.csv").write_text('"","Date","GDP_new","EMP_CAN"\n"1",2026-01-01,100,20\n"2",2026-02-01,NA,21\n')
    (q_dir / "CAN_QD.csv").write_text('"","Date","REAL_GDP"\n"1",2025-10-01,2500\n"2",2026-01-01,NA\n')
    (junk / "CAN_MD.csv").write_text("garbage")
    v = from_folder(str(tmp_path))
    assert list(v) == [pd.Timestamp("2026-03-01")]
    m, q = v[pd.Timestamp("2026-03-01")]
    assert list(m.columns) == ["GDP_new", "EMP_CAN"] and np.isnan(m.GDP_new.iloc[-1]) and m.EMP_CAN.iloc[-1] == 21
    assert q.REAL_GDP.iloc[0] == 2500


def test_transform_codes():
    s = pd.Series([100.0, 110.0, 121.0])
    assert apply_code(s, 2).iloc[-1] == pytest.approx(11.0)
    assert apply_code(s, 5).iloc[-1] == pytest.approx(100 * np.log(1.1))
    assert stationary(pd.DataFrame({"neg": [-1.0, 1.0, 2.0]})).neg.iloc[-1] == 1.0


def test_backtest_skips_published_quarters_and_beats_ar_on_simulated_data():
    vint, truth, _ = simulate(vint_start="2016-01-01", seed=1)
    df = run(vint, {"AR(1)": ar, "Bridge": bridge, "PCA": pca})
    for v, g in df.groupby("vintage"):
        published = vint[v][1].gdp.dropna().index
        assert not set(g.quarter) & set(published)
    s = score(df)
    near = s[s.months_to_release <= 3]
    assert (near[near.model != "AR(1)"].rmse_vs_ar < 1).all()
    assert set(s.months_to_release) <= {1, 2, 3, 4, 5}
