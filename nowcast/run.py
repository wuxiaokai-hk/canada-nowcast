"""Run the real-time backtest and write report/report.html and report/results.json.

    python -m nowcast.run                    # source from config.yaml
"""
import base64
import html
import json
import os
import warnings

import pandas as pd
import yaml

from .backtest import run, score
from .models import all_models
from .transforms import stationary, quarterly_growth
from .vintages import simulate, from_folder

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK2, GRID, SURF = "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "font.size": 9, "lines.linewidth": 2, "legend.frameon": False,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})


def lcdma_vintages(cfg):
    raw = from_folder(cfg["lcdma_folder"])
    if not raw:
        raise SystemExit(f"No vintages in {cfg['lcdma_folder']}: run python tools/get_lcdma.py first.")
    last = raw[max(raw)]
    col = cfg.get("target_column")
    if not col:
        cands = [c for c in last[1].columns if "gdp" in c.lower() or "pib" in c.lower()]
        raise SystemExit("Set target_column in config.yaml to the real GDP series. GDP-like quarterly columns: "
                         + ", ".join(cands[:30]))
    mcols = cfg.get("monthly_columns") or list(last[0].notna().sum().sort_values(ascending=False)
                                               .index[: cfg.get("max_monthly", 20)])
    out = {}
    for v, (m, q) in raw.items():
        mm = stationary(m[[c for c in mcols if c in m]])
        qq = q[[col]].rename(columns={col: "gdp"})
        if (qq.gdp.dropna() > 0).all() and qq.gdp.dropna().median() > 100:      # a level: make it growth
            qq = quarterly_growth(qq.gdp).to_frame("gdp")
        qq.index = qq.index.to_period("Q").start_time
        out[v] = (mm, qq)
    return dict(sorted(out.items()))


def _img(out, name):
    raw = open(os.path.join(out, name), "rb").read()
    w = int.from_bytes(raw[16:20], "big") * 96 // 150
    return f'<img width="{w}" style="max-width:100%;height:auto" src="data:image/png;base64,{base64.b64encode(raw).decode()}">'


def _tab(df):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{v:,.3f}</td>" if isinstance(v, float) else f"<td>{html.escape(str(v))}</td>"
                                    for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main(cfg_path="config.yaml"):
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load(open(cfg_path))
    out = cfg["report_dir"]
    os.makedirs(out, exist_ok=True)
    if cfg["source"] == "simulated":
        vint, truth, _ = simulate(vint_start=cfg["vintage_start"])
        label = "SIMULATED data (known factor structure, publication lags and revisions)"
    else:
        vint = lcdma_vintages(cfg)
        label = f"LCDMA real-time vintages ({min(vint):%b %Y} to {max(vint):%b %Y})"
    df = run(vint, all_models(cfg.get("dfm", True)))
    df.to_csv(os.path.join(out, "nowcasts.csv"), index=False)
    s_latest = score(df, "latest")
    s_first = score(df, "first_release")
    R = {"source": label, "vintages": len(vint), "nowcasts": len(df),
         "rmse_latest": s_latest.round(4).to_dict("records"), "rmse_first_release": s_first.round(4).to_dict("records")}

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, (mname, g) in enumerate(s_latest.groupby("model")):
        g = g.sort_values("months_to_release", ascending=False)
        ax.plot(g.months_to_release, g.rmse, marker="o", ms=5, color=SERIES[i % 4], label=mname)
    ax.invert_xaxis()
    ax.set_xlabel("months until GDP is first published")
    ax.set_ylabel("RMSE (latest GDP growth)")
    ax.set_title("Nowcast error shrinks as monthly data arrive")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rmse_by_horizon.png"), dpi=150)
    plt.close(fig)

    best = s_latest[s_latest.model != "AR(1)"].groupby("model").rmse.mean().idxmin()
    q = sorted(df.quarter.unique())[-4]
    g = df[(df.quarter == q)].pivot_table(index="vintage", columns="model", values="nowcast")
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, c in enumerate(g.columns):
        ax.plot(g.index, g[c], marker="o", ms=4, color=SERIES[i % 4], label=c)
    ax.axhline(df[df.quarter == q].latest.iloc[0], color=INK2, ls="--", lw=1)
    ax.set_title(f"Nowcasts of {pd.Timestamp(q).to_period('Q')} GDP growth as data arrive (dashed: outcome)")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "one_quarter.png"), dpi=150)
    plt.close(fig)
    R["best_model"] = best
    json.dump(R, open(os.path.join(out, "results.json"), "w"), indent=1, default=str)

    css = ("body{font:15px/1.55 -apple-system,Segoe UI,Inter,sans-serif;background:#fcfcfb;color:#0b0b0b;margin:0}"
           "main{max-width:940px;margin:0 auto;padding:28px 22px 70px}h2{margin-top:34px;border-top:1px solid "
           "#e4e3df;padding-top:12px;font-size:19px}table{border-collapse:collapse;width:100%;font-size:13px}"
           "th,td{padding:6px 8px;border-bottom:1px solid #e4e3df;text-align:left}.s{color:#52514e}")
    page = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Canada GDP nowcast</title><style>{css}"
            "</style></head><body><main><h1>Real-time GDP nowcast: backtest</h1>",
            f"<p class=s>Data: {html.escape(label)}. {len(vint)} monthly vintages, {len(df)} nowcasts.</p>",
            "<h2>Error by months until release (against the latest GDP)</h2>", _img(out, "rmse_by_horizon.png"),
            _tab(s_latest), "<h2>Against the first release</h2>", _tab(s_first),
            "<h2>One quarter, vintage by vintage</h2>", _img(out, "one_quarter.png"), "</main></body></html>"]
    open(os.path.join(out, "report.html"), "w", encoding="utf-8").write("\n".join(page))
    print(s_latest.round(3).to_string())
    print("report:", os.path.join(out, "report.html"))


if __name__ == "__main__":
    main()
