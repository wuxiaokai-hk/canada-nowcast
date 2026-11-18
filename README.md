# canada-nowcast: nowcasting Canadian GDP in real time

Statistics Canada publishes quarterly GDP about two months after the quarter ends. Monthly indicators
(monthly GDP by industry, jobs, hours, retail and manufacturing sales, trade, housing, prices) arrive much
sooner, each on its own schedule. A nowcast uses whatever has been published *so far* to estimate the quarter
before the official number is out.

The hard part is evaluating that honestly. This project backtests at every month using only the data that
existed on that date: each series cut at its own publication lag (the "ragged edge"), with GDP as it stood
before later revisions. It scores against both the first release and the latest value.

## Status

> **The pipeline and tests are complete. Real Canadian results are not in yet.** This workspace cannot reach
> Statistics Canada or the real-time database below, so the results shown are on **simulated** vintages with a
> known structure, publication lags and revisions. Run `python tools/get_lcdma.py` on your own computer, set
> `source: lcdma` in `config.yaml`, and re-run to get the real numbers. The reader matches the real LCDMA layout (one folder per vintage holding
> CAN_MD.csv and CAN_QD.csv, checked against the October 2026 files). A vintage with a different layout is
> skipped and listed; the May 2026 vintage is one.

## Models

| Model | Idea |
|---|---|
| AR(1) | the benchmark: GDP's own past |
| Bridge equations | one regression of GDP on each monthly indicator, with months not yet published filled by that indicator's own forecast; averaged across indicators |
| PCA factors | common factors of the monthly panel (missing cells filled iteratively), averaged to quarters |
| Dynamic factor model | the mixed-frequency DFM of Bańbura and Modugno (2014), via statsmodels `DynamicFactorMQ`; this is the approach central banks use |

## Backtest on simulated vintages (2010–2025)

RMSE relative to AR(1) (below 1 = better), against the latest GDP value:

| Months until GDP release | Bridge | PCA factors | Dynamic factor model |
|---|---|---|---|
| 5 (start of the quarter) | 0.97 | 1.07 | **0.89** |
| 4 | 0.79 | 0.77 | **0.70** |
| 3 | 0.75 | **0.60** | 0.76 |
| 2 (quarter over, data still arriving) | 0.65 | **0.42** | 0.76 |
| 1 | 0.64 | **0.42** | 0.76 |

Errors shrink as monthly data arrive, and all three models beat AR(1) from month 4 onward (Diebold-Mariano
p < 0.01). These numbers are for simulated data only; they show that the machinery works, not how well
Canada's GDP can be nowcast.

## Run

```bash
pip install -r requirements.txt
python -m nowcast.run                 # simulated vintages by default -> report/report.html
python -m pytest -q

# real data, in a terminal on your own computer (the download site is not reachable from every network):
python tools/get_lcdma.py --last 60   # about 400 MB of zipped vintages into data/lcdma/, unzipped
# config.yaml: source: lcdma   (target_column REAL_GDP and 16 monthly indicators are already set)
python -m nowcast.run
```

The tests check:
- No look-ahead: a series with a 2-month lag ends 2 months back, and GDP for a quarter appears only after its
  release.
- Quarters that are already published are never nowcast.
- Vintage files load correctly, including the transformation-code row.
- Transformation codes are applied correctly.
- The models beat AR(1) close to release on simulated data.

## Layout

```
nowcast/vintages.py    real vintages (folder), pseudo-real-time cuts, simulated vintages
nowcast/transforms.py  transformation codes (FRED-MD style), quarterly growth
nowcast/models.py      AR(1), bridge, PCA factors, mixed-frequency DFM
nowcast/backtest.py    nowcast at every vintage, score vs first release and latest, DM tests
nowcast/run.py         report/report.html and results.json
tools/get_lcdma.py     downloads the LCDMA vintages (run it on your own computer)
```

## Data

The real data source is the LCDMA real-time database: Fortin-Gagnon, Leroux, Stevanovic and Surprenant (2022),
"A large Canadian database for macroeconomic analysis", *Canadian Journal of Economics* 55(4). It has hundreds
of Canadian and provincial monthly series from 1981, with monthly vintages.
