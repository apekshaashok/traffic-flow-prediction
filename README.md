# Machine Learning for Traffic Flow Prediction in Urban Areas

Predicts urban traffic congestion from calendar, weather and recent-traffic features, as two tasks:
(1) five-level congestion classification (the task of the Stanford CS229 Spring 2021 project of the same
name) and (2) hourly traffic volume regression. Uses a different dataset (UCI Metro Interstate) and a
different approach (tree-based and linear models vs. baselines, plus a neural net) from that paper.

**Dataset:** [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
(hourly I-94 traffic, 2012-2018, with weather and holidays). It is downloaded automatically on first run.

## Setup

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd traffic-flow-prediction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# 1. Train and evaluate all models (downloads the dataset into data/ on first run)
python -m src.train

# 2. Generate the PDF write-up from the results
python -m src.make_writeup --author "Your Name" --srn "Your SRN"
```

Outputs (in `outputs/`): `results.json` (metrics), `pred_vs_actual.png` (figure), `writeup.pdf`.

If the automatic download fails (e.g. restricted network), download the zip from the UCI page, extract
`Metro_Interstate_Traffic_Volume.csv.gz`, and run `python -m src.train --csv path/to/that/file`.

`python -m src.train --data synthetic` runs the pipeline on generated data. It is only for testing
that the code runs; never report its numbers.

## Project layout

```
src/data.py          download, cleaning, feature engineering (calendar, weather, lags)
src/train.py         chronological split, baselines, models, metrics, figure
src/make_writeup.py  builds outputs/writeup.pdf from outputs/results.json
```

## Method summary

- **Split:** chronological 80/20 (train on the past, test on the future).
- **Features:** hour (raw + sin/cos), day of week, month, weekend/holiday flags, temperature, rain, snow,
  clouds, weather category; lag traffic at 1, 2, 3, 24 and 168 hours.
- **Metrics:** MAE, RMSE, R² on the test period.
