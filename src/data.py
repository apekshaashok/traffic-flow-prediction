"""Data loading and feature engineering for hourly urban traffic volume.

Primary dataset: UCI "Metro Interstate Traffic Volume" (hourly westbound I-94 traffic
near Minneapolis-St Paul, 2012-2018, with weather and holiday information).

A synthetic generator with the same schema is included ONLY for smoke-testing the
pipeline when the real dataset cannot be downloaded. Never report synthetic results.
"""
import io
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_GLOB = "Metro_Interstate_Traffic_Volume.csv*"

LAGS = [1, 2, 3, 24, 168]  # hours: recent past, same hour yesterday, same hour last week


def download_uci(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract the UCI dataset (cached in data/)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    found = list(data_dir.glob(CSV_GLOB))
    if found:
        return found[0]
    print(f"Downloading dataset from {UCI_URL} ...")
    with urllib.request.urlopen(UCI_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(data_dir)
    found = list(data_dir.glob(CSV_GLOB))
    if not found:
        raise FileNotFoundError("Dataset zip did not contain Metro_Interstate_Traffic_Volume.csv[.gz]")
    return found[0]


def make_synthetic(n_days: int = 900, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the same schema as the UCI file (pipeline testing only)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n_days * 24, freq="h")
    hour, dow = idx.hour.values, idx.dayofweek.values
    doy = idx.dayofyear.values
    weekday = (
        1200
        + 2900 * np.exp(-0.5 * ((hour - 7.5) / 1.8) ** 2)
        + 3300 * np.exp(-0.5 * ((hour - 16.5) / 2.2) ** 2)
        + 800 * np.exp(-0.5 * ((hour - 12) / 3.0) ** 2)
    )
    weekend = 600 + 2300 * np.exp(-0.5 * ((hour - 13) / 4.0) ** 2)
    base = np.where(dow >= 5, weekend, weekday)
    temp = 283 + 15 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 3, len(idx))
    rain = np.where(rng.random(len(idx)) < 0.08, rng.exponential(1.5, len(idx)), 0.0)
    snow = np.where((temp < 273) & (rng.random(len(idx)) < 0.05), rng.exponential(0.1, len(idx)), 0.0)
    clouds = rng.integers(0, 101, len(idx))
    weather = np.where(rain > 0, "Rain", np.where(snow > 0, "Snow", np.where(clouds > 60, "Clouds", "Clear")))
    noise = np.zeros(len(idx))
    for i in range(1, len(idx)):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 250)
    vol = base * (1 - 0.08 * (rain > 0) - 0.15 * (snow > 0)) + noise
    vol = np.clip(vol, 0, 7280).round()
    holiday = np.full(len(idx), "None", dtype=object)
    holiday[(idx.month == 12) & (idx.day == 25) & (hour == 0)] = "Christmas Day"
    return pd.DataFrame({
        "holiday": holiday, "temp": temp, "rain_1h": rain, "snow_1h": snow,
        "clouds_all": clouds, "weather_main": weather,
        "date_time": idx.strftime("%Y-%m-%d %H:%M:%S"), "traffic_volume": vol,
    })


def load_raw(source: str = "uci", csv: str | None = None) -> pd.DataFrame:
    if source == "synthetic":
        return make_synthetic()
    path = Path(csv) if csv else download_uci()
    return pd.read_csv(path)


def build_features(raw: pd.DataFrame):
    """Clean raw rows, build an hourly series, and return (X, y, feature_groups).

    Steps: dedupe timestamps -> regular hourly grid -> fix known sensor outliers ->
    calendar / weather / lag features. Rows with missing target or lags are dropped so
    every model is evaluated on exactly the same rows.
    """
    df = raw.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.drop_duplicates("date_time", keep="first").sort_values("date_time").set_index("date_time")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))

    # Known data-quality issues: temp == 0 K and absurd rain_1h spikes.
    df.loc[df["temp"] <= 0, "temp"] = np.nan
    df.loc[df["rain_1h"] > 100, "rain_1h"] = np.nan
    for col in ["temp", "clouds_all"]:
        df[col] = df[col].ffill().bfill()
    for col in ["rain_1h", "snow_1h"]:
        df[col] = df[col].fillna(0.0)
    df["weather_main"] = df["weather_main"].ffill().fillna("Unknown")

    # Holiday is only flagged on the first hour of the day in the raw data -> whole day.
    hol = (df["holiday"].notna() & (df["holiday"] != "None")).astype(int)
    df["is_holiday"] = hol.groupby(df.index.date).transform("max").values

    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour"] = hour

    weather = pd.get_dummies(df["weather_main"], prefix="wx").astype(int)

    calendar = ["hour", "hour_sin", "hour_cos", "dow", "month", "is_weekend", "is_holiday"]
    weather_cols = ["temp", "rain_1h", "snow_1h", "clouds_all"] + list(weather.columns)
    lag_cols = []
    for lag in LAGS:
        name = f"lag_{lag}"
        df[name] = df["traffic_volume"].shift(lag)
        lag_cols.append(name)

    X = pd.concat([df[calendar + ["temp", "rain_1h", "snow_1h", "clouds_all"] + lag_cols], weather], axis=1)
    y = df["traffic_volume"]
    keep = y.notna() & X[lag_cols].notna().all(axis=1)
    X, y = X[keep], y[keep]
    groups = {"calendar_weather": calendar + weather_cols, "lags": lag_cols}
    return X, y, groups
