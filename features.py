"""Shared cleaning and feature engineering for the freight rate model.

Everything that touches the data lives here so that training, validation
and prediction use exactly the same logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
DIST_BINS = [0, 150, 300, 500, 800, 1200, 2000, 5000]
# A training row whose rate is below half or above double the typical rate
# for its equipment / distance bucket is treated as a corrupted label.
RATE_FACTOR_LOW, RATE_FACTOR_HIGH = 0.5, 2.0

NUMERIC_FEATURES = [
    "distance", "log_distance", "weight",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "dlat", "dlon", "haversine", "circuity",
    "market_index", "mi_daily", "mi_roll7", "quote_signal", "qs_daily",
]
TIME_FEATURES = ["dayofyear", "dayofweek", "month"]
CATEGORICAL_FEATURES = ["equipment_code"]


def haversine_miles(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


DATA_DIR = "data"
TRAIN_PATH = f"{DATA_DIR}/train_test.csv"
VALIDATION_PATH = f"{DATA_DIR}/validation.csv"
TEMPLATE_PATH = f"{DATA_DIR}/validation_predictions_template.csv"
DECEMBER_PATH = f"{DATA_DIR}/december_chart_inputs.csv"


def load_raw():
    """Return (train, validation, december_inputs) as DataFrames with parsed dates."""
    tr = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    va = pd.read_csv(VALIDATION_PATH, parse_dates=["date"])
    dec = pd.read_csv(DECEMBER_PATH, parse_dates=["date"])
    return tr, va, dec


def city_lookup(*frames: pd.DataFrame) -> pd.DataFrame:
    """Each city has exactly one coordinate pair in the data; build a lookup table."""
    parts = []
    for f in frames:
        if "pickup_lat" in f:
            parts.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
            parts.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).dropna().groupby("city")[["lat", "lon"]].median()


def daily_market(*frames: pd.DataFrame) -> pd.DataFrame:
    """market_index is a daily market signal with per-load noise; take the daily median.

    Built from every frame that carries the signal (train AND validation), so
    the table also covers November and December.
    """
    allf = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    daily = allf.groupby("date").agg(mi_daily=("market_index", "median"), qs_daily=("quote_signal", "median"))
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max())).interpolate()
    daily["mi_roll7"] = daily.mi_daily.rolling(7, center=True, min_periods=1).mean()
    return daily


def clean_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Negative weights are sign-flip errors; a truck cannot carry -36,000 lb."""
    df = df.copy()
    df["weight"] = df["weight"].abs()
    return df


def corrupt_rate_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows whose posted_rate is implausible for the load.

    Expected rate per mile is the median for the same equipment and distance
    bucket. Corrupted rows sit at ~0.2x or ~3-5x that value; genuine rows stay
    within about +/-15%.
    """
    rpm = df.posted_rate / df.distance
    bucket = pd.cut(df.distance, DIST_BINS)
    expected = rpm.groupby([df.equipment, bucket], observed=True).transform("median")
    factor = rpm / expected
    return (factor < RATE_FACTOR_LOW) | (factor > RATE_FACTOR_HIGH)


def make_features(df: pd.DataFrame, daily: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw frame (train, validation or December inputs) into model features."""
    df = clean_weight(df)

    # Coordinates: fill from the city lookup when the file does not carry them.
    for side in ("pickup", "delivery"):
        for c in ("lat", "lon"):
            col = f"{side}_{c}"
            looked = df[side].map(cities[c])
            df[col] = df[col].fillna(looked) if col in df else looked

    df["log_distance"] = np.log(df.distance)
    df["dlat"] = df.delivery_lat - df.pickup_lat
    df["dlon"] = df.delivery_lon - df.pickup_lon
    df["haversine"] = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    df["circuity"] = df.distance / df.haversine.clip(lower=1)

    # Daily market level; fill missing per-load signals with the daily value.
    df = df.join(daily, on="date")
    if "market_index" not in df:
        df["market_index"] = np.nan
    if "quote_signal" not in df:
        df["quote_signal"] = np.nan
    df["market_index"] = df.market_index.fillna(df.mi_daily)
    df["quote_signal"] = df.quote_signal.fillna(df.qs_daily)

    df["dayofyear"] = df.date.dt.dayofyear
    df["dayofweek"] = df.date.dt.dayofweek
    df["month"] = df.date.dt.month
    df["equipment_code"] = df.equipment.map(EQUIPMENT_CODES)
    return df
