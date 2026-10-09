"""Two-stage freight rate model.

Stage 1 (shape): gradient boosting on log(rate) using only load features
    (distance, equipment, weight, geography, day of week, and each load's
    deviation from the day's market level). No calendar or trend inputs, so
    the model cannot overfit to months it has seen.

Stage 2 (level): the daily median residual of stage 1 is a smooth series
    driven by the daily market index and a slow upward trend. Several ways to
    project it into unseen months are supported; see LEVEL_MODES.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, corrupt_rate_mask, make_features

T0 = pd.Timestamp("2025-01-01")
SHAPE_FEATURES = (
    [c for c in NUMERIC_FEATURES if not c.startswith(("market", "mi_", "quote", "qs_"))]
    + ["dayofweek", "mi_dev", "qs_dev"]
    + CATEGORICAL_FEATURES
)
LEVEL_MODES = {
    "none":          "no level adjustment; the shape model's average level is kept",
    "mi_t_frozen":   "level = a + b*market_index + c*t, with t frozen at the last training day",
    "mi_t_trend":    "level = a + b*market_index + c*t, trend extrapolated forward",
    "recent":        "level = mean residual over the last 8 training weeks (flat)",
}


def _prep(df: pd.DataFrame, daily: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    f = make_features(df, daily, cities)
    f["t"] = (f.date - T0).dt.days.astype(float)
    f["mi_dev"] = f.market_index - f.mi_daily
    f["qs_dev"] = f.quote_signal - f.qs_daily
    return f


def _wls(cols, y, w):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    sw = np.sqrt(np.asarray(w, float))
    return np.linalg.lstsq(X * sw[:, None], np.asarray(y, float) * sw, rcond=None)[0]


class FreightRateModel:
    def __init__(self, level: str = "none", gbm_params: dict | None = None):
        assert level in LEVEL_MODES, f"level must be one of {list(LEVEL_MODES)}"
        self.level_mode = level
        self.gbm_params = dict(max_iter=800, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=40,
                               l2_regularization=1.0, random_state=0)
        if gbm_params:
            self.gbm_params.update(gbm_params)

    # ------------------------------------------------------------------ fit
    def fit(self, train_raw: pd.DataFrame, daily: pd.DataFrame, cities: pd.DataFrame):
        self.daily, self.cities = daily, cities
        f = _prep(train_raw, daily, cities)
        corrupt = corrupt_rate_mask(train_raw).values
        self.n_dropped_ = int(corrupt.sum())
        f = f[~corrupt]

        self.shape_ = HistGradientBoostingRegressor(
            categorical_features=[SHAPE_FEATURES.index(c) for c in CATEGORICAL_FEATURES], **self.gbm_params
        ).fit(f[SHAPE_FEATURES], np.log(f.posted_rate))

        resid = np.log(f.posted_rate) - self.shape_.predict(f[SHAPE_FEATURES])
        d = pd.DataFrame({"resid": resid, "mi": f.mi_daily, "t": f.t, "date": f.date}).groupby("date").agg(
            resid=("resid", "median"), mi=("mi", "first"), t=("t", "first"), n=("resid", "size"))
        self.daily_resid_ = d
        self.t_max_ = float(d.t.max())
        self.level_coef_ = _wls([d.mi, d.t], d.resid, d.n)
        recent = d[d.index > d.index.max() - pd.Timedelta(days=56)]
        self.recent_level_ = float(np.average(recent.resid, weights=recent.n))
        return self

    # ---------------------------------------------------------------- level
    def level(self, f: pd.DataFrame) -> np.ndarray:
        a, b, c = self.level_coef_
        if self.level_mode == "none":
            return np.zeros(len(f))
        if self.level_mode == "recent":
            return np.full(len(f), self.recent_level_)
        t = f.t.values if self.level_mode == "mi_t_trend" else np.minimum(f.t.values, self.t_max_)
        return a + b * f.mi_daily.values + c * t

    # -------------------------------------------------------------- predict
    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        f = _prep(raw, self.daily, self.cities)
        return np.exp(self.shape_.predict(f[SHAPE_FEATURES]) + self.level(f))

    def describe(self) -> str:
        a, b, c = self.level_coef_
        return (f"shape GBM on {len(SHAPE_FEATURES)} features, {self.n_dropped_} corrupt rows dropped; "
                f"level[{self.level_mode}]: resid = {a:+.4f} + {b:.4f}*mi + {c*100:.4f}*(t/100d); "
                f"recent 8-week level = {self.recent_level_*100:+.2f}%")
