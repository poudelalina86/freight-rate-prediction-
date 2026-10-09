"""Second validation experiment: how to forecast the daily rate LEVEL.

Two-stage model
  1. shape : GBM on log(rate) using load features only (no dates) ->
             learns distance / equipment / weight / geography effects.
  2. level : daily median of the shape residual, explained by the daily
             market index and a time trend; projected into the test months.

Train Jan-Aug, test Sep-Oct, same as validate.py.
Run: uv run python validate_level.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, city_lookup, corrupt_rate_mask,
                      daily_market, load_raw, make_features)

SPLIT = pd.Timestamp("2025-09-01")
tr, va, dec = load_raw()
daily = daily_market(tr, va)
cities = city_lookup(tr, va)
f = make_features(tr, daily, cities)
f["corrupt"] = corrupt_rate_mask(tr)
f["t"] = (f.date - pd.Timestamp("2025-01-01")).dt.days
f["mi_dev"] = f.market_index - f.mi_daily
f["qs_dev"] = f.quote_signal - f.qs_daily
train = f[(f.date < SPLIT) & ~f.corrupt]
test = f[f.date >= SPLIT]
tclean = ~test.corrupt.values
y = test.posted_rate.values

GEO = [c for c in NUMERIC_FEATURES if not c.startswith(("market", "mi_", "quote", "qs_"))] + ["dayofweek"] + CATEGORICAL_FEATURES
LOAD_MKT = ["mi_dev", "qs_dev"]           # per-load deviations from the daily market level


def gbm(features, seed=0):
    return HistGradientBoostingRegressor(max_iter=800, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=seed,
                                         categorical_features=[features.index(c) for c in CATEGORICAL_FEATURES])


def report(name, pred):
    e = pred - y
    ec = e[tclean]
    print(f"{name:58s} MAE(all)={np.abs(e).mean():7.2f}  MAE(clean)={np.abs(ec).mean():6.2f}  "
          f"MAPE(clean)={100*np.mean(np.abs(ec)/y[tclean]):5.2f}%  bias(clean)={100*np.mean(ec/y[tclean]):+5.2f}%")


# --- single-stage references ----------------------------------------------
m = gbm(GEO).fit(train[GEO], np.log(train.posted_rate))
report("S1 single GBM, load features only (= G7)", np.exp(m.predict(test[GEO])))
m = gbm(GEO + ["t"]).fit(train[GEO + ["t"]], np.log(train.posted_rate))
report("S2 single GBM + t (trees cannot extrapolate t)", np.exp(m.predict(test[GEO + ["t"]])))
m = gbm(GEO + ["mi_daily", "t"]).fit(train[GEO + ["mi_daily", "t"]], np.log(train.posted_rate))
report("S3 single GBM + mi_daily + t", np.exp(m.predict(test[GEO + ["mi_daily", "t"]])))

# --- two-stage -------------------------------------------------------------
shape_feats = GEO + LOAD_MKT
shape = gbm(shape_feats).fit(train[shape_feats], np.log(train.posted_rate))
train = train.assign(resid=np.log(train.posted_rate) - shape.predict(train[shape_feats]))
d = train.groupby("date").agg(resid=("resid", "median"), mi=("mi_daily", "first"), t=("t", "first"), n=("resid", "size"))
test_shape = shape.predict(test[shape_feats])
dt = test.groupby("date").agg(mi=("mi_daily", "first"), t=("t", "first"))


def wls(X, yv, w):
    X = np.column_stack([np.ones(len(yv))] + [np.asarray(c, dtype=float) for c in X])
    yv, sw = np.asarray(yv, dtype=float), np.sqrt(np.asarray(w, dtype=float))
    return np.linalg.lstsq(X * sw[:, None], yv * sw, rcond=None)[0]


def level_pred(coefs, cols):
    X = np.column_stack([np.ones(len(dt))] + [dt[c].values for c in cols])
    return pd.Series(X @ coefs, index=dt.index)

def two_stage(name, lvl):
    report(name, np.exp(test_shape + test.date.map(lvl).values))

b = wls([d.mi, d.t], d.resid, d.n)
print(f"\nlevel model on Jan-Aug: resid = {b[0]:+.4f} + {b[1]:.4f}*mi + {b[2]*100:.4f}*(t/100d)")
two_stage("T1 shape + level(mi + t, trend extrapolated)", level_pred(b, ["mi", "t"]))
dt_cap = dt.assign(t=np.minimum(dt.t, d.t.max()))
two_stage("T2 shape + level(mi + t, t frozen at last train day)", pd.Series(np.column_stack([np.ones(len(dt_cap)), dt_cap.mi, dt_cap.t]) @ b, index=dt.index))
b_mi = wls([d.mi], d.resid, d.n)
two_stage("T3 shape + level(mi only)", level_pred(b_mi, ["mi"]))
last6 = d[d.index >= d.index.max() - pd.Timedelta(weeks=6)]
two_stage("T4 shape + level = last-6-week mean (flat)", pd.Series(np.average(last6.resid, weights=last6.n), index=dt.index))
b_t = wls([d.t], d.resid, d.n)
two_stage("T5 shape + level(t only, extrapolated)", level_pred(b_t, ["t"]))
# trend fitted only on the last 4 months, mi + t
recent = d[d.index >= "2025-05-01"]
b_r = wls([recent.mi, recent.t], recent.resid, recent.n)
two_stage("T6 shape + level(mi + t fit on May-Aug only)", level_pred(b_r, ["mi", "t"]))
two_stage("T0 shape only, level = 0 (sanity)", pd.Series(0.0, index=dt.index))

# how did the actual Sep-Oct level compare with T1 / T2 forecasts, by week
test_resid = pd.Series(np.log(y) - test_shape, index=test.index)
act = test_resid[tclean].groupby(test.date[tclean]).median()
cmp = pd.DataFrame({"actual": act, "T1 trend": level_pred(b, ["mi", "t"]), "T2 frozen": pd.Series(np.column_stack([np.ones(len(dt_cap)), dt_cap.mi, dt_cap.t]) @ b, index=dt.index), "T3 mi only": level_pred(b_mi, ["mi"])})
print("\nWeekly mean level (log, x100 = %) in the holdout:")
print((cmp.resample("W").mean() * 100).round(2).to_string())
