"""Time-based validation experiment.

Train on Jan-Aug 2025, evaluate on Sep-Oct 2025. This mirrors the real task:
the final model is trained on Jan-Oct and must predict Nov-Dec, two months it
has never seen. A random split would leak future market conditions into
training and overstate accuracy.

Run: uv run python validate.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, TIME_FEATURES, city_lookup,
                      corrupt_rate_mask, daily_market, load_raw, make_features)

SPLIT_DATE = "2025-09-01"

tr, va, dec = load_raw()
daily = daily_market(tr, va)
cities = city_lookup(tr, va)

new_cities = (set(va.pickup) | set(va.delivery)) - (set(tr.pickup) | set(tr.delivery))
n_new = (va.pickup.isin(new_cities) | va.delivery.isin(new_cities)).sum()
print(f"Validation rows touching a city unseen in training: {n_new:,} of {len(va):,} ({n_new/len(va):.1%})")

feat = make_features(tr, daily, cities)
feat["corrupt"] = corrupt_rate_mask(tr)
train = feat[feat.date < SPLIT_DATE]
test = feat[feat.date >= SPLIT_DATE]
train_clean = train[~train.corrupt]
print(f"train {train.date.min():%b}-{train.date.max():%b}: {len(train):,} rows, {train.corrupt.sum()} corrupt dropped")
print(f"test  {test.date.min():%b}-{test.date.max():%b}: {len(test):,} rows ({test.corrupt.sum()} corrupt labels kept for the 'all rows' score)")


def metrics(y, p):
    err = p - y
    return {"MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "MAPE%": 100 * (np.abs(err) / y).mean(), "R2": 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()}


results = []
def record(name, pred):
    y = test.posted_rate.values
    m_all = metrics(y, pred)
    m_clean = metrics(y[~test.corrupt.values], pred[~test.corrupt.values])
    results.append({"model": name, **{f"{k} (all)": v for k, v in m_all.items()},
                    **{f"{k} (clean)": v for k, v in m_clean.items() if k in ("MAE", "MAPE%")}})


# ---- Baselines -------------------------------------------------------------
rpm = train_clean.posted_rate / train_clean.distance
record("B1 global median $/mile x distance", rpm.median() * test.distance.values)

bucket_tr = pd.cut(train_clean.distance, [0, 150, 300, 500, 800, 1200, 2000, 5000])
tbl = rpm.groupby([train_clean.equipment, bucket_tr], observed=True).median()
bucket_te = pd.cut(test.distance, [0, 150, 300, 500, 800, 1200, 2000, 5000])
rpm_te = pd.Series(list(zip(test.equipment, bucket_te))).map(tbl.to_dict()).values
record("B2 median $/mile by equipment x distance", rpm_te * test.distance.values)


# ---- Gradient boosting variants -------------------------------------------
def fit_predict(features, target="log", data=train_clean, loss="squared_error", seed=0):
    model = HistGradientBoostingRegressor(
        loss=loss, max_iter=800, learning_rate=0.04, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
        categorical_features=[features.index(c) for c in CATEGORICAL_FEATURES if c in features],
    )
    X, Xt = data[features].values, test[features].values
    if target == "log":
        model.fit(X, np.log(data.posted_rate))
        return np.exp(model.predict(Xt)), model
    if target == "rpm":
        model.fit(X, data.posted_rate / data.distance)
        return model.predict(Xt) * test.distance.values, model
    model.fit(X, data.posted_rate)
    return model.predict(Xt), model


base = NUMERIC_FEATURES + CATEGORICAL_FEATURES
p, _ = fit_predict(base, "log");                        record("G1 GBM log(rate), geo+market, no calendar", p)
p, _ = fit_predict(base + TIME_FEATURES, "log");        record("G2 G1 + calendar (doy, dow, month)", p)
p, _ = fit_predict(base + ["dayofweek"], "log");        record("G3 G1 + day-of-week only", p)
p, _ = fit_predict(base + ["dayofweek"], "rpm");        record("G4 G3 but target = $/mile", p)
p, _ = fit_predict(base + ["dayofweek"], "raw");        record("G5 G3 but target = raw rate", p)
p, _ = fit_predict(base + ["dayofweek"], "log", data=train); record("G6 G3 trained WITH corrupt rows", p)
no_mkt = [f for f in base + ["dayofweek"] if not f.startswith(("market", "mi_", "quote", "qs_"))]
p, _ = fit_predict(no_mkt, "log");                      record("G7 G3 without any market signals", p)

res = pd.DataFrame(results).set_index("model")
pd.set_option("display.width", 200)
print("\nHoldout Sep-Oct 2025 (all = every row incl. corrupt labels; clean = corrupt labels excluded)")
print(res.round(3).to_string())

# feature importance (permutation) for the chosen variant
from sklearn.inspection import permutation_importance
feats = base + ["dayofweek"]
_, model = fit_predict(feats, "log")
tc = test[~test.corrupt]
imp = permutation_importance(model, tc[feats].values, np.log(tc.posted_rate), n_repeats=3, random_state=0)
print("\nPermutation importance (drop in R2 on log rate):")
print(pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False).round(4).to_string())
