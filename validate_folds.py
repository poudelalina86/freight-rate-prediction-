"""Rolling-origin validation: three forecast folds, each predicting the two
months after its training window, exactly like the real Nov-Dec task.

    fold A  train Jan-Jun  -> test Jul-Aug
    fold B  train Jan-Jul  -> test Aug-Sep
    fold C  train Jan-Aug  -> test Sep-Oct

Run: uv run python validate_folds.py
"""
import numpy as np
import pandas as pd

from features import city_lookup, corrupt_rate_mask, daily_market, load_raw
from model import LEVEL_MODES, FreightRateModel

tr, va, dec = load_raw()
daily = daily_market(tr, va)
cities = city_lookup(tr, va)
corrupt = corrupt_rate_mask(tr)

FOLDS = {"A Jul-Aug": "2025-07-01", "B Aug-Sep": "2025-08-01", "C Sep-Oct": "2025-09-01"}
rows = []
for fold, cut in FOLDS.items():
    cut = pd.Timestamp(cut)
    train = tr[tr.date < cut]
    tmask = (tr.date >= cut) & (tr.date < cut + pd.DateOffset(months=2)) & ~corrupt
    test = tr[tmask]
    y = test.posted_rate.values
    for mode in LEVEL_MODES:
        m = FreightRateModel(level=mode).fit(train, daily, cities)
        p = m.predict(test)
        e = p - y
        rows.append({"fold": fold, "level": mode, "MAE": np.abs(e).mean(), "MAPE%": 100 * np.mean(np.abs(e) / y),
                     "bias%": 100 * np.mean(e / y), "RMSE": np.sqrt(np.mean(e ** 2))})
    print(f"fold {fold}: {m.describe()}")

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\nPer fold (corrupt labels excluded from the test rows):")
print(res.pivot(index="level", columns="fold", values="MAE").round(2).to_string())
print("\nAverage over the three folds:")
print(res.groupby("level")[["MAE", "RMSE", "MAPE%", "bias%"]].mean().round(3).sort_values("MAE").to_string())
