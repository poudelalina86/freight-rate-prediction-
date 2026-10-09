"""Train the final model on all labelled data, save it, and write the submission files.

Outputs
  models/freight_rate_model.joblib   the fitted model (self-contained; see predict.py)
  validation_predictions.csv         load_id,predicted_rate for the 12,000 validation loads
  data/december_chart_inputs.csv     predicted_rate column filled in place, as the brief requires
  model_summary.txt                  what was fitted, for the report

Run: uv run python train.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import DECEMBER_PATH, TEMPLATE_PATH, VALIDATION_PATH, city_lookup, corrupt_rate_mask, daily_market, load_raw
from model import SHAPE_FEATURES, FreightRateModel
from predict import MODEL_PATH, predict_file

tr, va, dec = load_raw()
daily = daily_market(tr, va)          # covers Jan-Dec: validation.csv carries the Nov-Dec market index
cities = city_lookup(tr, va)          # covers all 72 cities, including the 8 absent from training

model = FreightRateModel(level="none").fit(tr, daily, cities)
print("Fitted:", model.describe())

# ---- save the artifact -------------------------------------------------------
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

# ---- predictions, through the same code path as predict.py ----------------
out = predict_file(model, Path(VALIDATION_PATH), Path("validation_predictions.csv"), Path(TEMPLATE_PATH))
print(f"wrote validation_predictions.csv ({len(out):,} rows)")
dec_out = predict_file(model, Path(DECEMBER_PATH), Path(DECEMBER_PATH))   # filled in place, as the brief asks
print(f"filled predicted_rate in {DECEMBER_PATH} (31 rows)")
va_pred = va[["load_id"]].merge(out, on="load_id").predicted_rate.values
dec_pred = dec_out.predicted_rate.values

# ---- sanity checks -----------------------------------------------------------
clean = tr[~corrupt_rate_mask(tr)]
print("\nRate per mile, train (clean) vs predicted validation:")
print(pd.DataFrame({"train": (clean.posted_rate / clean.distance).describe(), "val_pred": (va_pred / va.distance).describe()}).round(3).T.to_string())
print("\nPredicted $/mile by equipment vs train:")
print(pd.DataFrame({"train": (clean.posted_rate / clean.distance).groupby(clean.equipment).median(),
                    "val_pred": pd.Series(va_pred / va.distance.values).groupby(va.equipment.values).median()}).round(3).to_string())
new = (set(va.pickup) | set(va.delivery)) - (set(tr.pickup) | set(tr.delivery))
touch = va.pickup.isin(new) | va.delivery.isin(new)
print(f"\nPredicted $/mile: rows with unseen cities {np.median(va_pred[touch]/va.distance[touch]):.3f} vs others {np.median(va_pred[~touch]/va.distance[~touch]):.3f}")

lane = clean[(clean.pickup == "Lexington") & (clean.delivery == "Fort Wayne") & (clean.equipment == "Dry Van")]
print(f"\nLexington->Fort Wayne Dry Van history: {len(lane)} loads, rate {lane.posted_rate.min():.0f}-{lane.posted_rate.max():.0f}, "
      f"Sep-Oct mean {lane[lane.date >= '2025-09-01'].posted_rate.mean():.0f}")
print(f"December predictions: {dec_pred.min():.2f}-{dec_pred.max():.2f}, mean {dec_pred.mean():.2f}")
print(dec_out[["date", "predicted_rate"]].T.to_string(header=False))

with open("model_summary.txt", "w") as fh:
    fh.write(model.describe() + "\n")
    fh.write("shape features: " + ", ".join(SHAPE_FEATURES) + "\n")
    fh.write(f"training rows used: {len(clean):,} of {len(tr):,}\n")
