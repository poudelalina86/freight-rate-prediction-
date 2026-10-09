"""Exploratory data analysis for the freight rate challenge.

Run: uv run python eda.py
Prints the facts that drive cleaning, feature and validation decisions.
"""
import numpy as np
import pandas as pd

from features import load_raw

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

tr, va, dec = load_raw()

def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")

section("SCHEMA")
print(tr.dtypes.to_string())

section("MISSING VALUES (train | validation)")
m = pd.DataFrame({"train": tr.isna().sum(), "val": va.reindex(columns=tr.columns).isna().sum()})
print(m[m.sum(axis=1) > 0].to_string())

section("DUPLICATES")
print("duplicate load_id:", tr.load_id.duplicated().sum())
feat_cols = [c for c in tr.columns if c not in ("load_id", "posted_rate")]
print("duplicate feature rows (same load, maybe different rate):", tr.duplicated(subset=feat_cols).sum())
print("fully duplicate rows:", tr.duplicated(subset=feat_cols + ["posted_rate"]).sum())

section("TARGET: posted_rate")
tr["rpm"] = tr.posted_rate / tr.distance
print(tr.posted_rate.describe(percentiles=[.001, .01, .05, .5, .95, .99, .999]).round(1).to_string())
print("\nrate per mile:")
print(tr.rpm.describe(percentiles=[.001, .01, .05, .5, .95, .99, .999]).round(3).to_string())
print("\nrpm < 0.5:", (tr.rpm < 0.5).sum(), "| rpm > 6:", (tr.rpm > 6).sum(), "| rpm > 10:", (tr.rpm > 10).sum())
print("\nTop 8 highest rate-per-mile rows:")
print(tr.nlargest(8, "rpm")[["load_id", "pickup", "delivery", "distance", "equipment", "weight", "date", "posted_rate", "rpm"]].to_string(index=False))
print("\nBottom 8 lowest rate-per-mile rows:")
print(tr.nsmallest(8, "rpm")[["load_id", "pickup", "delivery", "distance", "equipment", "weight", "date", "posted_rate", "rpm"]].to_string(index=False))

section("RATE PER MILE BY EQUIPMENT")
print(tr.groupby("equipment").rpm.describe()[["count", "25%", "50%", "75%"]].round(3).to_string())

section("RATE PER MILE VS DISTANCE BUCKET")
tr["dist_bin"] = pd.cut(tr.distance, [0, 150, 300, 500, 800, 1200, 2000, 4000])
print(tr.groupby("dist_bin", observed=True).rpm.median().round(3).to_string())

section("COORDINATES: constant per city?")
pk = tr.groupby("pickup")[["pickup_lat", "pickup_lon"]].agg(["nunique", "std"])
print("pickup cities with >1 distinct lat:", (pk[("pickup_lat", "nunique")] > 1).sum(), "of", len(pk))
print("max within-city lat std:", round(pk[("pickup_lat", "std")].max(), 4), "| lon std:", round(pk[("pickup_lon", "std")].max(), 4))
# do pickup and delivery coords agree for the same city name?
pc = tr.groupby("pickup")[["pickup_lat", "pickup_lon"]].median()
dc = tr.groupby("delivery")[["delivery_lat", "delivery_lon"]].median()
dc.columns = ["pickup_lat", "pickup_lon"]
both = pc.join(dc, rsuffix="_d", how="inner")
print("same city as pickup vs delivery, max |lat diff|:", round((both.pickup_lat - both.pickup_lat_d).abs().max(), 4))
print("cities:", sorted(set(tr.pickup) | set(tr.delivery))[:10], "...")
print("validation cities not in train:", (set(va.pickup) | set(va.delivery)) - (set(tr.pickup) | set(tr.delivery)))

section("DISTANCE vs HAVERSINE of coordinates")
def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))
tr["hav"] = hav(tr.pickup_lat, tr.pickup_lon, tr.delivery_lat, tr.delivery_lon)
ratio = tr.distance / tr.hav
print("distance / straight-line ratio:", ratio.describe(percentiles=[.01, .5, .99]).round(3).to_string())
# is distance constant per lane?
lane = tr.groupby(["pickup", "delivery"]).distance.agg(["nunique", "min", "max", "count"])
print("lanes:", len(lane), "| lanes with >1 distance value:", (lane["nunique"] > 1).sum())
print("max (max-min) distance within a lane:", round((lane["max"] - lane["min"]).max(), 1))

section("WEIGHT")
print(tr.weight.describe(percentiles=[.001, .01, .5, .99, .999]).round(0).to_string())
print("weight <= 0:", (tr.weight <= 0).sum(), "| weight > 48000:", (tr.weight > 48000).sum())
print("median rpm by weight bucket:")
print(tr.groupby(pd.cut(tr.weight, [0, 10000, 20000, 30000, 40000, 50000, 1e9]), observed=True).rpm.median().round(3).to_string())

section("MARKET SIGNALS: daily or per-load?")
for c in ["market_index", "quote_signal"]:
    g = tr.groupby("date")[c].agg(["mean", "std", "nunique"])
    print(f"{c}: overall std={tr[c].std():.4f} | mean within-date std={g['std'].mean():.4f} | mean distinct values per date={g['nunique'].mean():.1f}")
    print(f"  range {tr[c].min():.3f} .. {tr[c].max():.3f} | val range {va[c].min():.3f} .. {va[c].max():.3f}")
print("corr of rpm with market_index:", round(tr[["rpm", "market_index"]].corr().iloc[0, 1], 3), "| with quote_signal:", round(tr[["rpm", "quote_signal"]].corr().iloc[0, 1], 3))
print("corr market_index vs quote_signal:", round(tr[["market_index", "quote_signal"]].corr().iloc[0, 1], 3))
daily = tr.groupby("date").agg(mi=("market_index", "median"), qs=("quote_signal", "median"), rpm=("rpm", "median"))
print("corr of DAILY median rpm with daily market_index:", round(daily[["rpm", "mi"]].corr().iloc[0, 1], 3), "| with quote_signal:", round(daily[["rpm", "qs"]].corr().iloc[0, 1], 3))

section("TIME PATTERNS")
print("loads per day: train", round(len(tr) / tr.date.nunique(), 1), "| val", round(len(va) / va.date.nunique(), 1))
print("monthly median rpm:")
print(tr.groupby(tr.date.dt.to_period("M")).rpm.median().round(3).to_string())
print("day-of-week median rpm (0=Mon):")
print(tr.groupby(tr.date.dt.dayofweek).rpm.median().round(3).to_string())
print("daily median market_index by month (train then validation):")
print(pd.concat([tr, va]).groupby(pd.concat([tr, va]).date.dt.to_period("M")).market_index.median().round(3).to_string())

section("DECEMBER LANE: Lexington -> Fort Wayne, Dry Van")
lane_hist = tr[(tr.pickup == "Lexington") & (tr.delivery == "Fort Wayne")]
print(lane_hist[["date", "equipment", "weight", "distance", "market_index", "quote_signal", "posted_rate", "rpm"]].sort_values("date").to_string(index=False))
print("validation rows on this lane:", ((va.pickup == "Lexington") & (va.delivery == "Fort Wayne")).sum())
print("december inputs:\n", dec.head(3).to_string(index=False))
