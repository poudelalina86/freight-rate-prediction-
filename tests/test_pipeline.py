"""Unit and integration tests for the freight rate pipeline.

Run: uv run --group dev pytest
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # scripts use repository-relative paths

from features import (DECEMBER_PATH, TEMPLATE_PATH, VALIDATION_PATH, city_lookup, clean_weight,  # noqa: E402
                      corrupt_rate_mask, daily_market, load_raw)
from model import SHAPE_FEATURES, FreightRateModel, _prep  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return load_raw()


def test_clean_weight_makes_negatives_positive():
    df = pd.DataFrame({"weight": [-36485.0, 20000.0, np.nan]})
    out = clean_weight(df)
    assert out.weight.tolist()[:2] == [36485.0, 20000.0]
    assert np.isnan(out.weight.iloc[2])
    assert df.weight.iloc[0] == -36485.0, "input frame must not be mutated"


def test_corrupt_rate_mask_flags_only_implausible_rates():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "distance": rng.uniform(200, 280, n),
        "equipment": "Dry Van",
    })
    df["posted_rate"] = df.distance * rng.normal(2.1, 0.1, n)
    df.loc[0, "posted_rate"] *= 4.0     # 4x corruption
    df.loc[1, "posted_rate"] *= 0.2     # 0.2x corruption
    mask = corrupt_rate_mask(df)
    assert mask.iloc[0] and mask.iloc[1]
    assert mask.iloc[2:].sum() == 0


def test_daily_market_covers_every_day_through_december(raw):
    tr, va, _ = raw
    daily = daily_market(tr, va)
    assert daily.index.min() == pd.Timestamp("2025-01-01")
    assert daily.index.max() == pd.Timestamp("2025-12-31")
    assert len(daily) == 365
    assert not daily.isna().any().any()


def test_make_features_rebuilds_coordinates_for_december_inputs(raw):
    tr, va, dec = raw
    f = _prep(dec, daily_market(tr, va), city_lookup(tr, va))   # make_features + model-level deviations
    assert set(SHAPE_FEATURES) <= set(f.columns)
    assert f[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].notna().all().all()
    assert f[["mi_daily", "market_index"]].notna().all().all()


def test_city_lookup_has_one_coordinate_pair_per_city(raw):
    tr, va, _ = raw
    cities = city_lookup(tr, va)
    assert cities.index.is_unique
    assert len(cities) == 72          # 64 training cities + 8 validation-only cities


def test_model_fits_and_predicts_sensible_rates(raw):
    tr, va, _ = raw
    small = tr.head(4000)
    m = FreightRateModel(level="none", gbm_params={"max_iter": 60}).fit(small, daily_market(tr, va), city_lookup(tr, va))
    p = m.predict(va.head(500))
    assert np.isfinite(p).all() and (p > 0).all()
    rpm = p / va.head(500).distance.values
    assert 1.0 < np.median(rpm) < 4.0


def test_predict_file_formats(tmp_path):
    artifact = ROOT / "models" / "freight_rate_model.joblib"
    if not artifact.exists():
        pytest.skip("run train.py first to create the model artifact")
    from predict import load_model, predict_file
    model = load_model(artifact)

    out = predict_file(model, Path(VALIDATION_PATH), tmp_path / "val.csv", Path(TEMPLATE_PATH))
    assert list(out.columns) == ["load_id", "predicted_rate"]
    assert len(out) == 12_000 and out.load_id.is_unique
    assert (out.load_id.values == pd.read_csv(TEMPLATE_PATH).load_id.values).all()
    assert (out.predicted_rate > 0).all()

    dec = predict_file(model, Path(DECEMBER_PATH), tmp_path / "dec.csv")
    assert list(dec.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    assert len(dec) == 31 and (dec.predicted_rate > 0).all()
