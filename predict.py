"""Inference: load the saved model and price every row of a CSV.

The artifact is self-contained: it carries the fitted trees, the city
coordinate lookup and the daily market table, so any file with the training
schema (minus posted_rate) can be scored, including the December inputs that
have no coordinates or market columns.

Examples
  uv run python predict.py --input data/validation.csv --output validation_predictions.csv \
                           --template data/validation_predictions_template.csv
  uv run python predict.py --input data/december_chart_inputs.csv --output data/december_chart_inputs.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/freight_rate_model.joblib")


def load_model(path: Path = MODEL_PATH):
    if not path.is_file():
        raise SystemExit(f"model artifact not found at {path}; run train.py first")
    return joblib.load(path)


def predict_file(model, input_path: Path, output_path: Path, template_path: Path | None = None) -> pd.DataFrame:
    raw = pd.read_csv(input_path, parse_dates=["date"])
    rates = model.predict(raw).round(2)

    if "load_id" in raw.columns:
        # Submission format: load_id,predicted_rate, in template order when a template is given.
        out = pd.DataFrame({"load_id": raw.load_id, "predicted_rate": rates})
        if template_path is not None:
            order = pd.read_csv(template_path)[["load_id"]]
            out = order.merge(out, on="load_id", how="left")
    else:
        # December chart format: the original seven columns with predicted_rate filled in.
        out = pd.read_csv(input_path)
        out["predicted_rate"] = rates

    if out.predicted_rate.isna().any() or (out.predicted_rate <= 0).any():
        raise SystemExit("prediction produced missing or non-positive rates")
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Price every row of a CSV with the trained freight rate model.")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--template", type=Path, default=None, help="optional load_id template that fixes row order")
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    a = p.parse_args()
    out = predict_file(load_model(a.model), a.input, a.output, a.template)
    print(f"wrote {a.output} ({len(out):,} rows, rate ${out.predicted_rate.min():.2f}-{out.predicted_rate.max():.2f})")


if __name__ == "__main__":
    main()
