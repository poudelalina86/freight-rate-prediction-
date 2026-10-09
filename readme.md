# Freight Rate Prediction – Spotter ML Assessment

**Author:** Alina Chudali
Predicts the posted rate of a truckload shipment. Trained on 48,000 labelled loads
(January–October 2025), the model prices the 12,000 validation loads (November–December 2025)
and a fixed Lexington → Fort Wayne lane for every day of December 2025.

- Original brief: [ASSESSMENT_INSTRUCTIONS.md](ASSESSMENT_INSTRUCTIONS.md) and `freight-rate-ml-assessment.pdf`
- Report: [report.docx](report.docx) (also as [report.md](report.md))
- Visual walkthrough: [exploration.ipynb](exploration.ipynb)

## Deliverables

| Brief asks for | Where it is |
|---|---|
| Code, dependencies, run instructions | This repository: `pyproject.toml` + `uv.lock` (or `requirements.txt`), this README, `Makefile` |
| `validation_predictions.csv` with `load_id,predicted_rate` | [validation_predictions.csv](validation_predictions.csv), 12,000 rows |
| `data/december_chart_inputs.csv` with `predicted_rate` filled | [data/december_chart_inputs.csv](data/december_chart_inputs.csv) |
| Report with validation approach, data split and the December chart | [report.docx](report.docx) |
| 2–3 minute Loom | _link to be added on submission_ |

## Results

Rolling-origin validation: train on the first *N* months, predict the next two, for three cutoffs
(June, July, August). Errors on clean test labels.

| Model | Mean MAE | MAPE | Bias |
|---|---|---|---|
| Median $/mile by equipment × distance bucket (baseline, Sep–Oct fold) | $83 | 3.7% | – |
| **Final model** (gradient boosting on log rate, no time-level term) | **$50** | **2.1%** | -0.3% |
| Same model + best market-index level forecast | $70 | 3.0% | +1.8% |

![December 2025 predictions for the fixed Lexington → Fort Wayne lane](scorer_results/candidate_december.png)

The December curve is the weekly cycle the model learned (Thursday high, Sunday low) around the lane's
recent level of about $820. See the report, section 5, for why no further December movement is predicted.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Tested on Python 3.12 (uv) and 3.14 (pip).

```bash
make setup      # uv sync
make train      # fit on all data, save models/, write validation_predictions.csv, fill data/december_chart_inputs.csv
make score      # Spotter's score.py: validates both files, draws scorer_results/candidate_december.png
make test       # pytest: cleaning rules, feature pipeline, model, output formats
```

The same without make:

```bash
uv sync
uv run python train.py
uv run python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
uv run --group dev pytest
```

Inference only, from the saved model (`make predict`):

```bash
uv run python predict.py --input data/validation.csv --output validation_predictions.csv \
                         --template data/validation_predictions_template.csv
```

Analysis and experiments, each prints its findings:

```bash
make eda          # uv run python eda.py            exploratory analysis, text
make validate     # validate.py, validate_level.py, validate_folds.py   (several minutes)
make notebook     # jupyter lab exploration.ipynb   charts; re-runs the folds, ~3 minutes
```

Without uv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py && python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Repository layout

```
data/                         inputs supplied with the brief (december_chart_inputs.csv has predicted_rate filled)
features.py                   all cleaning and feature engineering, shared by every script
model.py                      FreightRateModel: gradient-boosting shape model + optional daily level model
train.py                      final fit on all clean data, saves the artifact, writes both prediction files
predict.py                    inference: loads the artifact and prices any CSV with the input schema
eda.py                        exploratory analysis (text)
validate.py                   experiment 1: single time split, feature and target choices
validate_level.py             experiment 2: how to forecast the daily rate level
validate_folds.py             experiment 3: rolling-origin validation, picks the final design
exploration.ipynb             visual walkthrough; imports from the modules, no logic of its own
tests/                        pytest suite
score.py                      provided by Spotter; validates the output files and draws the chart
models/freight_rate_model.joblib   trained model, self-contained (trees + city lookup + daily market table)
validation_predictions.csv    submission file
scorer_results/               chart produced by score.py
report.docx, report.md        validation report
```

## Approach in one paragraph

Rates are almost entirely explained by distance, with multiplicative effects for equipment, weight,
geography and day of week, plus a slowly moving market level. A `HistGradientBoostingRegressor` on
`log(rate)` learns the load effects from coordinates, not city names, because 8 validation cities never
appear in training. About 1.4% of training labels are corrupted (rates at ~0.2× or 3–5× the plausible
value) and are dropped; negative weights are sign flips and are made positive; missing market index is
filled from the daily median. Validation is strictly forward in time. It showed that the market index,
although strongly correlated with rates in-sample, hurts out-of-sample forecasts because its relationship
to rates shifted in late summer, so the final model carries no time-level term.

## Known limitations

- Week-scale rate bumps of 1–2% are not predictable from the inputs provided.
- Late-December holiday effects are absent from the January–October training window.
- About 1% of the hidden validation labels are probably corrupted in the same way as the training labels, which caps any achievable score.
# freight-rate-prediction-
