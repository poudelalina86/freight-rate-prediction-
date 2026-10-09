# Freight rate prediction - common tasks. Requires uv (https://docs.astral.sh/uv/).
.PHONY: setup train score predict test validate notebook eda clean

setup:            ## create the environment
	uv sync --group dev

train:            ## fit on all data, save models/, write both prediction files
	uv run python train.py

score:            ## Spotter's validator + December chart
	uv run python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

predict:          ## inference only, from the saved model
	uv run python predict.py --input data/validation.csv --output validation_predictions.csv --template data/validation_predictions_template.csv
	uv run python predict.py --input data/december_chart_inputs.csv --output data/december_chart_inputs.csv

test:             ## unit + integration tests
	uv run --group dev pytest -q

validate:         ## the three validation experiments (several minutes)
	uv run python validate.py
	uv run python validate_level.py
	uv run python validate_folds.py

eda:              ## exploratory analysis, text output
	uv run python eda.py

notebook:         ## open the visual walkthrough
	uv run --group notebook jupyter lab exploration.ipynb

clean:
	rm -rf __pycache__ tests/__pycache__ .pytest_cache
