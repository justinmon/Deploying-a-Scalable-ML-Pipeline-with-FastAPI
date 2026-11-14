# Census income classification with FastAPI

Public repository: https://github.com/justinmon/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

This WGU Machine Learning DevOps project trains a Census income classifier,
evaluates categorical slices, and serves predictions through FastAPI. It is
based on the Udacity starter repository. The intended use is educational;
see [model_card.md](model_card.md) for limitations and ethical considerations.

## Setup

Use Python 3.10, matching the GitHub Actions workflow. Clone this repository
and install the pinned project dependencies in an isolated environment.

```bash
git clone https://github.com/justinmon/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git
cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.10 -m venv .venv`, then
activate it with `.venv\Scripts\Activate.ps1` in PowerShell. The remaining
`python -m ...` commands work in either environment. Gunicorn is not needed
to run the local Uvicorn service.

## Train and evaluate

```bash
python train_model.py
```

The pipeline loads the supplied `data/census.csv`, trims whitespace, removes
24 duplicate complete records, and makes a stratified 80/20 train/test split
with random seed 42. It trains on 26,029 records and evaluates on 6,508
held-out records. Unknown values marked `?` remain explicit categories.
The category encoder and label binarizer are fitted only on training data.

The classifier is a random forest with 200 trees, maximum depth 12, and a
minimum of two samples per leaf. The positive class is `>50K`.

| Held-out metric | Result |
| --- | ---: |
| Precision | 0.805851 |
| Recall | 0.579719 |
| F1 | 0.674332 |

The script saves and reloads the model and both encoders, checks prediction
consistency, and writes all 102 categorical feature-value slices to
[slice_output.txt](slice_output.txt). Each slice includes sample and positive
counts. Undefined ratios use zero; categories without test rows use N/A.
The report is overwritten on each training run, so reruns do not append
stale results.

The committed artifacts are `model/model.pkl`, `model/encoder.pkl`, and
`model/lb.pkl`. Feature order, label mapping, versions, model configuration,
full metrics, and slice records are in `model/metadata.json`. The API can
use these committed artifacts immediately; retraining is optional. Only
load trusted pickle files.

## Tests and continuous integration

```bash
python -m pytest test_ml.py -v
python -m flake8 . --count --show-source --statistics
```

There are 12 passing test cases covering preprocessing, label mapping,
unseen-category inference, training behavior, known metric values, saved
artifact round trips, categorical slice selection, empty slices, and the
compatibility of committed model artifacts. They passed on Windows and in
the Udacity Linux workspace with Python 3.10.

The workflow in `.github/workflows/manual.yml` runs strict flake8 and pytest
on pushes and pull requests to main/master, and supports manual dispatch.
Both checks must pass. See [GitHub Actions](https://github.com/justinmon/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/actions)
for the live run history.

## Run and call the API

From the repository root, start the server in one terminal:

```bash
python -m uvicorn main:app --reload
```

Keep the server running. In a second terminal, activate the same environment,
change to the repository root, and run:

```bash
python local_api.py
```

The client sends a GET to `http://127.0.0.1:8000/` and a POST to
`http://127.0.0.1:8000/data/`. The actual sample response was:

```text
GET Status Code: 200
Welcome: Hello from the Census Income API!
POST Status Code: 200
Prediction: <=50K
```

GET `/` returns a JSON welcome message. POST `/data/` accepts one Census
record with the 14 features shown in `local_api.py`, using hyphenated JSON
names such as `education-num`, and returns `{"result": "<=50K"}` or
`{"result": ">50K"}`. No salary label is supplied as input. The API reuses
the saved encoders and canonical training feature order. Interactive API
documentation is available at `http://127.0.0.1:8000/docs` while it runs.

Live HTTP checks also confirmed that both salary classes match direct model
inference, missing required fields return 422, and an unseen category can
be processed without a server error. The sample prediction is a coursework
output, not a real-world assessment of an individual.

## Project evidence

The screenshots are from this repository's own runs:

- [unit_test.png](screenshots/unit_test.png) shows all 12 tests passing in the Udacity terminal.
- [local_api.png](screenshots/local_api.png) shows the Uvicorn server and successful client GET/POST responses in two terminals.
- [continuous_integration.png](screenshots/continuous_integration.png) shows the passing GitHub Actions checks.

[model_card.md](model_card.md) addresses all sections of the provided
template. [data_notes.md](data_notes.md) records the initial data review.
The original CSV is preserved. Development environments, credentials, and
machine-specific setup files are not included in the repository.
