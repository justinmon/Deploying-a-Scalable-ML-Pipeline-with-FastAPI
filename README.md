# Census income classification with FastAPI

[Public project repository](https://github.com/justinmon/Deploying-a-Scalable-ML-Pipeline-with-FastAPI)

WGU Machine Learning DevOps coursework based on the Udacity starter. This project is being completed section by section.

## Environment and checks

Use Python 3.10 for development and GitHub Actions. Create an isolated environment and install `requirements.txt`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m flake8 .
python -m pytest -v
```

The GitHub workflow runs flake8 and pytest on pushes and pull requests to main/master, with manual dispatch available. Lint failures fail the workflow. The inherited model and API functions are still incomplete, and the three inherited tests are placeholders; passing those placeholders is not evidence of completed model tests. The final passing workflow screenshot will be captured after implementation.

## Remaining project sections

- Inspect and clean the supplied Census dataset; review `ml/data.py`.
- Implement model training, saving/loading, inference, classification metrics, and slice metrics.
- Complete `train_model.py`, save model and encoder artifacts, and write `slice_output.txt` for every value of every categorical feature.
- Write meaningful tests and complete every section of `model_card.md` using actual results.
- Implement GET `/` and POST inference on a separate path; run GET and POST from `local_api.py`.
- Capture actual `screenshots/unit_test.png`, `screenshots/local_api.png`, and `screenshots/continuous_integration.png` from this project.

The project is not ready for submission yet. This README will be updated with verified results and final run instructions as sections are completed.
