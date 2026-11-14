"""Serve the saved Census income model through FastAPI."""

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


MODEL_DIRECTORY = Path(__file__).resolve().parent / "model"
encoder = load_model(MODEL_DIRECTORY / "encoder.pkl")
model = load_model(MODEL_DIRECTORY / "model.pkl")
lb = load_model(MODEL_DIRECTORY / "lb.pkl")
metadata = json.loads((MODEL_DIRECTORY / "metadata.json").read_text())

app = FastAPI(title="Census Income Coursework API", version="1.0.0")


@app.get("/")
async def get_root():
    """Return the API welcome message."""
    return {"message": "Hello from the Census Income API!"}


@app.post("/data/")
async def post_inference(data: Data):
    """Predict one salary category with the saved training artifacts."""
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens; Python variable names use underscores.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)
    # Match the original numeric and categorical feature ordering explicitly.
    data = data.loc[:, metadata["feature_columns"]]

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data_processed, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    prediction = inference(model, data_processed)
    return {"result": apply_label(prediction)}
