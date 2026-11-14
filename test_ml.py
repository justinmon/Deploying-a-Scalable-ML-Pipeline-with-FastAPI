"""Behavioral tests for Census preprocessing and model functions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from ml.data import apply_label, process_data
from ml.model import (
    compute_model_metrics, inference, load_model,
    performance_on_categorical_slice, save_model, train_model,
)
from train_model import CAT_FEATURES

ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def census_example():
    """Fit preprocessing on a small, disjoint training sample for fast tests."""
    data = pd.read_csv(ROOT / "data/census.csv").drop_duplicates().head(2000)
    train, test = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["salary"]
    )
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label="salary", training=False,
        encoder=encoder, lb=lb,
    )
    return train, test, X_train, y_train, X_test, y_test, encoder, lb


@pytest.fixture(scope="module")
def fitted_model(census_example):
    """Train once per session instead of once per individual test."""
    return train_model(census_example[2], census_example[3])


def test_preprocessing_shape_and_labels(census_example):
    """Encoding preserves row alignment and the correct binary label map."""
    train, test, X_train, y_train, X_test, y_test, encoder, lb = census_example
    assert set(train.index).isdisjoint(test.index)
    width = 6 + sum(len(values) for values in encoder.categories_)
    assert X_train.shape == (len(train), width)
    assert X_test.shape == (len(test), width)
    assert np.isfinite(X_train).all() and np.isfinite(X_test).all()
    assert list(lb.classes_) == ["<=50K", ">50K"]
    np.testing.assert_array_equal(y_train, (train["salary"] == ">50K").astype(int))
    np.testing.assert_array_equal(y_test, (test["salary"] == ">50K").astype(int))


def test_unseen_category_without_salary(census_example):
    """Unlabeled inference accepts unseen categories without changing width."""
    _, test, X_train, _, _, _, encoder, lb = census_example
    row = test.drop(columns="salary").iloc[[0]].copy()
    row.loc[:, "workclass"] = "category-not-seen-in-training"
    encoded, labels, returned_encoder, returned_lb = process_data(
        row, categorical_features=CAT_FEATURES, training=False, encoder=encoder,
        lb=lb,
    )
    assert encoded.shape == (1, X_train.shape[1])
    assert np.isfinite(encoded).all() and labels.size == 0
    assert returned_encoder is encoder and returned_lb is lb
    assert "category-not-seen-in-training" not in encoder.categories_[0]


def test_training_learns_separable_signal():
    """Training learns a clear signal and predicts correct labels on new rows."""
    X = np.concatenate([np.arange(-20, 0), np.arange(1, 21)]).reshape(-1, 1)
    y = np.array([0] * 20 + [1] * 20)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    np.testing.assert_array_equal(inference(model, [[-10.5], [10.5]]), [0, 1])


@pytest.mark.parametrize(
    "truth,predictions,expected",
    [
        ([1, 1, 0, 0], [1, 0, 1, 0], (0.5, 0.5, 0.5)),
        ([1, 1, 0, 0], [1, 0, 0, 0], (1.0, 0.5, 2 / 3)),
        ([0, 0], [0, 0], (0.0, 0.0, 0.0)),
    ],
    ids=["balanced-errors", "missed-positive", "no-positives"],
)
def test_metrics_known_values(truth, predictions, expected):
    """Scores agree with hand calculations, including zero-denominator cases."""
    assert compute_model_metrics(truth, predictions) == pytest.approx(expected)


@pytest.mark.parametrize("binary,label", [(0, "<=50K"), (1, ">50K")])
def test_apply_label(binary, label):
    """Model outputs map to the exact salary strings required by the API."""
    assert apply_label(np.array([binary])) == label


def test_artifact_round_trip(census_example, fitted_model, tmp_path):
    """Reloading the model and both encoders preserves predictions."""
    _, test, _, _, X_test, y_test, encoder, lb = census_example
    paths = {name: tmp_path / "artifacts" / f"{name}.pkl"
             for name in ["model", "encoder", "lb"]}
    for name, artifact in [("model", fitted_model), ("encoder", encoder), ("lb", lb)]:
        save_model(artifact, paths[name])
    restored_X, restored_y, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label="salary", training=False,
        encoder=load_model(paths["encoder"]), lb=load_model(paths["lb"]),
    )
    np.testing.assert_array_equal(restored_y, y_test)
    np.testing.assert_array_equal(
        inference(load_model(paths["model"]), restored_X),
        inference(fitted_model, X_test),
    )


def test_slice_uses_only_requested_rows(census_example, fitted_model):
    """Slice scores match independently filtered full-model predictions."""
    _, test, _, _, X_test, y_test, encoder, lb = census_example
    predictions = inference(fitted_model, X_test)
    for value in test["sex"].unique():
        mask = (test["sex"] == value).to_numpy()
        expected = precision_recall_fscore_support(
            y_test[mask], predictions[mask], average="binary", zero_division=0
        )[:3]
        actual = performance_on_categorical_slice(
            test, "sex", value, CAT_FEATURES, "salary", encoder, lb, fitted_model
        )
        assert actual == pytest.approx(expected)


def test_empty_slice_has_no_measured_metrics(census_example, fitted_model):
    """An absent category raises an error instead of inventing a score."""
    _, test, _, _, _, _, encoder, lb = census_example
    with pytest.raises(ValueError, match="No rows"):
        performance_on_categorical_slice(
            test, "sex", "absent-category", CAT_FEATURES, "salary", encoder, lb,
            fitted_model,
        )


def test_saved_pipeline_artifacts_are_compatible():
    """Committed model, encoders, and feature schema support real inference."""
    metadata = json.loads((ROOT / "model/metadata.json").read_text())
    row = pd.read_csv(ROOT / "data/census.csv").iloc[[0]]
    features, labels, _, _ = process_data(
        row, categorical_features=metadata["categorical_features"], label="salary",
        training=False, encoder=load_model(ROOT / "model/encoder.pkl"),
        lb=load_model(ROOT / "model/lb.pkl"),
    )
    model = load_model(ROOT / "model/model.pkl")
    assert features.shape[1] == model.n_features_in_ == metadata["encoded_features"]
    assert labels.shape == (1,)
    assert apply_label(inference(model, features)) in {"<=50K", ">50K"}
