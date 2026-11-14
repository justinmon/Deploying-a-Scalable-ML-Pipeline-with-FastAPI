"""Reusable training, prediction, persistence, and slice evaluation functions."""

import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train, y_train):
    """Fit a reproducible random forest on preprocessed training data.

    Restrict tree depth and leaf size to limit model size and overfitting.
    Two workers keep training suitable for the course workspace.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """Return precision, recall, and F1 for the positive salary class (>50K).

    Undefined ratios return zero instead of implying perfect performance.
    Callers should also report slice counts when interpreting these metrics.
    """
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=0)
    return precision, recall, fbeta


def inference(model, X):
    """Predict binary salary labels for preprocessed input rows."""
    return model.predict(X)


def save_model(model, path):
    """Save a model or fitted encoder, creating its parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path):
    """Load a trusted pickle artifact produced by this project's pipeline."""
    with Path(path).open("rb") as file:
        return pickle.load(file)


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """Evaluate rows where a categorical feature equals the requested value.

    Reuse the training encoders to preserve the trained feature representation.
    Raise ValueError for an empty slice, whose metrics cannot be measured.
    """
    subset = data.loc[data[column_name] == slice_value]
    if subset.empty:
        raise ValueError(f"No rows for {column_name}={slice_value!r}")

    X_slice, y_slice, _, _ = process_data(
        subset,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_slice)
    return compute_model_metrics(y_slice, preds)
