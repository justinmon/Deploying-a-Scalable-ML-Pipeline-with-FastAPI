"""Train, persist, and evaluate the Census income classifier."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

PROJECT_PATH = Path(__file__).resolve().parent
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def main():
    """Run the full pipeline using an untouched stratified test partition."""
    data_path = PROJECT_PATH / "data" / "census.csv"
    data = pd.read_csv(data_path, skipinitialspace=True)
    data.columns = data.columns.str.strip()
    for column in data.select_dtypes(include="object"):
        data[column] = data[column].str.strip()
    raw_rows = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    train, test = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["salary"]
    )
    print(f"Loaded {raw_rows:,} rows; removed {raw_rows - len(data)} duplicates.")
    print(f"Training rows: {len(train):,} | Test rows: {len(test):,}")

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    if list(lb.classes_) != ["<=50K", ">50K"]:
        raise ValueError("Unexpected target class ordering.")

    model = train_model(X_train, y_train)
    model_directory = PROJECT_PATH / "model"
    for name, artifact in [("model", model), ("encoder", encoder), ("lb", lb)]:
        path = model_directory / f"{name}.pkl"
        save_model(artifact, path)
        print(f"Saved model/{path.name}")

    original_predictions = inference(model, X_test)
    model = load_model(model_directory / "model.pkl")
    encoder = load_model(model_directory / "encoder.pkl")
    lb = load_model(model_directory / "lb.pkl")
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    np.testing.assert_array_equal(original_predictions, inference(model, X_test))
    predictions = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, predictions)
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print("Saved artifacts reloaded: identical held-out predictions.")

    slice_records = []
    with (PROJECT_PATH / "slice_output.txt").open("w", encoding="utf-8") as file:
        file.write("Held-out test metrics; positive label: >50K.\n")
        file.write("Undefined ratios use zero; empty slices use N/A.\n")
        file.write("Small slices are uncertain and do not establish fairness.\n\n")
        for column in CAT_FEATURES:
            # Include every value in the input data, even if absent from test.
            for value in sorted(data[column].unique()):
                subset = test.loc[test[column] == value]
                count = len(subset)
                positive_count = int((subset["salary"] == ">50K").sum())
                file.write(
                    f"{column}: {value}, Count: {count:,}, "
                    f"Positive count: {positive_count:,}\n"
                )
                record = {
                    "feature": column, "value": value, "count": count,
                    "positive_count": positive_count,
                }
                if count:
                    p, r, fb = performance_on_categorical_slice(
                        test, column, value, CAT_FEATURES, "salary", encoder, lb, model
                    )
                    file.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n")
                    record.update(precision=p, recall=r, f1=fb)
                else:
                    file.write(
                        "Precision: N/A | Recall: N/A | F1: N/A (no test rows)\n"
                    )
                    record.update(precision=None, recall=None, f1=None)
                slice_records.append(record)

    metadata = {
        "schema_version": 1,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "raw_rows": raw_rows,
        "duplicate_rows_removed": raw_rows - len(data),
        "unique_rows": len(data),
        "training_rows": len(train),
        "test_rows": len(test),
        "training_positive_rows": int((train["salary"] == ">50K").sum()),
        "test_positive_rows": int((test["salary"] == ">50K").sum()),
        "split": {"test_size": 0.2, "random_state": 42, "stratify": "salary"},
        "feature_columns": data.drop(columns="salary").columns.tolist(),
        "categorical_features": CAT_FEATURES,
        "numeric_features": [
            col for col in data.columns if col not in CAT_FEATURES + ["salary"]
        ],
        "encoded_features": int(X_train.shape[1]),
        "label_classes": lb.classes_.tolist(),
        "model_class": type(model).__name__,
        "model_parameters": model.get_params(),
        "metrics": {"precision": precision, "recall": recall, "f1": f1},
        "confusion_matrix_labels": ["<=50K", ">50K"],
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "slice_count": len(slice_records),
        "slices": slice_records,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        },
    }
    (model_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(slice_records)} categorical slices to slice_output.txt")
    print("Wrote model/metadata.json")


if __name__ == "__main__":
    main()
