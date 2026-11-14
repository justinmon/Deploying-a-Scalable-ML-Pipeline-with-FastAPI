# Census data review

This review covers the supplied `data/census.csv` and the starter
`ml/data.py` for Step 2: Data. The source CSV has not been modified.

## Observed data

- There are 32,561 records, 14 input features, and one target, `salary`.
- The target has 24,720 `<=50K` records (75.92%) and 7,841 `>50K`
  records (24.08%). Report precision, recall, and F1 for `>50K` rather
  than relying on accuracy alone.
- The supplied file already has column names and category strings without
  the leading spaces seen in some versions of the Adult dataset.
- No empty fields were found. The literal `?` represents unknown values:
  1,836 in `workclass`, 1,843 in `occupation`, and 583 in `native-country`.
  A total of 2,399 records contain at least one `?`.
- There are 24 duplicate complete records. Remove these before splitting
  in the training pipeline to avoid placing identical records in both sets.
- Keep `?` as an explicit category so records with unknown information
  remain available for training, prediction, and slice evaluation.

| Categorical feature | Distinct values, including `?` |
| --- | ---: |
| workclass | 9 |
| education | 16 |
| marital-status | 7 |
| occupation | 15 |
| relationship | 6 |
| race | 5 |
| sex | 2 |
| native-country | 42 |

| Numeric feature | Minimum | Maximum |
| --- | ---: | ---: |
| age | 17 | 90 |
| fnlgt | 12,285 | 1,484,705 |
| education-num | 1 | 16 |
| capital-gain | 0 | 99,999 |
| capital-loss | 0 | 4,356 |
| hours-per-week | 1 | 99 |

## How the supplied preprocessing works

`process_data` separates `salary` from the input features. It one-hot encodes
the eight categorical columns and appends those encoded columns to the six
numeric columns. `LabelBinarizer` converts the two salary strings into binary
labels; verify the class order when training and save the fitted binarizer.

In training mode, the function fits the category encoder and label binarizer.
In evaluation or inference mode, it reuses the supplied fitted objects.
`handle_unknown="ignore"` allows a new category at prediction time without
changing the number of features. This does not mean an unseen category has
been learned; its encoded block contains zeros.

`apply_label` converts a single binary prediction into `<=50K` or `>50K`.
Its assumed class order must agree with the saved label binarizer.

## Decisions for the next section

Use a reproducible, stratified 80/20 split after removing duplicate complete
records. Fit all preprocessing only on the training partition and transform
the held-out partition with those fitted objects. Save the model, category
encoder, and label binarizer together with the feature order needed by the API.
Use a tree-based classifier, for which numeric scaling is unnecessary.

Slice evaluation must cover each categorical feature and each of its values.
Report the sample count along with the metrics, including slices with no
positive labels. Small slices have uncertain metrics and do not establish
fairness. The data contain sensitive attributes and historical income
patterns; this coursework model should not be used to determine real
people's access to employment, credit, or services.

This review records the preprocessing decisions made before training.
The completed pipeline, measured performance, and limitations are documented
in `model_card.md`, with detailed results in `model/metadata.json` and
`slice_output.txt`. Run instructions are in `README.md`.

The course links to the [UCI Census Income dataset reference](https://archive.ics.uci.edu/ml/datasets/census+income).
The counts above were measured directly from the supplied project file on
November 14, 2026. Its SHA-256 is
`0D2B143D27CB24094CE1A01AA8CF642E49A9196519CD32795E71E8EE71AE72E8`.
