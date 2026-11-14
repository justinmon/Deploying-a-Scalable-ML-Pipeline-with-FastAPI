# Model Card

## Model Details

This coursework model is a scikit-learn 1.5.1 RandomForestClassifier that
predicts whether the salary label in the supplied Census dataset is `<=50K`
or `>50K`. It was trained on November 14, 2026, using Python 3.10.11,
pandas 2.2.2, and NumPy 2.2.6. The forest uses 200 trees, a maximum tree
depth of 12, a minimum of two samples per leaf, and random seed 42. These
settings were selected before evaluation; hyperparameter tuning was not
performed on the test set.

The model uses 14 input features. Eight categorical features are one-hot
encoded and concatenated with six numeric features, giving 108 processed
features. Preprocessing is fitted only on the training partition. The
saved category encoder ignores unseen categories by encoding their blocks
as zeros. The saved label binarizer maps `<=50K` to 0 and `>50K` to 1.
Predictions use the classifier's default decision rule.

The reproducible entry point is `python train_model.py`. The pipeline saves
`model/model.pkl`, `model/encoder.pkl`, `model/lb.pkl`, and
`model/metadata.json`. It reloads the three pickle artifacts and verifies
that their held-out predictions match the original model's predictions.
Only trusted pickle artifacts should be loaded.

## Intended Use

The intended users are the student and project reviewers. This model
demonstrates a reproducible training pipeline, categorical slice evaluation,
and an inference service for an educational assignment. It is not validated
for real-world decisions about individuals, including employment, lending,
insurance, eligibility for services, or compensation.

## Training Data

The input is the project's supplied `data/census.csv`, which the course links
to the [UCI Census Income dataset](https://archive.ics.uci.edu/ml/datasets/census+income).
The supplied file contains 32,561 rows and 15 columns, including the target.
Whitespace is stripped during loading. Removing 24 duplicate complete
records before splitting leaves 32,537 unique records. The original CSV is
preserved, and its SHA-256 is recorded in `model/metadata.json`.

The pipeline uses a stratified 80/20 split with random seed 42. The training
partition contains 26,029 rows, including 6,271 `>50K` examples. The literal
`?` is retained as an explicit unknown category instead of dropping records
with missing information. Before deduplication, it appears 1,836 times in
workclass, 1,843 times in occupation, and 583 times in native-country.

The categorical features are workclass, education, marital-status,
occupation, relationship, race, sex, and native-country. The numeric
features are age, fnlgt, education-num, capital-gain, capital-loss, and
hours-per-week. Numeric scaling is not applied to this tree-based model.
The fnlgt column is used as an input feature, rather than a sample weight;
training and evaluation results are unweighted.

## Evaluation Data

The held-out partition contains 6,508 rows, including 1,568 `>50K` examples
and 4,940 `<=50K` examples. No held-out records are used to fit the model
or encoders. Deduplication before splitting prevents identical complete
records from appearing in both partitions. Evaluation uses the training
encoder and binarizer unchanged.

This is one stratified random split from the same supplied dataset, rather
than an independent external or temporal evaluation. It does not establish
performance for present-day populations or new data sources.

## Metrics

Precision, recall, and F1 are measured with `>50K` as the positive class.
Precision is the fraction of predicted positive examples that are correct.
Recall is the fraction of actual positive examples that are identified.
F1 is their harmonic mean.

| Held-out metric | Value |
| --- | ---: |
| Precision | 0.805851 |
| Recall | 0.579719 |
| F1 | 0.674332 |

The confusion matrix contains 4,721 true negatives, 219 false positives,
659 false negatives, and 909 true positives. The model therefore misses
about 42% of the positive examples despite its higher precision.

`slice_output.txt` reports precision, recall, F1, sample counts, and positive
counts for every distinct value of every categorical feature, including
unknown values. There are 102 feature-value slices in total. Ratios with
zero denominators return zero, rather than an artificial perfect score.
The pipeline reports N/A if a category has no held-out records. Such slices
cannot be evaluated. A zero score for a slice with no positive examples
should not be interpreted as evidence of poor or good positive-class recall.

The complete unrounded metrics, model configuration, and slice records are
saved in `model/metadata.json`. Slice metrics are descriptive diagnostics;
they do not certify fairness or causal differences between groups.

## Ethical Considerations

The dataset includes sensitive demographic attributes, including race and
sex. Learning historical income labels can reproduce social and economic
inequalities, and other features can serve as proxies for sensitive traits.
Different error rates across groups can cause unequal harm if predictions
are used in consequential decisions. Retaining these features for the
coursework does not justify using them in a deployed decision system.

Small or underrepresented groups may have unstable metrics. Evaluating
one feature at a time does not address intersectional disparities, and
unknown categories may conceal differences in data collection. Anonymized
demographic records still deserve responsible handling. This repository
uses only the dataset supplied for the assignment and collects no new
personal information.

## Caveats and Recommendations

The target is a binary income category, not an estimate of a person's
current salary, financial need, abilities, or value. Class imbalance makes
accuracy alone an inadequate assessment. The measured recall is limited,
and the default decision rule has not been optimized for any real use case.

Before considering any use beyond coursework, obtain representative data,
define the costs of false positives and false negatives, evaluate multiple
splits or external data, and examine group and intersectional uncertainty.
Unseen categories are accepted technically, but their reliability has not
been established. Monitor input drift and performance if a future system
is developed. Do not infer that removing sensitive features alone would
eliminate unfairness. Any consequential use would require a separate
assessment of suitability, fairness, and governance.
