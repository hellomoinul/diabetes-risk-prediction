# Scoring New Data (Exam Runbook)

Score a surprise CSV with the already-trained diabetes model. No retraining, no
Kaggle credentials, ~seconds to run.

## The 30-second version

```bash
python predict.py surprise.csv
```

That is the whole exam command. It writes `surprise_predictions.csv` (input rows plus
`prediction` and `probability`) and prints how many patients were flagged.

## Pick a model

```bash
python predict.py surprise.csv --model xgboost              # default, best AUC (0.9796 CV)
python predict.py surprise.csv --model decision_tree        # most interpretable
python predict.py surprise.csv --model logistic_regression
python predict.py surprise.csv --model svm
python predict.py surprise.csv --out scored.csv             # choose the output path
```

Available models are whatever `*.joblib` files sit in `models/`. If the directory is
empty, run `python run_pipeline.py` first (~3 minutes) to rebuild them.

## Reading the output

The output CSV adds two columns:

| Column | Meaning |
|---|---|
| `prediction` | 0/1 label. `1` = flagged diabetic. |
| `probability` | Positive-class score 0–1 (XGBoost, LogReg, tree). For SVM it is `decision_score` instead — a ranking score, not a probability. |

Below the counts, if the CSV also carries the true target column (default name
`diabetes`), the script prints a metric block:

```
Metrics on the supplied labels:
  ROC-AUC   ...
  Accuracy  ...     <- always read against the baseline below, never alone
  Precision ...
  Recall    ...     <- the column that matters for screening (missed diabetics)
  F1-Score  ...
  TP ...  FP ...  TN ...  FN ...
```

XGBoost's test-set figures for reference: ROC-AUC 0.9799, Accuracy 0.9034,
Precision 0.4756, Recall 0.9277, F1 0.6288. A surprise slice will land near these
if it is drawn from the same distribution; a wildly different number means the
new data is distributed differently, not necessarily that the model is broken.

## If the teacher's columns differ

The script prints `Expected features (...)` and `Found columns (...)` before scoring.

**Same meaning, different name** — map it:

```bash
python predict.py exam.csv --map glucose:blood_glucose_level
python predict.py exam.csv --map age_years:age --map glucose:blood_glucose_level
```

**Genuinely different features** — the saved model cannot be used, full stop. Its
learned weights are tied to its exact inputs. Retrain instead:

```bash
python run_on_new_data.py --csv exam.csv --mode retrain
```

That runs the same 4-model workflow on the new data and writes fresh models,
tables and figures under `results_new/`. Or let it decide automatically:

```bash
python run_on_new_data.py --csv exam.csv        # auto: predict if columns match, else retrain
```

## Before the exam

```bash
python selftest_predict.py     # ~30 s. Proves all 4 models load and score.
```

It scores a labeled 5,000-row slice, prints ROC-AUC, and checks `--map` equality.
It must exit `SELF-TEST PASSED`. This is a functional check only — it does not
reproduce the 0.9799 test figure, which comes from the held-out split in
`run_pipeline.py`.

## What this does not do

The model **keeps** its knowledge when scoring and **replaces** it when retraining.
It never does both at once: there is no continual/incremental learning
(no `partial_fit`, no weight warm-starting). Retraining starts from zero.
