# Diabetes Risk Prediction

**Course:** Introduction to Machine Learning Lab
**Dataset:** [Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
(Kaggle, uploaded by Mohammed Mustafa), fetched programmatically with the official Kaggle API.
**Notebook:** [`notebooks/Diabetes Risk Prediction.ipynb`](notebooks/Diabetes%20Risk%20Prediction.ipynb)

---

## 0. Why these models

The candidate pool is **Linear Regression, Decision Tree, SVM, Logistic Regression, KNN and
XGBoost**. Each is assessed *before* any pipeline is built, so the shortlist is a decision
rather than an arbitrary pick.

| Candidate | Include? | Reasoning |
|---|---|---|
| **Linear Regression** | Excluded | Built for continuous targets, not class labels. On a 0/1 diagnosis outcome it violates its own assumptions (unbounded output, Gaussian errors) and gives no probabilistic class interpretation. Wrong tool for classification. |
| **Logistic Regression** (`Log St`) | **Included** | The natural discriminative baseline for binary classification. Emits calibrated probabilities, its coefficients map directly onto clinical risk factors, and it is cheap. The right first assumption for clinical data that is roughly linearly separable. |
| **Decision Tree** (`DT`) | **Included** | Captures non-linearity and feature interactions (e.g. "high glucose AND high BMI") without needing scaling, and its rule structure is readable by a clinician who needs to know *why* a patient was flagged. Natural stepping stone to XGBoost. |
| **SVM** | **Included** | Clinical data is rarely linearly separable in raw feature space. The RBF kernel maps features into a higher dimension for a cleaner boundary, and works well on small-to-medium tabular medical data. |
| **KNN** | Excluded | Vulnerable to the curse of dimensionality and to feature scaling, slow at inference, and it yields no model artifact — it memorises the training set, leaving nothing interpretable for a clinician. |
| **XGBoost** | **Included** | A gradient-boosted ensemble, typically strongest on structured tabular data. Handles non-linearity, interactions and missing values well, and supplies importance scores that restore some interpretability. Carried as the performance benchmark. |

**Shortlist: `Log St`, `DT`, `SVM`, `XGBoost`** — spanning simple and interpretable → interpretable
and non-linear → kernel-based → high-performance ensemble, so the four results are genuinely
comparable.

---

## 1. Data set

| Property | Value |
|---|---|
| Source | Kaggle, `iammustafatz/diabetes-prediction-dataset` |
| Raw rows | 100,000 |
| Columns | 9 (8 features + 1 target) |
| Numeric features | `age`, `hypertension`, `heart_disease`, `bmi`, `HbA1c_level`, `blood_glucose_level` |
| Categorical features | `gender`, `smoking_history` |
| Target | `diabetes` (1 = positive) |
| Class balance | 91,500 negative / 8,500 positive (8.5% positive) |

`src/fetch_data.py` downloads via the official Kaggle API on first run, then validates schema
and row count on every run. A truncated download or changed schema fails loudly rather than
silently producing wrong numbers.

---

## 2. Data exploration and cleaning

| Finding | Decision |
|---|---|
| 3,854 full-row duplicates (3.85%) | **Dropped.** Only one duplicate group disagrees on the label, so nothing is lost. → 96,146 rows. |
| `isnull()` reports zero missing | **Misleading.** 35,816 rows hide missingness in `smoking_history` as the string `"No Info"`. |
| `"No Info"` diabetes rate: 4.06% vs 9.53–17.00% for real smoking categories | **Kept as its own category.** Missingness this strongly correlated with the target is systematic, not random. Imputing it would discard real signal and would tell a doctor these 35,816 patients never smoked. |
| `gender == "Other"` (18 rows, 0% positive) | **Kept.** One-hot encoded. |
| Physiologically impossible values: min age 0.08 yrs, 17,219 patients under 18, 911 infants under 1 | **Reported, not corrected.** See §8.3 — blanking them would discard 17% of rows without fixing anything real. |
| `hypertension` / `heart_disease` present | Diabetes rate more than quadruples. |

Figures: `01_numeric_distributions`, **`01b_numeric_boxplots`**, `02_class_imbalance`,
`03_correlation_heatmap`, `04_category_diabetes_rates`, `05_missingness_sentinel`.

---

## 3. Data wrangling

Two paths, as every step is a fitted estimator inside the `ColumnTransformer` so training
statistics cannot leak into validation or test.

| Path | Steps | Why |
|---|---|---|
| Numeric | median imputation → `StandardScaler` | `bmi` and `blood_glucose_level` are right-skewed, so the median resists outliers better. Scaling is required by Logistic Regression and SVM. |
| Categorical | most-frequent imputation → one-hot | Keeps categories discrete; no false ordering. One-hot columns are **not** scaled. |

**Split:** stratified 70 / 15 / 15 → 67,302 / 14,422 / 14,422, with 1,272 positives in test.

**Majority-class baseline: 0.9118 accuracy, recall 0.** This is the number every model must be
judged against.

**Class imbalance (8.82% positive): resampling was deliberately not used.** Each model carries
a class-imbalance correction instead — `class_weight='balanced'` for scikit-learn models,
`scale_pos_weight=10.34` for XGBoost. This keeps the real class prevalence in the evaluation
set, so reported metrics describe what would happen on a real screening population.

---

## 4. Feature selection

Two independent rankings, then a decision to **keep all 15 columns** (6 numeric + 3 gender +
6 smoking) for every model, so the comparison stays fair.

- **Correlation analysis** (figure in §2): `blood_glucose_level` (r ≈ 0.42) and `HbA1c_level`
  (r ≈ 0.40) dominate and are only weakly correlated with each other, so neither is redundant.
- **Statistical ranking:** an ANOVA F-test per numeric feature. This is model-free, which is
  what feature selection should be based on. Model-based importance is shown in §7, once the
  models exist.
- **Decision:** with only 8 original features there is no dimensional-reduction problem to
  solve; dropping weaker columns would save negligible compute while making the four models
  harder to compare.

---

## 5. Model deployment

**Over-fitting prevention** — two deliberately unrestricted models make over-fitting
*measurable* rather than asserted:

| Model | Train AUC | Validation AUC | Gap |
|---|---|---|---|
| Decision Tree (unrestricted) | 1.0000 | 0.8512 | **0.1488** |
| XGBoost (deep, 2000 trees) | 1.0000 | 0.9659 | 0.0341 |

The unrestricted tree memorises the training set and generalises at only 0.8512. The tuned tree
reaches 0.9748 on test, so regularisation is worth roughly **0.12 ROC-AUC** here — which is why
`max_depth` and `min_samples_leaf` are tuned rather than left at defaults.

**Tuning:** 5-fold stratified CV scored on ROC-AUC (threshold-independent), with small
hand-picked grids. Best configurations:

| Model | Best CV ROC-AUC | Selected parameters |
|---|---|---|
| Logistic Regression | 0.9626 | `C=0.01` |
| Decision Tree | 0.9740 | `max_depth=8`, `min_samples_leaf=50` |
| SVM | 0.9609 | `C=1.0` |
| XGBoost | 0.9796 | `n_estimators=300`, `max_depth=3`, `learning_rate=0.05` |

SVM is the exception: it is tuned **and** refit on a stratified 20,000-row subsample because an
RBF fit is roughly quadratic in sample count. Recorded as a limitation in §8.2.

`SVC(probability=True)` is deliberately **not** used — deprecated in scikit-learn 1.9 and it
adds internal Platt scaling. ROC/PR metrics use `decision_function` instead.

---

## 6. Performance metrics / confusion matrix

All four models evaluated on the held-out test set (14,422 patients, 1,272 positive), never used
for fitting or tuning. Full table with TP/FP/TN/FN in `results/model_comparison.csv`.

**Every model's accuracy is below the 0.9118 baseline, and that is the correct result, not a
failure.** Class weighting buys recall at the cost of accuracy: XGBoost accepts 1,301 false
positives to catch 1,180 of the 1,272 diabetic patients. The baseline takes the opposite trade
and catches none. For screening the asymmetry is the point — a false positive costs a blood
test; a false negative lets a treatable disease go uncaught.

Figures: `06_roc_curves`, `07_precision_recall_curves`, `08_confusion_matrices`,
`09_metric_comparison`, `cv_logistic`, `cv_decision`, `cv_svm`, `cv_xgboost`.

---

## 7. Result

The table in the format required by the brief: columns **1. Model list, 2. AUC, 3. Accuracy,
4. Precision, 5. Recall, 6. F1-Score**, rows **`Log St`, `DT`, `SVM`, `XGBoost`**.
Also saved as `results/final_result_table.csv`.

| 1. Model list | 2. AUC | 3. Accuracy | 4. Precision | 5. Recall | 6. F1-Score |
|---|---|---|---|---|---|
| Log St | 0.9639 | 0.8878 | 0.4336 | 0.8876 | 0.5826 |
| DT | 0.9748 | 0.8807 | 0.4204 | 0.9316 | 0.5793 |
| SVM | 0.9619 | 0.8960 | 0.4548 | 0.9009 | 0.6044 |
| XGBoost | **0.9799** | 0.9034 | **0.4756** | 0.9277 | **0.6288** |

> **Reference line (not a fifth row, since the brief specifies four models):** majority-class
> baseline — Accuracy 0.9118, Recall 0.0000, PR-AUC 0.0882.

### 7.1 Which model performed best?

**XGBoost** is best on AUC (0.9799), F1 (0.6288) and precision (0.4756), and tied on recall.
The Decision Tree has the highest raw recall (0.9316) but the lowest precision (0.4204) — it
buys extra true positives with more false alarms.

### 7.2 Which model was most interpretable, and was XGBoost worth it?

| Model | AUC | What a clinician gets |
|---|---|---|
| Log St | 0.9639 | One coefficient per feature, directly readable as risk factors |
| DT | 0.9748 | A small set of if/then rules that mirror clinical reasoning |
| XGBoost | **0.9799** | 300 boosted trees; importance scores, but no single readable rule |

**XGBoost's AUC advantage over the Decision Tree is 0.0051.** For clinical decision support that
is a poor trade: you would surrender a rule set a doctor can audit line by line in exchange for
half a point of AUC, on a synthetic dataset whose real-world performance is unknown anyway.

**Recommendation: deploy the Decision Tree (or Logistic Regression), and report XGBoost only as
the performance ceiling.** The Decision Tree is the stronger interpretable option — it beats
logistic regression by 0.0109 AUC and its rules are easier to walk a clinician through. Model
complexity is not rewarded in this problem, because the signal is concentrated in two lab
measurements.

### 7.3 Feature importance

One-hot columns are summed back into their parent feature so both models read on the same eight
features. Both agree on the top three — `HbA1c_level`, `blood_glucose_level`, `age` — which is
reassuring, since they measure the condition by completely different means. `age` deserves a
caveat: all 17,219 patients under 18 contribute only 82 positives, so in this dataset being a
child is nearly equivalent to being negative.

They disagree at the bottom. The tree gives near-zero importance to `gender`, `heart_disease` and
`smoking_history` because greedy splits never need them once `HbA1c` and `glucose` are
available, whereas Logistic Regression retains `smoking_history` at 0.127 as additive evidence.
A tree's zero means "not used for splitting", not "no association".

---

## 8. Conclusions and limitations

### 8.1 What the exercise showed

- **Accuracy alone is actively misleading under imbalance.** The majority-class baseline scores
  0.9118 and catches nothing; the best model scores 0.9034 and catches 92.8% of cases.
- **Precision-Recall curves beat ROC curves** for a rare positive class — SVM looks acceptable on
  ROC-AUC (0.9619) and weak on PR-AUC (0.7822).
- **Regularisation is measurable.** An unrestricted tree loses 0.1488 ROC-AUC to over-fitting;
  tuning recovers it.
- **`"No Info"` is a category, not a blank.** Its 4.06% diabetes rate against 9.53–17.00% for
  real smoking categories makes imputing it actively misleading.
- **Model complexity was not rewarded.** The RBF kernel performed *worse* than logistic
  regression, and XGBoost's AUC gain over a single decision tree was only 0.0051 — which is why
  the interpretable model is the recommended one.

### 8.2 Limitations

- **The data is synthetic.** No institutional citation, and the generator produced 17,219
  patients under 18 with a minimum age of 0.08 years, including 911 infants under 1 year old.
  **No clinical claim can be supported by this dataset.**
- **No external validation.** A single 70/15/15 split on one dataset, with no cross-dataset or
  temporal check.
- **The operating threshold was never chosen.** `predict` uses the default 0.5. A real screening
  tool would tune the threshold on the validation split to hit a target recall; that step is not
  implemented.
- **Cost asymmetry is asserted, not modelled.** That a missed diagnosis is worse than a false
  alarm is a clinical assumption; no cost matrix is applied.
- **SVM fitted on a subsample.** Tuned and refit on a stratified 20,000-row subsample rather than
  all 67,302 training rows, so its hyperparameters and 0.9619 AUC are a weaker estimate than the
  other three.

### 8.3 Deviations from the approved project plan

1. **Implausible values were documented, not zeroed to missing.** The plan proposed treating
   biologically impossible values as missing; they were reported and left in place. A minimum age
   of 0.08 years and 911 infants indicate a synthetic generator rather than corrupted records, so
   blanking them would have discarded 17% of rows without fixing anything real.
2. **No second-dataset validation.** Offered as optional in the plan; the candidate dataset shares
   only three usable features, so an external test would have been weak evidence.
3. **Dataset changed from Pima to Kaggle.** Directed after the plan was written. The plan assumed
   roughly 65/35 class balance; the actual dataset is 91.5/8.5 — considerably worse, which is why
   a majority-class baseline and PR curves were added beyond the plan's scope.
4. **Added beyond the plan:** majority-class baseline, PR-AUC and PR curves, boxplots, and explicit
   documentation of the `"No Info"` sentinel.

---

## 9. Reusing the trained model on a new dataset

`run_pipeline.py` persists every tuned pipeline (preprocessing + classifier) to `models/`. Given a
new CSV that has the same feature columns, score it without retraining:

```bash
python predict.py newdata.csv                     # XGBoost, the best-AUC model
python predict.py newdata.csv --model decision_tree
python predict.py newdata.csv --out scored.csv
```

The output copies the input rows and adds `prediction` (0/1) and `probability` (or
`decision_score` for SVM). If the CSV still carries the `diabetes` column, the metrics are printed
too. Unknown categories in `gender` / `smoking_history` are ignored by the encoder rather than
crashing, and a missing feature column fails with a clear message. Run `python run_pipeline.py`
once first to create `models/`.

---

## 10. Running the workflow on a different dataset (exam)

When the exam hands you a **different dataset** (different feature columns, still binary
classification), the saved diabetes models do not apply — retrain the same workflow with
`run_generic_pipeline.py`. It infers the schema instead of hardcoding it:

```bash
python run_generic_pipeline.py data.csv
python run_generic_pipeline.py data.csv --target outcome --positive-label 1
python run_generic_pipeline.py data.csv --outdir results_exam
```

What it does, matching the methodology in this report:

- profiles the data (balance, missingness, duplicates, summary statistics),
- infers numeric vs. categorical features from the dtypes,
- treats the **minority class as positive** unless `--positive-label` says otherwise,
- stratified 70 / 15 / 15 split, `class_weight='balanced'` / `scale_pos_weight` imbalance
  handling,
- tunes the same four models with the same 5-fold ROC-AUC grids,
- evaluates on the held-out test set and writes the brief's required result table,
- draws ROC / PR / confusion / metric / feature-importance figures,
- saves the tuned pipelines to `<outdir>/models/`.

All output goes to `--outdir` (default `results_<csv-stem>`), so the diabetes results in
`results/` and `figures/` are never touched. Score more rows afterwards with:

```bash
python predict.py more.csv --model svm --models-dir results_exam/models
```

If the target column is not the last column, pass `--target NAME`.

---

## Repository layout

```
run_pipeline.py                     # end-to-end reproduction of every number and figure
run_generic_pipeline.py             # same 4-model workflow on ANY binary-classification CSV
predict.py                          # score a new CSV with a saved model
src/
  fetch_data.py                     # official Kaggle API download + strict validation
  data.py                           # cleaning, reports, split, ColumnTransformer
  models.py                         # 4 models, CV tuning, overfitting checks, importance
  evaluation.py                     # baseline, metrics, confusion matrices, result table
  plots.py                          # all 16 figures
notebooks/
  Diabetes Risk Prediction.ipynb    # the report (executed, outputs embedded)
data/                               # diabetes_prediction_dataset.csv
figures/                            # 16 PNG figures
results/                            # 12 result files, incl. final_result_table.csv
models/                             # tuned pipelines (*.joblib) saved by run_pipeline.py
Plan/                               # the approved project plan
```

## Reproducing

```bash
pip install -r requirements.txt
python run_pipeline.py              # ~2-3 minutes
```

Runtime ~132 s. `random_state = 42` for splitting, cross-validation and every model. Executing
the notebook reproduces the same values and re-renders the figures:

```bash
jupyter nbconvert --to notebook --execute --inplace "notebooks/Diabetes Risk Prediction.ipynb"
```

Figures are written to `figures/`, tables to `results/`. `results/final_result_table.csv` holds
the §7 table in the brief's required format.

### Google Colab

Upload `ML_lab_colab.zip` (generated in the project root), unzip it, and open the notebook from
the Colab file panel. Full steps in [`COLAB.md`](COLAB.md).

```python
!unzip -q ML_lab_colab.zip -d /content
```

No Kaggle credentials are required, because the archive includes
`data/diabetes_prediction_dataset.csv` and `fetch()` only contacts the Kaggle API when that file
is absent.
