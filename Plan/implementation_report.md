# Implementation Report — How We Built the Plan

**Project:** Early Medical Diagnosis System (Diabetes Risk Prediction)
**Plan of record:** [`project_plan.md`](project_plan.md)
**Course:** Introduction to Machine Learning Lab — Instructor: Bashar Sir

This document records, step by step, what was actually built to satisfy
[`project_plan.md`](project_plan.md): which plan step each piece of work satisfies, what we did,
how we did it, and what came out. It is written as an honest build log, including the decisions
that changed and the bugs we hit, because those are part of the engineering work.

---

## What was built

| Artefact | Path |
|---|---|
| End-to-end reproducible pipeline | `run_pipeline.py` |
| Data acquisition (official Kaggle API) | `src/fetch_data.py` |
| Cleaning, reports, split, preprocessing | `src/data.py` |
| Model definitions, CV tuning, over-fitting checks, importance | `src/models.py` |
| Metrics, confusion matrices, required result table | `src/evaluation.py` |
| All 16 figures | `src/plots.py` |
| Executed report (the submission notebook) | `notebooks/Diabetes Risk Prediction.ipynb` |
| Written report | `README.md` |
| 16 figures / 12 result files | `figures/`, `results/` |

Runtime: **~132 s** end to end. `random_state = 42` throughout (splitting, cross-validation, and
every model), so every number is reproducible.

---

## Step 0 — Model selection (satisfies plan §0)

**What we did.** Assessed all six candidates against the problem — binary clinical
classification on structured tabular data — *before* writing any pipeline, and recorded the
verdict in the notebook and README as a table with a written reason per candidate.

**How.** Excluded **Linear Regression** (continuous-target model; violates its own assumptions on
a 0/1 outcome, no class interpretation) and **KNN** (curse of dimensionality, scaling-sensitive,
slow at inference, and no model artifact to interpret). Carried forward **Logistic Regression,
Decision Tree, SVM and XGBoost**, deliberately spanning simple/interpretable → interpretable
non-linear → kernel-based → high-performance ensemble.

**Why this mattered in practice.** Section 0 states the expected outcome before the models run,
and section 5 later confirmed one part of it empirically: the RBF kernel performed *worse* than
plain logistic regression (−0.0020 AUC), which is not a result anyone would have predicted from
the plan alone.

---

## Step 1 — Data set (satisfies plan §1)

**What we did.** Sourced the Kaggle *Diabetes prediction dataset*
(`iammustafatz/diabetes-prediction-dataset`) — 100,000 rows × 9 columns, 6 numeric + 2
categorical features, binary `diabetes` target, 8.5% positive.

**How.** `src/fetch_data.py` calls the official Kaggle API
(`from kaggle.api.kaggle_api_extended import KaggleApi`), downloading through the authenticated
API rather than a hand-placed file, and then validates the result on **every** run — expected
column set, expected row count (100,000), expected categorical levels, and a numeric target.
Validation is strict on purpose: a truncated download or a changed schema must fail loudly
instead of silently producing wrong numbers. Subsequent runs reuse the local copy and re-validate
it. Credentials are read from the standard Kaggle credentials file and are never printed.

**Also decided here:** the plan's optional second-dataset validation was **declined**. The
available candidate datasets share only three meaningfully overlapping features, so a
cross-dataset test would have been weak evidence. This is recorded as a limitation (§8.2 of the
notebook) rather than silently dropped.

---

## Step 2 — Data exploration and cleaning (satisfies plan §2)

**What we did.** Profiled the data, then made three explicit decisions.

| Finding | Decision |
|---|---|
| 3,854 full-row duplicates (3.85%) | **Dropped.** Only *one* duplicate group disagrees on the label, so removing them discards no information. 100,000 → **96,146** rows. |
| `isnull()` reports **zero** missing values | Treated as **misleading**: 35,816 rows hide missingness inside `smoking_history` as the string `"No Info"`. |
| `"No Info"` diabetes rate **4.06%** vs **9.53–17.00%** for every genuine smoking category | **Kept as its own category.** Missingness this strongly correlated with the target is *systematic*, not random. Imputing it would discard real signal and would tell a doctor that 35,816 patients never smoked. |
| Physiologically impossible values: min age **0.08 years**, **17,219** patients under 18, **911** infants under 1 year | **Reported, not corrected** (deliberate deviation — see *Deviations* below). |
| `gender == "Other"` (18 rows, 0% positive) | **Kept**, one-hot encoded. |
| `hypertension` / `heart_disease` present | Noted: diabetes rate more than quadruples. |

**How.** Each of these is computed by a dedicated function in `src/data.py`
(`duplicate_report`, `missing_value_report`, `category_breakdown`, `data_quality_report`) that
returns a DataFrame — so every cleaning claim in the report is a printed table, not an assertion.

**Figures produced (plan §2 asked for histograms, boxplots, heatmap):**
`01_numeric_distributions`, **`01b_numeric_boxplots`**, `02_class_imbalance`,
`03_correlation_heatmap`, `04_category_diabetes_rates`, `05_missingness_sentinel`.

The **boxplot** was added specifically to satisfy the plan's "boxplots to spot outliers"
requirement, and it earned its place: it makes the right-skew of `bmi` and
`blood_glucose_level` and the long low-age tail visible in a way the histograms do not.

**The age artifact, stated plainly.** The dataset contains 911 infants under one year old and
17,219 patients under 18, and those children contribute only **82 positives — 0.48%**, against
8.82% overall. In this dataset, being a child is nearly equivalent to being negative. That is
recorded as a synthetic-data artifact and carried as a caveat wherever `age` is discussed as an
important feature.

---

## Step 3 — Data wrangling (satisfies plan §3)

**What we did.** Built a two-path `ColumnTransformer`, then a stratified 70/15/15 split.

| Path | Steps | Why |
|---|---|---|
| Numeric | median imputation → `StandardScaler` | `bmi` and `blood_glucose_level` are right-skewed, so the median resists outliers better than the mean. Scaling is required by Logistic Regression and SVM. |
| Categorical | most-frequent imputation → one-hot | Keeps categories discrete; one-hot columns are deliberately **not** scaled, since a 0/1 indicator belongs on its natural scale. |

**Split result:** 67,302 train / 14,422 validation / 14,422 test — 5,937 / 1,273 / 1,272
positives.

**Imbalance handling — the most consequential decision in the project.** The plan left this open
("class weighting, SMOTE, or stratified sampling"). We chose **class weighting, and explicitly
not resampling**:

* `class_weight='balanced'` for the scikit-learn models; `scale_pos_weight=10.34` for XGBoost
  (the computed negative-to-positive ratio in train).
* **No SMOTE.** The evaluation set keeps the real 8.82% prevalence, so the reported metrics
  describe what would actually happen on a real screening population. Resampling the test set
  would have inflated every headline number and made them meaningless.

**Majority-class baseline established as the number to beat: 0.9118 accuracy, recall 0.**

---

## Step 4 — Feature selection (satisfies plan §4)

**What we did.** Two independent rankings, then a decision to **keep all 15 columns**
(6 numeric + 3 gender one-hot + 6 smoking one-hot) for every model.

**How.**
* **Correlation analysis** (figure in §2): `blood_glucose_level` (r ≈ 0.42) and `HbA1c_level`
  (r ≈ 0.40) dominate the target correlation and are only weakly correlated with *each other*, so
  neither is redundant. No pair warranted dropping one.
* **ANOVA F-test** per numeric feature gave a model-free ranking:
  `blood_glucose_level` (F ≈ 21,113) > `HbA1c_level` (≈ 19,022) > `age` (≈ 7,257) >
  `bmi` (≈ 4,657) > `hypertension` (≈ 3,829) > `heart_disease` (≈ 2,886).
* **Categorical levels** ranked by diabetes rate.
* **PCA was skipped.** The plan called it optional and exploratory; with 8 original features there
  is no dimensional-reduction problem to solve.

**Why keep everything.** The plan requires one feature subset used consistently across all four
models. With so few features, dropping the weaker columns would have saved negligible compute
while making the four models harder to compare. Model-based importance is reported in §7 instead,
*after* the models are fitted — using it to select features beforehand would have been circular.

---

## Step 5 — Model deployment (satisfies plan §5)

**What we did.** Trained all four models on the identical processed training set, and tuned each
by 5-fold stratified cross-validation scored on ROC-AUC (threshold-independent).

**Over-fitting prevention, made measurable.** Rather than asserting that regularisation matters,
we fitted two deliberately unrestricted models and measured the gap:

| Model | Train AUC | Validation AUC | Gap (over-fit) |
|---|---|---|---|
| Decision Tree (unrestricted, `max_depth=None`, `min_samples_leaf=1`) | 1.0000 | 0.8512 | **0.1488** |
| XGBoost (deep, 2000 trees, depth 12) | 1.0000 | 0.9659 | 0.0341 |

The unrestricted tree memorises the training set and generalises at only 0.8512. The tuned tree
reaches 0.9748 on test — so regularisation is worth roughly **0.12 ROC-AUC** here, which is why
`max_depth` and `min_samples_leaf` are tuned rather than left at defaults.

**Selected hyperparameters and CV scores:**

| Model | Best CV ROC-AUC | Selected parameters |
|---|---|---|
| Logistic Regression | 0.9626 | `C=0.01` |
| Decision Tree | 0.9740 | `max_depth=8`, `min_samples_leaf=50` |
| SVM | 0.9609 | `C=1.0` |
| XGBoost | 0.9796 | `n_estimators=300`, `max_depth=3`, `learning_rate=0.05` |

Grids were kept small and hand-picked on purpose: with ~67,000 training rows the aim is to
demonstrate regularisation and model selection, not to squeeze out the last fraction of a percent.
Each model's full train-vs-validation grid is plotted (`cv_logistic`, `cv_decision`, `cv_svm`,
`cv_xgboost`).

**SVM is fitted on a subsample.** An RBF fit is roughly quadratic in sample count, so SVM is
tuned **and** refit on a stratified 20,000-row subsample rather than all 67,302 rows. This is a
real limitation on the fairness of the comparison and is recorded as such in the report rather
than quietly omitted.

---

## Step 6 — Performance metrics / confusion matrix (satisfies plan §6)

**What we did.** Evaluated all four models on the held-out test set (14,422 patients, 1,272
positives), which was never used for fitting or tuning, and reported every metric the plan
requires — including the TP/FP/TN/FN counts alongside accuracy, precision, recall, F1 and AUC.

**The central finding of the whole project.** *Every model's accuracy is below the 0.9118
majority baseline, and that is the correct result, not a failure.* Class weighting buys recall at
the cost of accuracy: XGBoost accepts 1,301 false positives to catch 1,180 of the 1,272 diabetic
patients. The baseline takes the opposite trade and catches none. For screening, a false positive
costs a blood test; a false negative lets a treatable disease go uncaught.

**Figures:** `06_roc_curves` (all four overlaid, as the plan requires), `07_precision_recall_curves`,
`08_confusion_matrices`, `09_metric_comparison`.

**Why PR curves were added beyond the plan.** At an 8.8% positive rate the ROC curve flatters
every model. SVM is the clearest proof: a competitive ROC-AUC of 0.9619 but the *weakest* PR-AUC
of 0.7822 — the classic symptom of scores that are poorly ordered in the region that actually
matters. Reporting ROC alone would have hidden that.

---

## Step 7 — Result (satisfies plan §7)

Final table in the exact format the brief requires — columns **1. Model list, 2. AUC,
3. Accuracy, 4. Precision, 5. Recall, 6. F1-Score**, rows **Log St, DT, SVM, XGBoost** — also
saved to `results/final_result_table.csv`:

| 1. Model list | 2. AUC | 3. Accuracy | 4. Precision | 5. Recall | 6. F1-Score |
|---|---|---|---|---|---|
| Log St | 0.9639 | 0.8878 | 0.4336 | 0.8876 | 0.5826 |
| DT | 0.9748 | 0.8807 | 0.4204 | 0.9316 | 0.5793 |
| SVM | 0.9619 | 0.8960 | 0.4548 | 0.9009 | 0.6044 |
| XGBoost | **0.9799** | 0.9034 | **0.4756** | 0.9277 | **0.6288** |

> **Reference line (not a fifth row, since the brief specifies four models):** majority-class
> baseline — Accuracy 0.9118, Recall 0.0000, PR-AUC 0.0882.

Full table including PR-AUC and confusion counts: `results/model_comparison.csv`.

**Best overall:** XGBoost, on AUC (0.9799), F1 (0.6288) and precision (0.4756), tied on recall.
The Decision Tree has the highest raw recall (0.9316) but the lowest precision (0.4204) — it buys
extra true positives with more false alarms.

**Most interpretable, and the performance/interpretability trade-off** — the discussion the plan
asks for:

| Model | AUC | What a clinician gets |
|---|---|---|
| Log St | 0.9639 | One coefficient per feature, directly readable as risk factors |
| DT | 0.9748 | A small set of if/then rules that mirror clinical reasoning |
| XGBoost | **0.9799** | 300 boosted trees; importance scores, but no single readable rule |

**XGBoost's AUC advantage over the Decision Tree is 0.0051.** For clinical decision support that
is a poor trade: you would surrender a rule set a doctor can audit line by line for half a point
of AUC, on a synthetic dataset whose real-world performance is unknown anyway. **Recommendation:
deploy the Decision Tree (or Logistic Regression) and report XGBoost only as the performance
ceiling.** Model complexity is not rewarded in this problem, because the signal is concentrated in
two lab measurements.

**Feature importance** (`10_…logistic_regression`, `11_…decision_tree`): one-hot columns are summed
back into their parent feature so both models read on the same eight features. Both agree on the
top three — `HbA1c_level`, `blood_glucose_level`, `age`. They disagree at the bottom: the tree
gives near-zero importance to `gender`, `heart_disease` and `smoking_history` because greedy
splits never need them once `HbA1c` and `glucose` are available, whereas Logistic Regression keeps
`smoking_history` at 0.127 as additive evidence. **A tree's zero means "not used for splitting",
not "no association"** — worth stating explicitly so the chart is not misread.

---

## Step 8 — Conclusions and limitations (report §8)

Five findings are drawn out: accuracy is actively misleading under imbalance; PR curves beat ROC
curves for a rare positive class; regularisation is measurably worth ~0.12 AUC; `"No Info"` is a
category and not a blank; and model complexity was not rewarded.

Five limitations are stated rather than buried — most importantly that **the data is synthetic**
(no institutional citation, 911 infants under one year old), so **no clinical claim can be
supported by it**; plus no external validation, no tuned decision threshold (`predict` uses the
default 0.5), an asserted-but-not-modelled cost asymmetry, and the SVM subsample.

---

## Deviations from the plan

1. **Implausible values were documented, not zeroed to missing.** The plan proposed treating
   biologically impossible values as missing data. We reported and kept them. A minimum age of
   0.08 years and 911 infants under one year old indicate a *synthetic generator* rather than
   corrupted records, so blanking them would have discarded 17% of the rows without fixing
   anything real. The consequence is stated in the limitations.
2. **No second-dataset validation.** The plan offered it as optional; declined for the reason
   given in Step 1, and recorded as a limitation instead.
3. **Dataset changed from Pima to Kaggle**, by direction after the plan was first drafted. Note the
   original plan assumed roughly 65/35 class balance; the actual dataset is 91.5/8.5 —
   considerably worse, which is why a majority-class baseline, PR-AUC and PR curves were added
   beyond the plan's scope.
4. **Added beyond the plan:** majority-class baseline, PR-AUC and PR curves, boxplots, and
   explicit documentation of the `"No Info"` sentinel — all driven by the severity of the
   imbalance.
5. **SVM fitted on a 20,000-row subsample** rather than the full training split, for runtime
   reasons; recorded as a limitation on the fairness of the comparison.

---

## Bugs found and fixed during the build

Recorded because they are part of the work, and because each would have produced wrong numbers
rather than an error:

| Bug | Why it mattered |
|---|---|
| **Inverted majority baseline** — `y_true.max()` was used to pick the majority class | With an 8.8% positive rate the majority class is 0, so `max()` returned 1 and predicted *every* patient as diabetic, inverting the entire baseline. |
| **Over-fitting check evaluated XGBoost twice** instead of the two intended models | The reported over-fit gap was for the wrong model. |
| **`stratified_subsample` returned its arguments in the wrong order** | The SVM search would have received features as labels. |
| **pandas 3.0 dtype change broke missingness detection** | `"No Info"` sentinel reporting silently broke. |
| **Correlation heatmap excluded the target column** | The most important correlation in the dataset was missing from the figure. |
| **`SVC(probability=True)` deprecated in scikit-learn 1.9** | Removed; ROC/PR metrics now use `decision_function`, which is sufficient and avoids internal Platt scaling. |
| **A notebook cell was overwritten by an earlier patch**, destroying the Conclusions and Limitations section | The notebook executed cleanly with zero errors while the report was silently incomplete — caught by a structural audit, not by execution. The section was fully restored. |
| **`tune_model` docstring overstated the SVM refit** — it claimed a full-data refit, but the caller passes the subsample as `X_train`, so the refit happens there | The report claimed something the code did not do. Since changing it would have changed the verified metrics, we corrected the docstring and the report text instead. |

---

## Verification

A static audit script re-derives every number claimed in the report from the raw data and checks
it against the written claims. It covers data facts, split sizes, cleaning decisions, every
metric in the required result table, the over-fitting gaps, model-selection coverage, notebook
structure and section order, README table headers, and artefact counts.

**Result: 169 / 169 checks pass, 0 failures.**

The audit caught three genuine problems that execution alone did not: the destroyed conclusions
section, a `## 8.` heading that appeared without its body, and a source-file line ending that had
dropped a blank line out of a printed table.

---

## Reproducing

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Re-execute the report notebook to regenerate every figure and output:

```bash
jupyter nbconvert --to notebook --execute --inplace "notebooks/Diabetes Risk Prediction.ipynb"
```

`random_state = 42` for splitting, cross-validation and every model, so the run is deterministic.
Figures go to `figures/`, tables to `results/`.
