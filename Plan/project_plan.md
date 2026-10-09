# Introduction to Machine Learning Lab — Project Plan

**Course:** Introduction to Machine Learning Lab
**Instructor:** Bashar Sir
**Project Topic:** Early Medical Diagnosis System (Diabetes Risk Prediction)

> **Status of this document.** This is the plan of record. It supersedes the two earlier
> drafts (`early_medical_diagnosis_project.md` and `early_medical_diagnosis_project_by_claude2.md`),
> which were written against the *Pima Indians Diabetes Dataset*. The dataset was changed to the
> Kaggle *Diabetes prediction dataset* by direction after those drafts were written, so the plan
> below is restated against the dataset actually used. The full record of what was built, and of
> every deviation from the earlier drafts, is in `implementation_report.md`.

---

## Objective

Build a binary classification system that analyses clinical attributes (age, BMI, glucose,
HbA1c, hypertension, heart disease, gender, smoking history) to predict whether a patient is at
high risk of diabetes, and to report honestly how well four different model families can do
that job.

The project follows a complete applied machine-learning workflow: dataset selection, exploration
and cleaning, data wrangling, feature selection, model selection and justification, model
deployment, performance evaluation, and result reporting.

---

## 0. Model Selection — Why These Models?

Before building the pipeline, the candidate pool — **Linear Regression, Decision Tree, SVM,
Logistic Regression, KNN and XGBoost** — is evaluated against the nature of the problem (binary
clinical classification on structured/tabular data) to decide what to carry forward.

| Candidate | Include? | Reasoning |
|---|---|---|
| **Linear Regression** | Excluded | Designed for continuous-valued targets, not class labels. Applying it to a 0/1 diagnosis outcome violates its own assumptions (unbounded output, Gaussian error) and gives no probabilistic class interpretation. It is not appropriate for this classification task and is dropped from the final comparison. |
| **Logistic Regression (Log St)** | **Included** | The natural discriminative baseline for binary classification. Outputs calibrated probabilities, is highly interpretable (coefficients relate directly to clinical risk factors), computationally cheap, and works well as long as the classes are near-linearly separable in feature space — a reasonable starting assumption for clinical data. |
| **Decision Tree (DT)** | **Included** | Captures non-linear relationships and feature interactions (e.g. "high glucose AND high BMI" risk combinations) without requiring feature scaling. Highly interpretable via its rule structure, which matters in a medical context where clinicians want to understand *why* a prediction was made. Also serves as a natural stepping stone toward the ensemble method (XGBoost) below. |
| **SVM** | **Included** | Clinical data is rarely linearly separable in raw input space. SVM with an RBF kernel maps features into a higher-dimensional space to find a clearer decision boundary, making it a strong non-linear classifier that often performs well on small-to-medium tabular medical datasets. |
| **KNN** | Excluded | Sensitive to the curse of dimensionality and to feature scaling, computationally expensive at inference time on larger datasets, and — critically — offers no interpretability or model artifact (it simply memorises the training data), which limits its usefulness for a clinical decision-support narrative. It is used only as an optional exploratory baseline, not in the final comparison table. |
| **XGBoost** | **Included** | A gradient-boosted ensemble that typically achieves the strongest raw predictive performance on structured/tabular data of this kind. Handles non-linearities, feature interactions and missing values well, and provides feature importance scores that add interpretability despite being an ensemble method. Included as the "strongest performer" benchmark against the simpler, more interpretable models above. |

**Final model shortlist carried into the pipeline: Logistic Regression, Decision Tree, SVM and
XGBoost.**

This selection deliberately spans the spectrum from simple and interpretable (Logistic
Regression) → interpretable and non-linear (Decision Tree) → non-linear and kernel-based (SVM)
→ high-performance ensemble (XGBoost), giving a meaningful basis for comparison rather than four
arbitrary models.

---

## 1. Data set

* **Primary dataset:** Kaggle *Diabetes prediction dataset* (`iammustafatz/diabetes-prediction-dataset`).
* **Size and shape:** 100,000 rows × 9 columns.
* **Features:** 6 numeric (`age`, `hypertension`, `heart_disease`, `bmi`, `HbA1c_level`,
  `blood_glucose_level`) + 2 categorical (`gender`, `smoking_history`).
* **Target:** `diabetes` (1 = positive, 0 = negative).
* **Class balance:** 91,500 negative / 8,500 positive — 8.5% positive, i.e. severely imbalanced.
* **Acquisition:** downloaded programmatically through the official Kaggle API, with strict
  schema and row-count validation on every run, so a truncated download or a changed schema
  fails loudly rather than silently producing wrong numbers.
* **External validation:** a second dataset was considered and **declined** — the candidates
  available share only three meaningfully overlapping features, so a cross-dataset test would
  have been weak evidence. This is recorded as a limitation instead.

---

## 2. Data exploration and cleaning

* Summary statistics (mean, median, std, min/max) for each feature.
* Class balance check on the target label.
* Visualisations: histograms/distribution plots per feature, **boxplots to spot outliers**,
  correlation heatmap, categorical rate chart, and a missingness-sentinel chart.
* Identify biologically implausible values (e.g. minimum age of 0.08 years, 911 infants under
  one year old) — these are **flagged and reported**, and the decision taken about them is
  recorded explicitly.
* Duplicate-record analysis, including whether any duplicate group disagrees on the label.

---

## 3. Data wrangling

* Impute missing/invalid values — median for numeric, most-frequent for categorical.
* Handle the class imbalance, decided from the exploration step.
* Feature scaling/standardisation (required for Logistic Regression and SVM; harmless for the
  tree models, but applied consistently so the comparison stays fair).
* Train/validation/test split (70/15/15) with stratification on the target label.

---

## 4. Feature selection

* Correlation analysis to check for redundant or highly correlated features.
* Statistical tests (ANOVA F-test, plus a chi-square style comparison across categorical levels)
  to rank feature relevance.
* Optionally apply dimensionality reduction (e.g. PCA) as a comparison point, though the feature
  count here is small enough that this is mainly exploratory.
* Finalise the feature subset used **consistently across all four models** for a fair comparison.

---

## 5. Model deployment

* Train each of the four shortlisted models (Logistic Regression, Decision Tree, SVM, XGBoost) on
  the same processed training set.
* Tune hyperparameters for each via cross-validation — regularisation strength for Logistic
  Regression; `max_depth` / `min_samples_leaf` for the Decision Tree; `C` for the RBF SVM;
  learning rate, `n_estimators` and `max_depth` for XGBoost. This directly demonstrates the
  syllabus topics of **over-fitting prevention and model selection** by comparing regularised
  against unregularised/default variants.
* Evaluate all tuned models on the held-out test set, which is never used for fitting or tuning.

---

## 6. Performance metrics / confusion matrix

For each of the four models, report:

* **Confusion matrix** (True Positives, False Positives, True Negatives, False Negatives)
* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **AUC (Area Under the Curve)**
* **ROC (Receiver Operating Characteristic) curve** — plotted for all four models on a single
  overlaid chart for direct visual comparison.
* A **Precision-Recall curve** is reported alongside the ROC curve, because under severe
  imbalance it is the more informative view.
* A **majority-class baseline** is reported as a reference line, so the accuracy column is never
  read in isolation.

---

## 7. Result

Final comparison table, in the exact format required by the brief — columns **1. Model list,
2. AUC, 3. Accuracy, 4. Precision, 5. Recall, 6. F1-Score**, rows **Log St, DT, SVM, XGBoost**:

| 1. Model list | 2. AUC | 3. Accuracy | 4. Precision | 5. Recall | 6. F1-Score |
|---|---|---|---|---|---|
| Log St | | | | | |
| DT | | | | | |
| SVM | | | | | |
| XGBoost | | | | | |

*(Values filled in after the experiments are run — see `results/final_result_table.csv`.)*

Alongside the table, present:

* The overlaid **ROC curve plot** for all four models.
* A short discussion of which model performed best overall, which was most interpretable, and the
  trade-off between performance and interpretability in a clinical decision-support context —
  i.e. whether the marginal AUC gain from XGBoost (if any) justifies its reduced interpretability
  compared with Logistic Regression or the Decision Tree in a medical setting.
* An honest statement of the project's limitations and of any deviation from this plan.

---

## Why This Project Works

* **Justified model choice.** Rather than arbitrarily picking models, each candidate — including
  the excluded Linear Regression and KNN — is evaluated against the problem characteristics
  first, giving the project a defensible methodology.
* **Complete pipeline.** Covers the full applied ML workflow — dataset, exploration/cleaning,
  wrangling, feature selection, deployment, evaluation and result reporting — rather than
  jumping straight to modelling.
* **Fair, interpretable comparison.** All four models are trained and evaluated on the identical
  processed dataset and feature set, with a consistent metrics table (AUC, Accuracy, Precision,
  Recall, F1-Score) and ROC curves, making the final comparison meaningful.
* **Real-world impact.** Demonstrates practical utility of machine learning in healthcare and
  early screening, with a results-driven discussion of the performance–interpretability trade-off
  that matters in real clinical deployment decisions.
