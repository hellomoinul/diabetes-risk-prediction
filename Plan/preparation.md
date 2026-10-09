# Preparation — Model Choices & the Accuracy Question

**Project:** Early Medical Diagnosis System (Diabetes Risk Prediction)
**Purpose:** Answers to the questions a teacher is most likely to ask about *which models we used, why, why not the others,* and *why our accuracy differs from published results*. Written for viva/defence preparation.

**All numbers below are taken from the executed project** (`results/final_result_table.csv`, `results/model_comparison.csv`, `results/pipeline_summary.json`) and independently re-verified by a 169-check audit.

---

## Quick revision card

| Item | Answer |
|---|---|
| Models used | **Logistic Regression, Decision Tree, SVM, XGBoost** |
| Models rejected | **Linear Regression, KNN** |
| Best on metrics | **XGBoost** — AUC 0.9799, F1 0.6288 |
| Recommended to deploy | **Decision Tree** — AUC 0.9748, and interpretable |
| Why not the best model? | XGBoost beats DT by only **0.0051** AUC; not worth losing auditability |
| Our accuracy | **0.9034** (XGBoost), 0.8807–0.8960 for the others |
| Majority baseline | **0.9118** accuracy, **recall 0** |
| Is below-baseline accuracy a failure? | **No — it is the intended trade.** We bought 92.77% recall at a cost of 0.84% accuracy |
| Kaggle's own accuracy | **The Kaggle page reports no accuracy figure at all** |
| Published claims | 95%–99.7%, mostly with SMOTE/resampling → not comparable |
| Our recall | **0.9277** — the baseline gets **0.0** |

---

## 1. What models did we use, and why?

We used **four** models, chosen to span the interpretability–performance spectrum rather than to be four arbitrary algorithms:

| Model | Test AUC | Test accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Log St** (Logistic Regression) | 0.9639 | 0.8878 | 0.4336 | 0.8876 | 0.5826 |
| **DT** (Decision Tree) | 0.9748 | 0.8807 | 0.4204 | 0.9316 | 0.5793 |
| **SVM** (RBF kernel) | 0.9619 | 0.8960 | 0.4548 | 0.9009 | 0.6044 |
| **XGBoost** | **0.9799** | 0.9034 | **0.4756** | 0.9277 | **0.6288** |

**The logic of the shortlist:**

1. **Logistic Regression** — the correct first assumption for binary classification on roughly linearly separable clinical data. Gives calibrated probabilities and coefficients that map directly onto risk factors. Cheapest model, fully interpretable.
2. **Decision Tree** — captures non-linearity and feature interactions ("high glucose **and** high BMI") with no scaling needed, and produces if/then rules a clinician can read. Natural bridge to the ensemble.
3. **SVM (RBF)** — clinical data is rarely linearly separable; the kernel maps it into a higher dimension to find a cleaner boundary. Represents the "non-linear but not an ensemble" option.
4. **XGBoost** — the strongest performer on structured tabular data; used as the **performance benchmark** to test whether the simpler models leave anything on the table.

**"Which model did you finally choose?"** — This has a two-part answer, and giving only one half is a trap:

- **XGBoost performed best** on AUC (0.9799), F1 (0.6288), precision (0.4756), and tied on recall.
- **But we recommend deploying the Decision Tree.** XGBoost's entire advantage is **0.0051 AUC**. For clinical decision support, trading a rule set a doctor can audit line-by-line for half a point of AUC is a bad deal.

So: *XGBoost is the ceiling, the Decision Tree is the ship.*

---

## 2. Why not the other two?

**Linear Regression — rejected.**
It is built for continuous targets, not class labels. Applying it to a 0/1 diagnosis violates its own assumptions: the output is unbounded (it can predict −0.3 or 1.7, which are meaningless as a diagnosis), and it assumes Gaussian, homoscedastic errors that a binary outcome cannot satisfy. It also gives no probabilistic class interpretation. Wrong tool for classification.

**KNN — rejected.**
Three reasons, in order of importance:
1. **No model artifact.** It simply memorises the training set; there is nothing to inspect, explain, or hand to a clinician. In a medical setting, "this patient is similar to those 5 past patients" is not an explanation anyone can act on.
2. **Curse of dimensionality + scaling sensitivity.** Distance-based, so it degrades as features increase and is completely dependent on our scaling being right.
3. **Slow at inference** and expensive as data grows.

---

## 3. The accuracy question — the one to prepare for properly

> *"Other papers on this dataset report 95%–99.7% accuracy. You got 90.34%. Why is yours lower?*

### 3.1 First, the honest headline

Our accuracy is **0.9034**. Published claims on this dataset are **0.95–0.997**. So yes — **ours is lower, and deliberately so.** The gap has two separate causes, and it is important to keep them apart:

- **Cause A (protocol):** many published numbers are *not comparable* to ours because they resample the data.
- **Cause B (genuine):** part of the gap is real model-quality difference, and we should not hide it.

### 3.2 Cause A — the published numbers are usually measured on different data

**The most important fact to state: on a 91.5% / 8.5% split, accuracy is nearly meaningless.**

A model that answers *"no diabetes"* to every single patient scores **0.9118** — and catches **zero** diabetic patients. So a claim of "95% accuracy" tells you almost nothing unless the author also states the baseline and the recall.

**Why resampling inflates it:** many papers apply SMOTE or class balancing to the evaluation data, which changes the class ratio to roughly 50/50. On a balanced set, accuracy becomes the average of sensitivity and specificity — so the majority class no longer dominates the score. Two models with identical real-world behaviour can report 90% on real data and 97% on resampled data.

**Kaggle itself reports no accuracy figure** for this dataset, so there is no official number to match.

### 3.3 The number that proves our model is not weak

If our test set were class-balanced — the same condition under which those 95%+ papers are reporting — our XGBoost accuracy would be:

```
sensitivity (recall)      = 0.9277
specificity               = 11,849 / 13,150 = 0.9011
balanced accuracy         = (0.9277 + 0.9011) / 2 = 0.9144
```

**So on a balanced evaluation our accuracy is 91.4%, not 90.3%.** The 0.9034 figure is being *penalised* by the real 8.82% prevalence, which is exactly what should happen when you refuse to resample. We deliberately kept the test set at its true prevalence so the numbers describe a real screening population.

### 3.4 Cause B — the honest part: we would still need a better model

We should not claim 95% is impossible. On our test split, reaching 95% accuracy while keeping recall at 0.9277 would require cutting false positives from **1,301 to about 629** — roughly halving them. That is a genuinely better classifier than the one we built, not a measurement artefact.

Why we did not get there:

- **The data is synthetic and noisy.** It contains 911 infants under one year old and a minimum age of 0.08 years. A generator produced that, so the feature→label relationship carries noise that caps achievable performance.
- **We optimise for recall, not accuracy.** Class weighting (`class_weight='balanced'`, `scale_pos_weight=10.34`) deliberately spends accuracy to catch positives. That is a *design choice*, not an error.
- **We never tuned the decision threshold.** `predict` uses the default 0.5. Tuning it on the validation set to hit a recall target would change the accuracy/recall balance — we list this as a limitation.
- **No external validation**, and the split is 70/15/15 rather than 80/20.

### 3.5 How to say it in one sentence

> *"Our accuracy is lower, and that is the intended trade. We report 0.9034 against a 0.9118 majority baseline that catches nobody, and we reach 92.77% recall where the baseline reaches zero. Most published higher figures come from SMOTE-resampled evaluation, which changes the class ratio and makes accuracy uninterpretable — on a balanced version of our own test set we score 91.4%. The remaining gap is real, and it reflects a noisy synthetic dataset."*

---

## 4. Anticipated questions and answers

### A. Models and choices

**Q1. Why did you choose these four models?**
Binary classification on structured tabular clinical data. I picked one model from each family: a linear probabilistic baseline (Logistic Regression), an interpretable non-linear model (Decision Tree), a kernel method (SVM), and a gradient-boosted ensemble (XGBoost). That spans the interpretability–performance range and makes the comparison meaningful.

**Q2. Why is Linear Regression not used for a classification problem?**
It assumes a continuous, unbounded target and Gaussian errors. On a 0/1 label it can output values like 1.7, and it offers no class probability. It is the wrong model family.

**Q3. Why not KNN?**
No model artifact — it memorises training data, so nothing is interpretable. Also sensitive to feature scaling and to high dimensionality, and slow at inference.

**Q4. Which is the best model?**
XGBoost on the headline metrics: AUC 0.9799, F1 0.6288, precision 0.4756. But the *recommended* model is the Decision Tree, because XGBoost's gain is only 0.0051 AUC — not enough to justify an uninterpretable model in medicine.

**Q5. Why is SVM *worse* than plain Logistic Regression?**
Because the decision boundary here is close to linear. The RBF kernel maps the data into a higher dimension, and there was nothing extra to find — so it cost a little without gaining (−0.0020 AUC). This was not expected in advance; it is why we measured rather than assumed.

### B. Metrics and evaluation

**Q6. Which metric is the most important, and why?**
Recall, for a screening task. A false positive costs a blood test; a false negative lets a treatable disease go uncaught. We would rather over-test than miss a diabetic patient. AUC (0.9799) is our headline *ranking* metric because it is threshold-independent.

**Q7. Your accuracy is below the majority baseline. Isn't that a failure?**
No, it is the intended trade, and I can show it with the confusion matrix. The baseline gets 0.9118 by predicting "no" for everyone, with 0 positives caught. XGBoost gives up 0.0084 accuracy to catch 1,180 of 1,272 diabetic patients. Under class weighting, accuracy and recall move in opposite directions by design.

**Q8. Why is your precision only 0.4756?**
Because we deliberately accepted false positives. We cast a wide net: 1,301 false positives against 1,180 true positives. That is the correct behaviour when missing a diagnosis is much worse than a wasted test. Precision would rise only if we narrow the net, which would cost recall.

**Q9. What is AUC and why is it your headline number?**
It measures how well a model ranks positives above negatives across *all* thresholds, not at one arbitrary cutoff (0.5). It is therefore independent of our threshold choice, which makes it the fairest way to compare four models. ROC-AUC 0.9799 for XGBoost.

**Q10. Why did you add Precision-Recall curves when the brief only asked for ROC?**
Because at 8.8% positives the ROC curve flatters every model. SVM is the proof: a respectable ROC-AUC of 0.9619 but the **worst** PR-AUC of 0.7822. PR curves show performance in the region that actually matters. Reporting ROC alone would have hidden that.

**Q11. What is the majority baseline and why report it?**
The accuracy from always predicting the majority class — 0.9118 here, with recall 0. It is the minimum any classifier must beat, and most published papers on this dataset never state it.

**Q12. Why did you use `decision_function` instead of `predict_proba` for SVM?**
`SVC(probability=True)` is deprecated in scikit-learn 1.9 and performs internal Platt scaling. `decision_function` gives the ranking scores that AUC and PR-AUC need, without the extra machinery.

### C. Data handling

**Q13. Why did you drop 3,854 duplicate rows? Is that safe?**
They were exact full-row duplicates, and I checked before removing: only **one** duplicate group disagreed on the label. So no information was lost and no label conflict was hidden. We went from 100,000 to 96,146 rows.

**Q14. The data has no missing values, but you said 35.8% is missing. Why?**
`isnull()` reports zero NaN cells, but 35,816 rows hide missingness inside `smoking_history` as the **string** `"No Info"`. A pipeline trusting `isnull()` alone would treat them as complete.

**Q15. Why did you keep `"No Info"` as a category instead of imputing it?**
I measured it first. `"No Info"` has a **4.06%** diabetes rate against **9.53%–17.00%** for every genuine smoking category. Missingness that correlates that strongly with the target is *systematic*, not random. Imputing it would destroy real signal and would tell a doctor that 35,816 patients never smoked.

**Q16. The data has children aged 0.08 years. Why didn't you remove them?**
That is recorded as a limitation, and it is the honest conclusion. A minimum age of 0.08 and 911 infants under one year indicate a **synthetic generator**, not corrupted records. Blanking them would have deleted 17% of rows without fixing anything real. Keeping them is what let us *discover* the dataset is synthetic — which is why we make no clinical claims from it.

**Q17. Why not use SMOTE to improve the scores?**
Because it would have made the numbers meaningless. We would be evaluating on a resampled set that does not resemble a real screening population. Instead we used **class weighting** — `class_weight='balanced'` and `scale_pos_weight=10.34` — which corrects the imbalance at training time while leaving the test set at its true 8.82% prevalence.

**Q18. Why did you use a 70/15/15 split rather than 80/20?**
The approved plan specified 70/15/15, and I followed it, though I should flag that the validation portion ends up unused in the final pipeline since cross-validation handles model selection. The test set still holds 1,272 positives, which is plenty for the model differences to be meaningful. I am happy to change it to 80/20 if you prefer that convention.

### D. Method and rigour

**Q19. What is overfitting and how did you prevent it?**
Fitting the training set so well that performance collapses on unseen data. I made it *measurable* rather than asserting it: I fitted a deliberately unrestricted tree, which reached training AUC 1.0000 but only 0.8512 on validation — a gap of **0.1488**. The tuned tree reaches 0.9748 on test, so regularisation was worth roughly **0.12 AUC** here.

**Q20. How did you tune the models?**
5-fold stratified cross-validation, scored on ROC-AUC because it is threshold-independent. Selected values: Logistic Regression `C=0.01`; Decision Tree `max_depth=8`, `min_samples_leaf=50`; SVM `C=1.0`; XGBoost `n_estimators=300`, `max_depth=3`, `learning_rate=0.05`. Grids were deliberately kept small — the aim was to demonstrate regularisation and model selection, not to chase the last fraction of a percent.

**Q21. How do you know the test set was not used in training?**
It was held out from the start. All tuning used cross-validation *within* the training split, and the test set was touched exactly once, at final evaluation. Every preprocessing step — imputation, scaling, one-hot encoding — is a fitted estimator inside a `ColumnTransformer`, so no test statistic could leak backwards into training.

**Q22. Why did you keep all 15 features? Didn't you do feature selection?**
I did the analysis — correlation plus an ANOVA F-test per feature — and decided to keep everything, deliberately. With only 8 original features there is no dimensional-reduction problem to solve, and the brief requires one consistent feature set across all four models. Dropping the weaker columns would have saved negligible compute while making the comparison less fair. I report model-based importance separately, after fitting, because using it to select features beforehand would be circular.

**Q23. Your feature importance shows smoking_history = 0 for the Decision Tree. So smoking has no effect?**
No. A tree's zero means "never used for a split", not "no association". Once `HbA1c` and `glucose` are available, greedy splits have no need for smoking. Logistic Regression, which uses all features additively, gives smoking a weight of 0.127. The two models measuring the same condition differently and agreeing on the top three features — `HbA1c`, `glucose`, `age` — is the reassuring result.

### E. Critical thinking

**Q24. Is your model actually useful in a hospital?**
No — not on this data, and I would not claim otherwise. The dataset is synthetic with no institutional citation, there is no external or temporal validation, and the operating threshold was never tuned. What the project demonstrates is the *method* and the *reasoning*, not clinical readiness.

**Q25. What are the main limitations?**
Five: (1) the data is synthetic, so no clinical claim is supported; (2) no external validation; (3) the decision threshold was never optimised — `predict` uses 0.5; (4) the cost asymmetry is assumed, not modelled with a cost matrix; (5) SVM was fitted on a 20,000-row subsample because an RBF fit is roughly quadratic, so it is not perfectly comparable.

**Q26. What would you improve next?**
In order: tune the decision threshold on the validation set to hit a target recall; validate on a second, real dataset; replace SMOTE-free evaluation with cost-sensitive metrics tied to clinical costs; and refit the SVM on the full training split.

**Q27. If you had more time, which model would you push further?**
XGBoost, but with the caveat that its ceiling here is only 0.0051 above a single decision tree — the headroom is in the *data*, not the model. The signal sits in two lab measurements (`HbA1c`, glucose), so better features would help far more than a bigger ensemble.

---

## 5. If you are asked only one question

> **"Which model did you use, and why?"**

> *"We compared four models — Logistic Regression, Decision Tree, RBF SVM and XGBoost — chosen to span the interpretability-to-performance range, and excluded Linear Regression (wrong model family for a binary label) and KNN (no model artifact, so nothing a clinician could interpret). XGBoost performed best with an AUC of 0.9799, but we recommend the Decision Tree at 0.9748, because XGBoost's entire advantage is 0.0051 AUC — not enough to give up a rule set a doctor can audit. Our accuracy of 0.9034 sits below the 0.9118 majority baseline by design: we accepted false positives to reach 92.77% recall where the baseline reaches zero, and we deliberately kept the real 8.82% class prevalence instead of using SMOTE, so the reported metrics describe a real screening population rather than a resampled one."*

---

## 6. Numbers you may be asked to recall

| Fact | Value |
|---|---|
| Raw rows / after cleaning | 100,000 / 96,146 |
| Duplicates removed | 3,854 (only 1 group had label conflicts) |
| Class balance (cleaned) | 91.18% negative / 8.82% positive |
| `"No Info"` rows | 35,816 (35.8%), diabetes rate 4.06% |
| Genuine smoking categories' rate | 9.53% – 17.00% |
| Split | 67,302 train / 14,422 val / 14,422 test |
| Test positives | 1,272 |
| Majority baseline | 0.9118 accuracy, recall 0.0000 |
| Balanced accuracy (XGBoost) | 0.9144 |
| `scale_pos_weight` | 10.34 |
| Engineered features | 15 (6 numeric + 3 gender + 6 smoking) |
| Overfit gap, unrestricted tree | 0.1488 (train 1.0000 vs val 0.8512) |
| Overfit gap, deep XGBoost | 0.0341 |
| Children under 18 / infants under 1 | 17,219 / 911 |
| Positives among children | 82 (0.48% vs 8.82% overall) |
| Best CV AUC | XGBoost 0.9796 |
| Kaggle's own reported accuracy | **none published** |

---

*Cross-references: `project_plan.md` (the plan), `implementation_report.md` (what we built and how), `../README.md` and `../notebooks/Diabetes Risk Prediction.ipynb` (the full report).*
