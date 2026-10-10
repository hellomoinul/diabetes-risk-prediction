# Running this project on Google Colab

Three routes. **Route A (upload)** is the quickest. **Route B (GitHub)** is better if you need to
re-run it repeatedly or share the link. **Route C (exam)** covers handing in a *different* dataset,
given as a CSV **or** a Kaggle link.

Both A and B were tested by extracting the archive to a clean directory and running the pipeline
there — it completed with byte-identical results (XGBoost AUC 0.9799, 16 figures, 12 result files)
and without needing Kaggle credentials.

---

## Route A — upload the archive (recommended)

### 1. Get the archive

`ML_lab_colab.zip` is already in the project root (2.6 MB, 28 files). It contains `src/`,
`data/`, the notebook, `models/` (the four trained pipelines, so scoring works instantly),
the exam scripts (`predict.py`, `run_on_new_data.py`, `run_generic_pipeline.py`,
`selftest_predict.py`), and the docs (`README.md`, `COLAB.md`, `PREDICT.md`, `Plan/`).
It deliberately excludes `figures/` and `results/`, because the run regenerates both.

**No upload needed:** the same bundle is attached to the GitHub release — pull it in one
cell and skip step 2 entirely:

```python
!wget -q https://github.com/hellomoinul/diabetes-risk-prediction/releases/download/v1.0/ML_lab_colab.zip
!unzip -q ML_lab_colab.zip -d /content
```

(Rebuilds replace the asset with `--clobber`, so this URL always serves the current bundle.)

### 2. Open Colab and upload it

Go to [colab.research.google.com](https://colab.research.google.com), then:

**Runtime → Change runtime type → CPU** (nothing here needs a GPU; the whole run is ~3 minutes).

Use the upload button in the file panel (or `Ctrl+Shift+P` → *Upload file*) and select
`ML_lab_colab.zip`.

### 3. Unzip it

```python
!unzip -q ML_lab_colab.zip -d /content
```

### 4. (Optional) verify before the notebook

```python
!cd /content/ML_lab_project && python run_pipeline.py
```

Takes ~3 minutes and prints the whole build: cleaning decisions, split sizes, per-model tuning
scores, and the required result table. Worth doing first — if this works, the notebook will too.

### 5. Open the notebook

In the Colab file panel, navigate `ML_lab_project` → `notebooks` →
`Diabetes Risk Prediction.ipynb`, and double-click to open it.

### 6. Run it

**Runtime → Run all**, or `Ctrl+F9`. Takes about 3 minutes, most of it the 5-fold cross-validation
grid searches.

### 7. Save a copy (optional)

**File → Download as → Notebook (.ipynb)** to get the executed notebook with all 16 figures
embedded, the same file you produce locally.

---

## Route B — GitHub clone

Better if you will run this more than once, or want a link you can share.

The project is on GitHub at
<https://github.com/hellomoinul/diabetes-risk-prediction>. In Colab:

```python
!git clone https://github.com/hellomoinul/diabetes-risk-prediction.git /content/ML_lab_project
```

then open `/content/ML_lab_project/notebooks/Diabetes Risk Prediction.ipynb` from the file panel.
Because cloning lands everything in `/content/ML_lab_project`, Route A's unzip cell can be skipped
(the setup cell finds `src/` automatically).

**One-click open (browser bookmark):** the notebook carries a bootstrap cell that clones the repo
itself when `src/` is missing, so this link *just works* with no manual preparation:
<https://colab.research.google.com/github/hellomoinul/diabetes-risk-prediction/blob/main/notebooks/Diabetes%20Risk%20Prediction.ipynb>

---

## Route C — exam: any dataset, CSV or Kaggle link

When the teacher hands over a **new dataset**, `run_on_new_data.py` decides the workflow for you:

- the columns are the **same 8 diabetes features** → score the rows with the saved model
  (`predict.py`, no retraining);
- **different columns** (still binary classification) → retrain the same 4-model workflow
  (`run_generic_pipeline.py`).

Same meaning under a different column name (e.g. `glucose` for `blood_glucose_level`)? Map it
instead of retraining: `python predict.py exam.csv --map glucose:blood_glucose_level`. Full exam
runbook: [`PREDICT.md`](PREDICT.md). The archive already ships `models/`, so predict-mode needs
no rebuild first; verify with `python selftest_predict.py` (~30 s).

### 1. Data arrives as a CSV

Upload it to `/content` with the file panel, then:

```python
!git clone https://github.com/hellomoinul/diabetes-risk-prediction.git /content/ML_lab_project
%cd /content/ML_lab_project
!python run_on_new_data.py --csv /content/exam.csv
```

### 2. Data arrives as a Kaggle link

Authenticate once (upload `kaggle.json` from kaggle.com/settings):

```python
!pip install -q kaggle
from google.colab import files; files.upload()
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

Then download and run in one command:

```python
!git clone https://github.com/hellomoinul/diabetes-risk-prediction.git /content/ML_lab_project
%cd /content/ML_lab_project
!python run_on_new_data.py --kaggle "https://www.kaggle.com/datasets/<owner>/<slug>"
```

### 3. Optional flags

```python
!python run_on_new_data.py --csv /content/exam.csv --target churn        # label column name
!python run_on_new_data.py --csv /content/exam.csv --positive-label 1   # which class is positive
!python run_on_new_data.py --csv /content/exam.csv --mode retrain        # force retrain
!python run_on_new_data.py --csv /content/exam.csv --outdir results_exam # where outputs go
```

Outputs land in `results_new/` (or `--outdir`): for the same-schema case a `predictions.csv` plus
metrics; for the different-schema case the full result table, figures, tuned models and a
`summary.json`. `run_on_new_data.json` beside them records which path ran and why.

---

## Things worth knowing

**No Kaggle credentials needed.** `fetch()` only calls the Kaggle API when
`data/diabetes_prediction_dataset.csv` is missing. The archive already contains that file, so the
data step re-reads and re-validates it locally and never touches the network. If you *do* upload
without the data file, you get a clear message telling you exactly what to do rather than an
opaque auth error.

**Dependencies are handled automatically.** The second code cell installs only what is missing, so
it is a silent no-op if Colab's preinstalled stack already covers it. To pin exact versions
instead, run `!pip install -r requirements.txt` from inside `/content/ML_lab_project` first — but
be aware the pins are strict (pandas 3.0.6, scikit-learn 1.9.1) and may not have wheels for
Colab's Python build. The setup cell prints the versions it found either way.

**Paths are location-independent.** Every path in `src/` is derived from `__file__`, not the
working directory, so the project works from any location. The notebook's setup cell searches both
the working directory *and* `/content`, which is why it needs no editing for Colab.

**Opening the notebook directly from GitHub works too.** A bootstrap cell at the top of the
notebook clones the repo into `/content/ML_lab_project` on its own whenever it cannot find `src/`,
so Route B's one-click link needs no manual preparation.

**Runtime is ~3 minutes** (181 s measured for the pipeline, ~2.5 min for the notebook). Colab's
free tier allows far more.

**CPU is enough.** There is no GPU code in this project; XGBoost runs on the 67k-row training
split fine on CPU.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No folder containing 'src/' was found` | The archive has not been unzipped, or you opened the notebook before step 3 | Run the unzip cell, then re-run the setup cell |
| `FileNotFoundError: ... diabetes_prediction_dataset.csv is missing and no Kaggle credentials` | You uploaded the notebook without the data file | Re-upload `ML_lab_colab.zip` (it contains `data/`) |
| A `pip install` cell appears to do nothing | Correct behaviour | Colab already provides these packages; the cell only installs what is absent |
| Joblib `resource_tracker KeyError` tracebacks | Cosmetic Windows/joblib shutdown noise | Ignore — the run still completes and writes its output |
| Results differ from the documented values | Almost impossible — `random_state = 42` everywhere | Check you ran all cells, and that you started from the unmodified archive |

---

## Sanity check

Whichever route you use, the final cell of section 7 must print:

```
               2. AUC  3. Accuracy  4. Precision  5. Recall  6. F1-Score
1. Model list
Log St         0.9639       0.8878        0.4336     0.8876       0.5826
DT             0.9748       0.8807        0.4204     0.9316       0.5793
SVM            0.9619       0.8960        0.4548     0.9009       0.6044
XGBoost        0.9799       0.9034        0.4756     0.9277       0.6288
```

If it matches, the Colab run reproduced the documented results exactly.