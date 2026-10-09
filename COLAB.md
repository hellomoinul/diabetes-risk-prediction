# Running this project on Google Colab

Two routes. **Route A (upload)** is the quickest. **Route B (GitHub)** is better if you need to
re-run it repeatedly or share the link.

Both were tested by extracting the archive to a clean directory and running the pipeline there —
it completed with byte-identical results (XGBoost AUC 0.9799, 16 figures, 12 result files) and
without needing Kaggle credentials.

---

## Route A — upload the archive (recommended)

### 1. Get the archive

`ML_lab_colab.zip` is already in the project root (2.5 MB). It contains `src/`, `data/`, the
notebook, the pipeline entry point, and the docs. It deliberately excludes `figures/` and
`results/`, because the run regenerates both.

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

Push the project to a GitHub repository, then in Colab:

```python
!git clone https://github.com/<your-user>/<your-repo>.git /content/project
```

then open `/content/project/notebooks/Diabetes Risk Prediction.ipynb` from the file panel.

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