# Per-patient seizure forecasting on EEG

Most seizure-prediction work reports one average sensitivity across a whole
cohort. That number is not much use to an individual patient: a model at 92%
average can still sit at 70% on the person actually wearing it. This project
asks a narrower question instead. Can we promise a *specific* patient that the
forecaster will catch at least 90% of their seizures, and can we tell in
advance which patients that promise will hold for?

The short version of what we found:

- Training a per-patient classifier to separate preictal from interictal EEG
  works in-sample and falls apart on held-out seizures. Sensitivity drops to
  0.43 even though training accuracy is above 0.97.
- The reason is that a patient's seizures do not look like each other. We split
  the calibration seizures randomly and then in time order to separate
  seizure-to-seizure variation from drift over the recording. Essentially all
  of the failure is variation between seizures, not drift.
- Scoring each window by how far it sits from the patient's own interictal
  baseline, rather than training a discriminator, recovers most of the loss
  (0.78 under leave-one-seizure-out).
- Setting the alarm threshold from the interictal data, which is plentiful,
  rather than from the handful of calibration seizures, takes per-patient
  sensitivity to 0.94 at a realised false-alarm rate of 0.20.
- A detectability index computed before deployment predicts which patients the
  target will be met for (r = -0.86). Refusing to serve the patients it flags
  leaves every remaining patient above target.

## Data

| Source | Subjects used | Type |
| --- | --- | --- |
| CHB-MIT Scalp EEG Database | 21 extracted, 18 evaluated | 23-channel scalp, 256 Hz, paediatric |
| Siena Scalp EEG Database | 4 | adult scalp |
| AES Seizure Prediction Challenge | 7 (5 dogs, 2 humans) | intracranial depth electrodes |

None of the recordings are in this repository. All three are public; the
notebook reads them from the paths they mount at on Kaggle, which you will need
to change if you run it elsewhere.

From CHB-MIT we dropped chb06 and chb24, whose seizures are packed too closely
together for any segment to satisfy our interictal separation rule, and chb12,
whose montage is inconsistent across files. Of the 21 patients left, 18 have at
least three seizures, which is the minimum for calibrating on some seizures and
testing on others.

## How the pipeline works

**Labelling.** For a seizure at time T, the preictal window runs from T-35 min
to T-5 min. The last 5 minutes before onset are thrown away deliberately, so
the model has to forecast rather than detect an onset already underway. The
seizure itself, the hours after it, and anything within 4 h of a seizure are
discarded. Interictal data comes only from recordings with no seizure in them,
at least two files away from any recording that does have one.

Everything is band-pass filtered 0.5-50 Hz with a 60 Hz notch and cut into
non-overlapping 30 s windows. Non-overlapping matters: overlapping windows
would put near-duplicates on both sides of the calibration/test split.

**Features.** Per window and per channel: relative power in five bands (delta,
theta, alpha, beta, gamma), line length, and spectral entropy. Channel counts
differ between patients because we intersect the channel sets across each
patient's own files, but since every model is fit per patient that never
matters.

**Scoring.** Each window gets a Mahalanobis distance to the patient's
interictal distribution. A seizure's score is the highest score among its
preictal windows, and it counts as forecast when that exceeds the threshold.
Nothing preictal is used to fit the score, which is the whole point: the only
assumption is that preictal data looks unusual against interictal, and that
assumption carries across seizures even when a shared preictal signature does
not.

**Threshold.** Set as a percentile of the interictal scores from a calibration
split, with the false-alarm rate reported on a disjoint interictal split rather
than on the data that set the threshold. Results are averaged over 10 random
calibration/test partitions.

**Abstention.** The detectability index D is the fraction of a patient's
seizures whose peak score does not clear the interictal 95th percentile. It
uses no test labels, so it is available before deployment. Serving only
patients with D below 0.10 means every patient served meets the target.

## Results

CHB-MIT, 18 patients, target sensitivity 0.90. All figures are what the
notebook prints.

| Approach | Mean sensitivity | Patients meeting 0.90 | False-alarm rate |
| --- | --- | --- | --- |
| Classifier + conformal threshold | 0.426 | 4/18 | 0.044 |
| Same, with adaptive threshold | 0.389 | 1/18 | 0.041 |
| Classifier + PCA, leave-one-seizure-out | 0.517 | 0/18 | 0.039 |
| Mahalanobis anomaly, leave-one-seizure-out | 0.777 | 1/18 | 0.085 |
| Mahalanobis + interictal calibration | 0.939 ± 0.004 | 13.8/18 | 0.199 ± 0.002 |
| Same, serving only patients with D < 0.10 | 1.000 | 14/14 | as above |

Relaxing the false-alarm budget trades monotonically: 0.960 sensitivity at a
realised rate of 0.261, 0.963 at 0.321. Standard deviations across the 10
partitions are small, so the operating point is stable.

The four patients who miss the target are chb01, chb03, chb04 and chb13. They
average 30% anomaly-indistinguishable seizures against 3% for the patients who
pass, and the correlation between that fraction and realised sensitivity is
r = -0.859 (p < 0.0001).

Splitting calibration seizures randomly gives mean sensitivity 0.946; splitting
them in time order gives 0.963. The two are close enough that drift contributes
nothing measurable, which also means online methods built to track drift would
not help here.

On the intracranial cohort the method runs unchanged and reaches 1.000
sensitivity on all seven subjects, against 0.824 for the classifier baseline,
at false-alarm rates of 0.22 to 0.46. D is zero for every subject. Intracranial
recordings are simply cleaner, so that cohort cannot exercise the abstention
logic at all.

Siena was inconclusive. The recordings are short enough that there are too few
interictal windows to estimate a stable high-dimensional covariance, and we had
to relax the interictal gap from 4 h to 1 h just to get usable segments. We
report it rather than quietly dropping it, but we would not read anything into
the numbers.

## What is in here

```
notebooks/
  01_chbmit_features_and_baseline.ipynb          19 cells
  02_chbmit_calibration_and_detectability.ipynb  22 cells
  03_siena_and_decomposition.ipynb               35 cells
  04_full_pipeline_with_ieeg.ipynb               50 cells
  05_full_pipeline_with_ieeg.ipynb               50 cells
requirements.txt
```

These are the five Kaggle sessions the work was done in, kept in order. Each
one extends the one before it: the first 19 cells of every file are the same 19
cells, the first 22 are the same 22, and so on. The last file is the complete
pipeline. The names describe how far each session got, not five separate
programs. Notebooks 04 and 05 are the same notebook saved twice.

Read notebook 05 if you want the finished thing. The earlier files are worth keeping
because they hold outputs the later ones lost. Kaggle clears a cell's stored
output when a session restarts without re-running it, so the CHB-MIT results in
cells 1-21 survive only in the first two files, while the Siena and
intracranial results in cells 22-49 survive only in the last two. No single
file carries all of them.

| Where the outputs live | Cells | Covers |
| --- | --- | --- |
| 01 | 1-18 | CHB-MIT up to the separability analysis |
| 02 | 1-21 | the above, plus the tradeoff curve and detectability index |
| 03, 04, 05 | 22-34 | Siena, abstention, heterogeneity vs drift |
| 04, 05 | 35-49 | intracranial cohort and cross-modality comparison |

Taking notebook 05 as the reference, the pipeline runs in these stages:

| Cells | Stage |
| --- | --- |
| 0-5 | locate the data, parse the seizure annotations, build the inventory |
| 6-8 | window labelling, feature extraction, cohort selection |
| 9-15 | classifier baseline, including the adaptive-threshold variant |
| 16-18 | Mahalanobis anomaly score, per-seizure separability |
| 19-21 | interictal calibration, tradeoff curve, detectability index |
| 22-34 | Siena, and the heterogeneity-versus-drift decomposition |
| 35-49 | intracranial cohort and the cross-modality comparison |

Cells are left in the order they were written, which includes the dead ends.
The classifier sections are kept on purpose, since the fact that they fail is
the reason the rest of the pipeline looks the way it does.

Figures are written to `artifacts/figures/` as PNG and PDF when the notebook
runs. They are not committed.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/05_full_pipeline_with_ieeg.ipynb
```

Paths are hard-coded to Kaggle mounts near the top of the data-loading cells.
Point `BASE`, `SIENA_BASE` and `AES_BASE` at wherever you have the data and the
rest follows. Feature extraction over all of CHB-MIT is the slow part and takes
a few hours; it caches to `artifacts/<patient>_features.npz`, so the analysis
cells can be re-run on their own afterwards.

NumPy 2.0 or newer is required, since the feature code uses `np.trapezoid`.

## Limitations

The main evaluation is paediatric scalp EEG from a single database, and the
intracranial cohort was too easy to test the part of the method we care most
about. A realised false-alarm rate of 0.20 is higher than anyone would want to
live with continuously. The features and the score are both deliberately plain,
so we cannot yet say whether a better score would raise the detectability
ceiling or whether that ceiling is a real property of the signal.

## Team

- Lalitaditya Tickoo
- Md Samar Aazmi
- Ayushman Mayank
- Tushar Biswas
