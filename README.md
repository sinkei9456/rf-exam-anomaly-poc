# rf-exam-anomaly-poc

English | [한국어](README.ko.md)

Proof-of-concept code that checks, on synthetic data, whether hidden wireless devices used for exam cheating (Bluetooth earphones, AI glasses) can be found in receiver spectrograms. The signal shape of each device is not known in advance, so the methods learn only the normal state and flag deviations (anomaly detection).

There are two experiments.

- **Experiment 2 (2026-11-19, current direction):** per-site training. Normal IQ data is recorded at the site first, a lightweight model is trained on it, and the same site is then monitored. Sites: two indoor sessions (same room, different days) and a park. Methods: (A) pixel difference to the previous snapshot, (B) pixel difference to the per-pixel normal mean, and convolutional autoencoders (CAE) of three sizes. All use normal data only.
- **Experiment 1 (2026-11-18):** one park background. In addition to the methods above, a CNN and a CNN + CAE ensemble were compared.

| Document (Korean) | Content |
|---|---|
| [doc/00_overview_ver2.md](doc/00_overview_ver2.md) | Experiment 2 write-up |
| [doc/00_overview_rev2.md](doc/00_overview_rev2.md) | Experiment 2, copy-edited |
| [doc/00_overview_ver1.md](doc/00_overview_ver1.md) | Experiment 1 write-up |
| [doc/00_overview_rev1.md](doc/00_overview_rev1.md) | Experiment 1, copy-edited |
| [doc/prior_work.md](doc/prior_work.md) | Prior-work survey |

## Experiment 2 results: per-site training

**The anomaly signals in these experiments were not recorded from real cheating devices (AI glasses, Bluetooth earphones).** They are synthetic signals imitating the Bluetooth specification, added to real background IQ recordings. Nothing has been checked with real device signals yet.

The 1 s IQ files of each site are split in time order: earlier files for training, later files for testing. Weak Wi-Fi and Bluetooth bursts (normal) and two unseen Bluetooth patterns (anomalies) are synthesized on top. Test-set results are shown below. The threshold (the score above which a snapshot is called anomalous) is the top 0.1 % of the validation normal scores of each site. CAE values are means over 3 seeds.

- FPR: fraction of normal snapshots (no cheating-device signal) flagged as anomalous
- recall: fraction of snapshots with a visible cheating-device signal (synthetic) that are flagged
- SNR: how many dB the signal is above the background noise level within the same pixel. A pixel containing the signal is 3.0 dB brighter than the background at SNR 0 dB and 12.3 dB brighter at 12 dB. Normal Wi-Fi and Bluetooth were synthesized at 2–10 dB.
- Alarm rule: one snapshot per second; alarm when at least 2 of the last 10 snapshots (10 s) are anomalous.
- False alarms per hour: how often the alarm goes off per hour with no cheating-device signal present
- Event recall: fraction of 40 cases with a synthetic cheating-device signal added for 10 s in which the alarm went off

**Test FPR and false alarms per hour by site**

| Method | FPR (%), indoor | FPR (%), indoor (drone off) | FPR (%), park | False alarms per hour, indoor | False alarms per hour, indoor (drone off) | False alarms per hour, park |
|---|---:|---:|---:|---:|---:|---:|
| (A) previous-snapshot difference | 0.20 | 0.21 | 0.09 | 0.0 | 0.0 | 0.2 |
| (B) normal-mean difference | 0.14 | 0.10 | 0.11 | 0.2 | 0.0 | 0.2 |
| CAE, 65,265 parameters | 1.28 | 0.02 | 0.05 | 4.5 | 0.0 | 0.0 |
| CAE, 21,121 parameters | 1.11 | 0.05 | 0.03 | 4.1 | 0.0 | 0.0 |
| CAE, 6,489 parameters | 1.06 | 0.08 | 0.04 | 3.6 | 0.1 | 0.0 |

**Event recall at anomaly SNR 9 dB (%)**

| Method | Earphone, indoor | Earphone, indoor (drone off) | Earphone, park | BLE, indoor | BLE, indoor (drone off) | BLE, park |
|---|---:|---:|---:|---:|---:|---:|
| (A) previous-snapshot difference | 82.5 | 80.0 | 90.0 | 0.0 | 0.0 | 0.0 |
| (B) normal-mean difference | 87.5 | 57.5 | 97.5 | 0.0 | 0.0 | 0.0 |
| CAE, 65,265 parameters | 100.0 | 80.8 | 99.2 | 88.3 | 30.0 | 94.2 |
| CAE, 21,121 parameters | 95.8 | 86.7 | 95.8 | 73.3 | 23.3 | 66.7 |
| CAE, 6,489 parameters | 96.7 | 71.7 | 94.2 | 63.3 | 25.0 | 70.0 |

**Test FPR (%) when the CAE (65,265 parameters) trained at one site is used at another**

| Site the model was trained at | Tested at: indoor | Tested at: indoor (drone off) | Tested at: park |
|---|---:|---:|---:|
| indoor | 1.27 | 7.44 | 0.47 |
| indoor (drone off) | 2.68 | 0.02 | 0.30 |
| park | 16.90 | 12.46 | 0.05 |

**CAE size, model file size and compute time (NVIDIA DGX Spark)**

| Model | Trainable parameters | Model file (KB) | GPU training time, 20 epochs (s) | Scoring one snapshot on one CPU core (ms) |
|---|---:|---:|---:|---:|
| CAE, 65,265 parameters | 65,265 | 270 | 87 | 8.3 |
| CAE, 21,121 parameters | 21,121 | 97 | 47 | 6.6 |
| CAE, 6,489 parameters | 6,489 | 38 | 27 | 4.9 |

- Each site needs its own model. A CAE trained in the quiet park had an FPR of 2.65–16.90 % indoors.
- CAEs trained at the same site had FPRs of 0.02–0.08 % in the park and the indoor (drone off) site. Indoors the FPR was 1.06–1.28 %, and 62–74 % of those false alarms came from one scene in the test IQ file (a short real signal repeated three times at 2,451 MHz). With only 3–5 s of background per site, the FPR is driven by a handful of scenes.
- The CAE with 65,265 parameters had the highest recall. The 1/3 and 1/10 models lost recall at SNR 6–9 dB; at 12 dB the gap was at most 12 percentage points.
- All three CAEs are light. At one snapshot per second, even the largest uses less than 1 % of one CPU core.
- All results use synthetic anomaly signals. Nothing has been verified in a real exam room or with real devices.

## Experiment 1 results: one park background

Trained and tested on 3 s of park background. The threshold is the top 0.1 % of the validation normal scores; CNN, CAE and the ensemble are means over 3 seeds.

**Single-snapshot FPR and recall (%)**

| Method | FPR | recall, SNR 0–3 dB | recall, SNR 6 dB | recall, SNR 9 dB | recall, SNR 12 dB |
|---|---:|---:|---:|---:|---:|
| (A) previous-snapshot difference | 0.09 | 0.0–0.6 | 0.0–0.3 | 0.5–37.1 | 46.1–88.8 |
| (B) normal-mean difference | 0.11 | 0.0–0.6 | 0.0–0.3 | 2.7–56.5 | 70.9–92.6 |
| (C) CNN | 0.30 | 89.4–95.5 | 90.7–92.8 | 93.5–95.2 | 94.6–96.7 |
| (D) CAE | 0.05 | 0.0–2.2 | 18.4–41.2 | 73.9–76.8 | 85.4–91.9 |
| (E) CNN + CAE ensemble | 0.28 | 88.9–95.3 | 89.9–93.6 | 93.8–96.0 | 94.4–97.4 |

Ranges are the lower and higher values over the two anomaly types (earphone audio streaming, BLE data streaming). The CNN was trained to separate normal snapshots from normal snapshots with synthetic rectangular blobs; the blob sizes cover the test packets, which favors it over the CAE. The ensemble performed like the CNN (recall within 1 percentage point). Experiment 2 keeps only methods that use normal data, so the CNN and the ensemble were dropped.

## Repository layout

```
rf-exam-anomaly-poc/
├── data/
│   ├── background/      background IQ files: park Park02_1–3.npy (this release), indoor/ and indoor_droneoff/ (made by site_setup.py)
│   └── raw/hackrf/      raw indoor IQ data (from hackrf_3_test_indoor.tar in the rf-drone-detector release)
├── doc/                 four write-ups, figures (fig/ experiment 1, fig/site/ experiment 2), prior-work survey
├── models/              experiment 1 models (3 CNNs, 3 CAEs, normal statistics, thresholds, ensemble scaling)
│   └── site/            experiment 2 site folders (per site: 9 CAEs, normal statistics, background noise level, thresholds)
├── results/             experiment 1 reference results; results/site/ experiment 2 reference results
├── sites/               site folders you train for new sites (site_train.py output, not tracked)
├── src/                 training and evaluation code, inference pipeline
├── requirements.txt
└── LICENSE
```

## Install

Tested with Python 3.11. Install PyTorch for your platform following [pytorch.org](https://pytorch.org), then:

```
git clone https://github.com/sinkei9456/rf-exam-anomaly-poc.git
cd rf-exam-anomaly-poc
pip install -r requirements.txt
```

Training needs an NVIDIA GPU. Inference falls back to the CPU. Figures contain Korean text; install a Korean font (e.g. NanumGothic) to render it.

## Get the data

The background IQ files are attached to releases because of GitHub's file size limit. All were recorded during the D4D hackathon with a HackRF One (20 MS/s, LNA 16 dB, VGA 16 dB) with the drone and controller switched off, and are published under CC BY 4.0. Run from the repository root.

**Park (both experiments):** three files from this repository's release (480 MB).

```
curl -LO https://github.com/sinkei9456/rf-exam-anomaly-poc/releases/download/v1.0/background_park02.tar
tar xf background_park02.tar        # data/background/Park02_1.npy, Park02_2.npy, Park02_3.npy
cd data/background && sha256sum -c SHA256SUMS.txt && cd ../..
```

Each file is 1 s (complex64); files 1, 2, 3 are the background for the training, validation and test sets. They come from the park session in [rf-drone-detector](https://github.com/sinkei9456/rf-drone-detector) release file hackrf_1_train_background.tar; Bluetooth-shaped blobs were removed and the 3 s were split into 1 s files (src/neo2_pieces.py in that repository).

**Two indoor sites (experiment 2):** download and extract hackrf_3_test_indoor.tar (600 MB) from the rf-drone-detector v1.0 release, then convert the 1 s .cs8 files.

```
curl -LO https://github.com/sinkei9456/rf-drone-detector/releases/download/v1.0/hackrf_3_test_indoor.tar
tar xf hackrf_3_test_indoor.tar     # data/raw/hackrf/1_nodrone_indoor/..., data/raw/hackrf/2_neo2_indoor/...
python src/site_setup.py            # data/background/indoor/01–05.npy, data/background/indoor_droneoff/01–05.npy (1.5 GB)
```

| Site | Source session | Center frequency (MHz) | Training | Validation | Test |
|---|---|---:|---|---|---|
| indoor | 1_nodrone_indoor/20261007_antenna_on | 2,450.0 | files 1–3 | file 4 | file 5 |
| indoor (drone off) | 2_neo2_indoor/20261008_drone_off | 2,450.5 | files 1–3 | file 4 | file 5 |
| park | 3_neo2_park/02_telescopic_OFF_near1m | 2,449.5 | file 1 | file 2 | file 3 |

## Usage 1: a new site (train, set thresholds, infer)

Record IQ data at the site in the normal state (no cheating device), train the site's model on it, then score data recorded during the exam with that model. Two commands:

```
python src/site_train.py --out sites/study_cafe --normal normal_capture.cs8     # train, pick kernels and thresholds, save the site folder
python src/predict.py --site sites/study_cafe --iq exam_capture.cs8             # same preprocessing as in training, then decisions and alarms
```

site_train.py follows the experiment 2 rules, without synthetic normal signals (real site data already contains the site's normal traffic):

1. Split the normal data in time order: first 75 % training, last 25 % validation.
2. Measure the background noise level on the training part, divide all data by it, and turn 20 ms pieces into 256 × 97 snapshots.
3. Fit the per-pixel statistics and kernel of (B) normal-mean difference.
4. Train the CAE(s); pick the epoch by validation reconstruction error and the kernel with shapeless rectangles added to validation snapshots.
5. Save the top 0.1 % and top 1 % of the validation normal scores as decision thresholds.

The site folder holds site.json (background noise level, kernels, thresholds, training log), pixel_stats.npz and cae<size>_s<seed>.pt. predict.py --site divides new IQ data by the stored noise level, so keep the receiver and its settings (center frequency, 20 MS/s, gain) the same as in training.

| site_train.py option | Default | Meaning |
|---|---|---|
| `--normal FILE ...` | (required) | normal IQ data of the site, in recording order; same formats as predict.py (.npy, .cf32, .cs8, .sigmf-data) |
| `--out DIR` | (required) | site folder |
| `--layout` | chunks | chunks = concatenated 20 ms pieces (predict.py input format), continuous = gap-free recordings (random 20 ms crops) |
| `--sizes` | 65k | CAE sizes (any of 65k, 21k, 6k): 65,265, 21,121 or 6,489 parameters |
| `--seeds` | 0 | CAE seeds (one or more) |
| `--epochs` | 20 | training epochs |
| `--val-frac` | 0.25 | fraction at the end used for validation |
| `--n-train`, `--n-val` | 10000, 5000 | snapshots cropped with continuous; with chunks, --n-train sets the training amount per epoch |

| predict.py --site option | Default | Meaning |
|---|---|---|
| `--model` | cae | cae or pixel ((B) normal-mean difference) |
| `--cae` | default_cae in site.json | which CAE, e.g. 65k_s0, 21k_s0 |
| `--fpr` | 0.001 | 0.001 = top 0.1 % of the validation normal scores as threshold, 0.01 = top 1 % |
| `--k`, `--n` | 2, 10 | alarm when k of the last n snapshots are anomalous |
| `--noise-ref FILE` | none | re-measure the noise level on this normal data instead of using the stored value |

**Record about 30 minutes of normal data (at least 10).** With a 20 ms piece every 0.1 s, 30 minutes give about 18,000 snapshots (about 14 GB as HackRF .cs8) and 10 minutes give 6,000 (about 4.8 GB). One piece per second leaves too few validation snapshots (the last 25 %) to set the threshold, so record several pieces per second. site_train.py warns when there are fewer than 1,000 validation snapshots.

**End-to-end check on real IQ data:** trained on indoor (10-07) 1 s files 1–4 and scored file 5, which was not used in training. Training all three CAE sizes took 217 s (NVIDIA DGX Spark, including 44 s of snapshot generation).

```
H=data/raw/hackrf/1_nodrone_indoor/20261007_antenna_on/recordings
python src/site_train.py --layout continuous --out sites/indoor --sizes 65k 21k 6k --normal $H/0[1-4]/*.cs8
python src/make_site_demo_iq.py --background $H/05/*.cs8 --out data/site_demo_iq.npy --kind 1 --snr 9
python src/predict.py --site sites/indoor --iq data/site_demo_iq.npy
```

| Check | CAE, 65,265 parameters | (B) normal-mean difference |
|---|---|---|
| file 5 as recorded (50 normal 20 ms pieces) | 3 flagged, 1 alarm | 2 flagged, 1 alarm |
| 1 min (60 pieces) from file 5 with an earphone-type anomaly (SNR 9 dB) in pieces 20–29 | alarm 21–37 s on the anomaly, 1 false alarm in the normal part (45–50 s) | alarm 21–37 s on the anomaly, 2 false alarms (9–18 s, 46–52 s) |

An alarm stays on until 10 snapshots after the last anomalous one, so it ends at 37 s. The false alarms in the normal part come from having only 3 s of training data, the same limitation as in experiment 2. The experiment 2 models are in models/site/ in the same site-folder format (`python src/predict.py --site models/site/indoor --iq ...`). The park folder's noise level is in the units of this release's Park02_*.npy files, so pass `--noise-ref` for other park recordings.

## Usage 2: reproduce experiment 2 (per-site training)

```
bash src/run_sites.sh
```

The steps below run for each of the three sites, followed by the cross-site evaluation and the result tables. About 35 min on one GPU.

| Step | Script | What it does | Time per site |
|---|---|---|---:|
| 1 | src/make_data.py | synthesize 41,400 snapshots (training 10,000, validation 7,000, test 24,400; data/site/<site>/, about 1.0 GB) | ~1 min 40 s |
| 2 | src/pixeldiff.py | (A)(B): pick smoothing size and thresholds on the validation set, score | ~20 s |
| 3 | src/cae.py --size 65k, 21k, 6k | train the three CAE sizes, 3 seeds each | ~9 min |
| 4 | src/site_evaluate.py | test-set evaluation, model size and timing, appendix figures (doc/fig/site/) | ~10 s |
| 5 | src/site_cross.py | FPR of models trained at another site (once, after all sites) | ~2 min 30 s |
| 6 | src/site_summary.py | result tables (runs/site/results.md) and summary figures | seconds |

- To run one site, set EXAM_SITE (indoor, indoor_droneoff, park) and call the scripts, e.g. `EXAM_SITE=park python src/make_data.py`. Without EXAM_SITE every script runs the experiment 1 setup.
- Finished steps (existing data files, scores_pixel.npz, cae<size>_s2.pt) are skipped.
- Compare runs/site/results.md with the reference results in results/site/results.md.
- The two explanatory figures of the write-up: `EXAM_SITE=park python src/site_figures.py snr`, `EXAM_SITE=indoor python src/site_figures.py dense`.

## Usage 3: reproduce experiment 1 (one park background)

```
bash src/run_all.sh                         # everything, about 30 min on one GPU
bash src/run_all.sh --use-released-models   # skip training, score with models/ (about 8 min)
```

| Step | Script | What it does | Time |
|---|---|---|---:|
| 1 | src/make_data.py | synthesize 55,000 snapshots (training 20,000, validation 7,000, test 28,000; data/, 1.3 GB) | ~3 min |
| 2 | src/pixeldiff.py | (A)(B): pick smoothing size and thresholds on the validation set, score | ~1 min |
| 3 | src/cnn.py | (C) CNN training, 3 seeds | ~20 min |
| 4 | src/cae.py | (D) CAE training, 3 seeds | ~5 min |
| 5 | src/ensemble.py | (E) ensemble: scale CNN/CAE scores on validation normals, take the larger; no training | seconds |
| 6 | src/evaluate.py | test-set evaluation, tables (runs/results.md), figures (doc/fig/) | ~3 min |
| 7 | src/compare_results.py | compare with the reference results (results/) | seconds |

Use `PY=/path/python` to choose the interpreter and `WORKERS` (default 16) for data generation (also for run_sites.sh). Existing data/*.npy files are reused. After retraining, `python src/export_models.py` copies the new models into models/.

## Usage 4: inference with the experiment 1 models

Without --site, src/predict.py scores with the experiment 1 (park) models in models/. For a new site, use a site folder trained on that site's data (Usage 1). To get only the trained models and code, download a release file: rf-exam-anomaly-inference-v1.0.zip (2.6 MB, experiment 1) in v1.0, or rf-exam-anomaly-site-v2.0.zip (experiment 2 and new-site training) in v2.0. Try it on a demo IQ file (1 min, earphone-type anomaly at SNR 6 dB in seconds 20–29):

```
python src/make_demo_iq.py
python src/predict.py --iq data/demo_iq.npy --noise-ref data/demo_noise_ref.npy --model cnn
python src/predict.py --iq data/demo_iq.npy --noise-ref data/demo_noise_ref.npy --model ens
```

| Input option | Meaning |
|---|---|
| `--iq FILE` | IQ data: 20 ms chunks (20 MS/s, 400,000 samples) captured once per second and concatenated. .npy (complex), .cf32, .cs8 or .sigmf-data (HackRF int8) |
| `--snapshots FILE` | precomputed snapshots (uint8, N × 256 × 97), as written by src/make_data.py |
| `--eval-sets` | score the generated validation and test sets into runs/ (used by --use-released-models) |

| Option | Default | Meaning |
|---|---|---|
| `--model` | cnn | cnn = (C), cae = (D), ens = (E) ensemble, pixel = (B) |
| `--seed` | 0 | CNN/CAE/ensemble seed (0, 1, 2); ensemble seed k combines CNN seed k and CAE seed k |
| `--fpr` | 0.001 | which threshold to use: 0.001 = top 0.1 % of the validation normal scores, 0.01 = top 1 % |
| `--noise-ref FILE` | none | normal IQ data used to normalize the noise power (recommended: same site and receiver settings) |
| `--k`, `--n` | 2, 10 | alarm when k of the last n snapshots are anomalous |
| `--interval` | 1.0 | seconds between snapshots (time column only) |
| `--out` | runs/predict.csv | output CSV: index, time_s, score, threshold, anomaly, alarm |

For your own recordings, keep the receiver settings of the training data (HackRF, 2,449.5 MHz, 20 MS/s, fixed gain) and give a normal reference with `--noise-ref`. As experiment 2 shows, the FPR changes a lot between sites: for a new site, retrain on that site's normal data (Usage 1).

## Reproducibility

**Experiment 1:** both paths were run from a fresh copy of this repository and compared with results/ (NVIDIA DGX Spark GB10, ARM CPU, Python 3.11, PyTorch 2.14, CUDA 13).

- With the released models (`--use-released-models`): all 52 metrics of all 11 models (including the 3 ensembles) were identical, as were the synthetic data and pixel-difference results.
- Full retraining (`bash src/run_all.sh`): pixel-difference results were identical; CNN, CAE and ensemble numbers changed slightly even on the same machine. Per seed, a single metric differed by up to 9.3 percentage points (CNN seed 0).
- The 3-seed means used in the write-up moved much less (table below). The conclusions do not change.

Full retraining vs. reference results (3-seed means; threshold = top 0.1 % of the validation normal scores)

| Model | Largest snapshot recall difference (points) | Largest event recall difference (points) | Test FPR, reference (%) | Test FPR, retrained (%) | Alarms per hour, reference | Alarms per hour, retrained |
|---|---:|---:|---:|---:|---:|---:|
| (C) CNN | 2.7 | 0.0 | 0.30 | 0.27 | 0.5 | 0.4 |
| (D) CAE | 0.5 | 1.7 | 0.05 | 0.05 | 0.0 | 0.0 |
| (E) CNN + CAE ensemble | 2.2 | 0.0 | 0.29 | 0.18 | 0.4 | 0.2 |

**Experiment 2:** results/site/ comes from one run on the same machine on 2026-11-19. In a fresh copy of this repository, the 10 indoor background files made by site_setup.py from the downloaded archive were identical to those used in the study. PyTorch does not guarantee bit-identical GPU training, so CAE numbers can differ slightly on a rerun.

## Limitations

- Each site has only 3–5 s of background IQ data. In practice, several minutes of normal data per site are needed.
- Normal Bluetooth traffic was synthesized as short packets only (≤ 0.4 ms).
- The two anomaly patterns imitate the Bluetooth specification; they are not recordings of real devices. AI glasses may also use Wi-Fi to talk to the phone; this was not checked.
- The two indoor sites are the same room on different days with different receiver settings (center frequency, filter, capture software).
- Seat localization and body shadowing are not covered.

## License and sources

- Code: MIT (LICENSE)
- Background IQ data, documents and figures: CC BY 4.0
- Original background recordings: rf-drone-detector, https://doi.org/10.5281/zenodo.23656790
