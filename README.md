# rf-exam-anomaly-poc

English | [한국어](README.ko.md)

Proof-of-concept code that checks, on synthetic data, whether hidden wireless devices used for exam cheating (Bluetooth earphones, AI glasses) can be found in receiver spectrograms. Five anomaly detection methods that learn the normal state and flag deviations are compared:

- (A) pixel difference to the previous snapshot
- (B) pixel difference to the per-pixel normal mean (z-score)
- (C) CNN: a classifier trained to separate normal snapshots from normal snapshots with synthetic rectangular energy blobs
- (D) CAE (convolutional autoencoder): trained on normal snapshots only; the part it cannot reconstruct is the anomaly
- (E) CNN + CAE ensemble: CNN and CAE scores of the same seed are put on a common scale using validation normal snapshots, and the larger one is used

The full write-up (in Korean) is [doc/00_overview.md](doc/00_overview.md), also available as a web page with a table of contents ([write-up web page](https://sinkei9456.github.io/rf-exam-anomaly-poc/doc/00_overview.html)).

## Results

Background noise is 3 s of signal-free IQ data recorded in a park. Weak Wi-Fi and Bluetooth bursts (normal) and two unseen Bluetooth patterns (anomalies) are synthesized on top. Test-set results below use a threshold (the score above which a snapshot is called anomalous) set so that only 0.1 % of the validation normal snapshots (5 of 5,000) exceed it; CNN, CAE and the ensemble are means over 3 seeds.

- FPR: fraction of normal snapshots flagged as anomalous
- recall: fraction of snapshots with a visible cheating-device signal that are flagged
- SNR: per-Hz signal power above the noise floor. Normal Wi-Fi and Bluetooth were synthesized at 2–10 dB.

**Single-snapshot FPR and recall (%)**

| Method | FPR | recall, SNR 0–3 dB | recall, SNR 6 dB | recall, SNR 9 dB | recall, SNR 12 dB |
|---|---:|---:|---:|---:|---:|
| (A) previous-snapshot difference | 0.09 | 0.0–0.6 | 0.0–0.3 | 0.5–37.1 | 46.1–88.8 |
| (B) normal-mean difference | 0.11 | 0.0–0.6 | 0.0–0.3 | 2.7–56.5 | 70.9–92.6 |
| (C) CNN | 0.30 | 89.4–95.5 | 90.7–92.8 | 93.5–95.2 | 94.6–96.7 |
| (D) CAE | 0.05 | 0.0–2.2 | 18.4–41.2 | 73.9–76.8 | 85.4–91.9 |
| (E) CNN + CAE ensemble | 0.28 | 88.9–95.3 | 89.9–93.6 | 93.8–96.0 | 94.4–97.4 |

Ranges are the lower and higher values over the two anomaly types (earphone audio streaming, BLE data streaming).

- Pixel differences only find anomalies stronger than the normal signals.
- The CAE, trained on normal data only, reaches about 3 dB weaker signals than the pixel differences and has the fewest false alarms.
- The CNN finds anomalies at almost any SNR, but its synthetic training blobs cover the size of the test packets, which favors it over the CAE.
- The ensemble performed like the CNN (recall within 1 percentage point) with slightly less seed-to-seed variation in false alarms. The test signals lie inside the size range the CNN was trained on, so the CAE had little to add.
- The receiver sees only 20 MHz, so 17–40 % of snapshots show nothing even while a device transmits. Decisions are therefore made over 10 s ("alarm if 2 of the last 10 snapshots are anomalous"). With this rule the false alarms per hour were 0–1.3 (CNN), 0 (CAE) and 0–0.7 (ensemble).
- All results are on synthetic data. Nothing has been verified in a real exam room.

## Repository layout

```
rf-exam-anomaly-poc/
├── data/background/   background-noise IQ files (download from the release)
├── doc/               write-up (00_overview.md, .html), figures, prior-work survey
├── models/            trained models: 3 CNNs, 3 CAEs, normal statistics, thresholds, ensemble scaling
├── results/           reference results reported in the write-up
├── src/               training pipeline, inference pipeline, evaluation
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

The three background-noise IQ files (480 MB) are attached to the release because of GitHub's file size limit. From the repository root:

```
curl -LO https://github.com/sinkei9456/rf-exam-anomaly-poc/releases/download/v1.0/background_park02.tar
tar xf background_park02.tar        # data/background/Park02_1.npy, Park02_2.npy, Park02_3.npy
cd data/background && sha256sum -c SHA256SUMS.txt && cd ../..
```

Each file is 1 s at 20 MS/s (complex64); files 1, 2, 3 are the background for the training, validation and test sets. They come from a HackRF One recording (center 2,449.5 MHz) made in a park with the drone and controller switched off, published under CC BY 4.0 in the [rf-drone-detector](https://github.com/sinkei9456/rf-drone-detector) release file hackrf_1_train_background.tar. Bluetooth-shaped blobs were removed and the 3 s were split into 1 s files (src/neo2_pieces.py in that repository).

## Usage 1: training pipeline

```
bash src/run_all.sh                         # everything, about 30 min on one GPU
bash src/run_all.sh --use-released-models   # skip training, score with models/ (about 8 min)
```

| Step | Script | What it does | Time |
|---|---|---|---:|
| 1 | src/make_data.py | synthesize 55,000 snapshots (data/, 1.3 GB) | ~3 min |
| 2 | src/pixeldiff.py | (A)(B): pick smoothing size and thresholds on the validation set, score | ~1 min |
| 3 | src/cnn.py | (C) CNN training, 3 seeds | ~20 min |
| 4 | src/cae.py | (D) CAE training, 3 seeds | ~5 min |
| 5 | src/ensemble.py | (E) ensemble: scale CNN/CAE scores on validation normals, take the larger; no training | seconds |
| 6 | src/evaluate.py | test-set evaluation, tables (runs/results.md), figures (doc/fig/) | ~3 min |
| 7 | src/compare_results.py | compare with the reference results (results/) | seconds |

Use `PY=/path/python` to choose the interpreter and `WORKERS` (default 16) for data generation. Existing data/*.npy files are reused. After retraining, `python src/export_models.py` copies the new models into models/.

## Usage 2: inference pipeline

src/predict.py runs with the files in models/ only. To get just the trained models and the inference code, download rf-exam-anomaly-inference-v1.0.zip (2.6 MB) from the release. Try it on a demo IQ file (1 min, earphone-type anomaly at SNR 6 dB in seconds 20–29):

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
| `--fpr` | 0.001 | which threshold to use: 0.001 = exceeded by 0.1 % of the validation normal snapshots, 0.01 = exceeded by 1 % |
| `--noise-ref FILE` | none | normal IQ data used to normalize the noise power (recommended: same site and receiver settings) |
| `--k`, `--n` | 2, 10 | alarm when k of the last n snapshots are anomalous |
| `--interval` | 1.0 | seconds between snapshots (time column only) |
| `--out` | runs/predict.csv | output CSV: index, time_s, score, threshold, anomaly, alarm |

For your own recordings, keep the receiver settings of the training data (HackRF, 2,449.5 MHz, 20 MS/s, fixed gain), give a normal reference with `--noise-ref`, and re-check thresholds on normal data from the new site: the CNN's threshold was set so that 0.1 % of validation normal snapshots exceed it, yet 0.01–0.65 % of test normal snapshots did when only the background changed.

## Reproducibility

Both paths were run from a fresh copy of this repository and compared with results/ (NVIDIA DGX Spark GB10, ARM CPU, Python 3.11, PyTorch 2.14, CUDA 13).

- With the released models (`--use-released-models`): all 52 metrics of all 11 models (including the 3 ensembles) were identical, as were the synthetic data and pixel-difference results.
- Full retraining (`bash src/run_all.sh`): pixel-difference results were identical; CNN, CAE and ensemble numbers changed slightly even on the same machine. Per seed, a single metric differed by up to 9.3 percentage points (CNN seed 0). The epoch the CNN picked on the validation set also changed for seeds 0 and 2.
- The 3-seed means used in the write-up moved much less (table below). After retraining, ensemble recall was still within 1.2 points of the CNN. The conclusions do not change.

Full retraining vs. reference results (3-seed means; threshold exceeded by 0.1 % of validation normal snapshots)

| Model | Largest snapshot recall difference (points) | Largest event recall difference (points) | Test FPR, reference (%) | Test FPR, retrained (%) | Alarms per hour on the normal stream, reference | Alarms per hour on the normal stream, retrained |
|---|---:|---:|---:|---:|---:|---:|
| (C) CNN | 2.7 | 0.0 | 0.30 | 0.27 | 0.5 | 0.4 |
| (D) CAE | 0.5 | 1.7 | 0.05 | 0.05 | 0.0 | 0.0 |
| (E) CNN + CAE ensemble | 2.2 | 0.0 | 0.29 | 0.18 | 0.4 | 0.2 |

Recall differences are taken per anomaly type and SNR, and the largest is shown. Alarms use the 2-of-10 rule (an alarm when at least 2 of the last 10 snapshots are anomalous).

PyTorch does not guarantee bit-identical GPU training across runs or hardware, so CNN/CAE numbers can differ slightly; use src/compare_results.py to check.

## Limitations

- Normal Bluetooth traffic was synthesized as short packets only (≤ 0.4 ms). Allowed devices with long packets (e.g. medical BLE) or devices in neighboring rooms may change the results.
- The two anomaly patterns imitate the Bluetooth specification; they are not recordings of real devices.
- The background noise is 3 s from one site.
- Seat localization and body shadowing are not covered.

## License and sources

- Code: MIT (LICENSE)
- Background IQ data, documents and figures: CC BY 4.0
- Original background recording: rf-drone-detector, https://doi.org/10.5281/zenodo.23656790
