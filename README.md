# Wi-Fi Signal Decoding

Deep-learning experiments for decoding the 24-bit IEEE 802.11 L-SIG header directly from 5 MHz complex I/Q waveforms.

The repository compares frequency-domain residual CNNs, channel attention, a time-domain ablation, and masked-autoencoder experiments. It was developed as a UNC Data Science Practicum project.

## Why this matters

Commodity low-bandwidth receivers do not observe the full Wi-Fi channel. Recovering control information from their partial waveform measurements is therefore both a signal-processing problem and a structured multi-label prediction problem.

Each packet is normalized and represented either as:

- a `2 × 12 × 64` real/imaginary per-symbol FFT tensor; or
- a `2 × 768` real/imaginary time-domain tensor for the no-FFT ablation.

The models predict the 24 L-SIG bits for Non-HT packets. Evaluation uses per-bit accuracy and exact 24-bit match.

## Technical approach

| Experiment | Input | Model |
|---|---|---|
| DeepMon baseline | Per-symbol FFT | 2D residual CNN |
| Attention | Per-symbol FFT | Residual CNN with channel attention |
| No-FFT ablation | Raw I/Q | 1D residual CNN |
| Depth and augmentation ablations | Per-symbol FFT | Controlled baseline variants |
| MAE / CrossMAE | Raw and cross-modal I/Q | Masked-autoencoder variants |

## Recorded results

The historical CrossMAE evaluation logs report the following held-out test results on 7,359 Non-HT packets:

| Model | Bit accuracy | Exact match | Transmission-time MAE | Length MAE |
|---|---:|---:|---:|---:|
| DeepMon baseline | 0.9768 | 0.6570 | 0.0770 ms | — |
| CrossMAE, full fine-tune | 0.9880 | 0.9337 | 0.0114 ms | 20.9 B |

These numbers are preserved in [`results/historical/logs/crossmae_eval.log`](results/historical/logs/crossmae_eval.log). They are historical experiment records, not a fully reproducible benchmark: the CrossMAE model, dataset adapter, L-SIG decoding utility, and checkpoints used for that run are no longer available in this repository. The maintained runnable paths below cover the residual-CNN experiments only.

## Run the maintained experiments

The packet dataset is not included. Supply a root with this structure:

```text
DATA_ROOT/
├── Train/<experiment>/Low/Packet_*.bin
├── Train/<experiment>/Label/Packet_*.mat
├── Test/<experiment>/Low/Packet_*.bin
└── Test/<experiment>/Label/Packet_*.mat
```

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a maintained experiment:

```bash
python train_baseline.py --data-root /path/to/DATA_ROOT
python train_attention.py --data-root /path/to/DATA_ROOT
python train_nofft.py --data-root /path/to/DATA_ROOT
```

All training scripts also accept `--output-dir`, `--epochs`, `--batch-size`, `--workers`, and `--seed`. `WIFI_DATA_ROOT` and `WIFI_OUTPUT_DIR` provide environment-variable equivalents for the paths.

## Repository map

```text
dataset.py / dataset_nofft.py     Signal discovery and preprocessing
model.py                          2D residual CNN
model_attention.py                Channel-attention variant
model_1d.py                       Time-domain ablation
train_*.py                        Maintained training entry points
run_*_ablation.py                 Controlled experiment sweeps
archived_experiments/             Incomplete MAE-era source retained for context
results/historical/               Original logs and figures
docs/presentation_guide.pdf       Project presentation notes
tests/                             Data-shape and model smoke tests
```

## Reproducibility boundary

- Raw packet data and trained checkpoints are not distributed.
- CrossMAE artifacts are retained rather than silently presented as runnable code.
- The recorded dataset split was 61,104 train, 10,784 validation, and 7,359 test packets.
- No new metrics have been inferred or recomputed during this repository cleanup.
