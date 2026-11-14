# BandSplitFuse

**BandSplitFuse: A Lightweight Frequency-Domain Plug-and-Play Module for Infrared and Visible Image Fusion**

[中文说明](README_zh-CN.md) · [Reproducibility](REPRODUCIBILITY.md) · [Runtime validation](VALIDATION.md) · [GitHub upload guide](UPLOAD_TO_GITHUB.md)

BandSplitFuse is a lightweight feature-fusion module for infrared–visible image fusion. It separates radial frequency bands, applies amplitude gating to low frequencies and phase-consistency gating to high frequencies, and uses a zero-initialized residual branch so that training starts from the host network's additive fusion behavior.

This repository contains **model, training, testing, metric-evaluation, ablation, latency, and downstream-detection code**, plus small aggregated result tables for the three-seed paired validation. It intentionally contains no plotting or visualization scripts.

> **Status:** research code accompanying a manuscript under review. Datasets, pretrained checkpoints, generated images, and raw logs are not included. Do not interpret the repository as a claim of state-of-the-art performance.

## Features

- Radial soft masks for low-/high-frequency decomposition.
- Amplitude and phase-consistency gates for cross-modal feature fusion.
- Zero-initialized residual fusion for stable plug-and-play initialization.
- CDDFuse integration with full two-stage training and frozen-backbone plugin training.
- DenseFuse integration for cross-backbone evaluation.
- EN, SD, SF, MI, SCD, VIF, Qabf, and SSIM evaluation.
- Controlled ablation, latency measurement, and fixed-detector YOLOv8n evaluation.
- Three-seed paired fine-tuning for the baseline, original module, and continuous-routing alternative.
- Batch size 2 in BandSplitFuse training scripts for 8 GB GPU memory configurations.

## Repository layout

```text
BandSplitFuse/
├── FourierFuse/                  # Core Fourier and band-split modules
├── CDDFuse/                      # CDDFuse host, training, testing and evaluation
├── densefuse_pytorch-master/     # DenseFuse host and plugin evaluation
├── downstream_detection/         # Fixed YOLOv8n COCO evaluation
├── MSRS-main/                    # Data placement instructions only
├── tests/                        # Lightweight CPU unit tests
├── requirements.txt
├── environment.yml
└── REPRODUCIBILITY.md
```

## Installation

Python 3.10 is recommended. Install a PyTorch build matching your CUDA version before installing the remaining dependencies. See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

```bash
conda create -n bandsplitfuse python=3.10 -y
conda activate bandsplitfuse
# Install torch and torchvision for your CUDA/CPU setup first.
pip install -r requirements.txt
```

Run the lightweight unit tests:

```bash
python -m unittest discover -s tests -v
```

## Data and checkpoint placement

Download the MSRS dataset from the [official MSRS repository](https://github.com/Linfeng-Tang/MSRS). TNO, RoadScene, and the labeled detection subset must be obtained from their respective original sources and used according to their licenses.

Expected training and test layout:

```text
CDDFuse/
├── MSRS_train/
│   ├── ir/
│   └── vi/
├── test_img/
│   ├── MSRS/{ir,vi}/
│   ├── TNO/{ir,vi}/
│   └── RoadScene/{ir,vi}/
├── data/                         # Generated HDF5 training data
└── models/
    ├── CDDFuse_IVF.pth           # Official CDDFuse baseline
    └── YOUR_BANDSPLITFUSE.pth    # Your trained checkpoint
```

Additional weights required for optional experiments:

```text
densefuse_pytorch-master/train_result/model_weight.pkl
downstream_detection/yolov8n.pt
```

The BandSplitFuse checkpoint must be trained by the user or downloaded from a release link added by the repository owner. No model-download URL is fabricated in this package.

## Quick start

All commands below start from the repository root unless a different directory is shown.

### 1. Preprocess MSRS

```bash
cd CDDFuse
python dataprocessing.py
```

This creates `data/MSRS_train_imgsize_128_stride_200.h5`.

### 2. Train BandSplitFuse with CDDFuse

Full two-stage training, with 40 reconstruction epochs followed by fusion training:

```bash
cd CDDFuse
python train_full_plugin.py 120 40
```

Frozen-backbone, plugin-only training:

```bash
python train_plugin.py 20 1e-4
```

### 3. Evaluate fusion

```bash
cd CDDFuse
python test_IVF.py
python test_IVF_plugin.py models/YOUR_BANDSPLITFUSE.pth MSRS,TNO,RoadScene
```

The output includes fused images and the eight metrics listed above. The original baseline script uses batch size 8; reduce it if your GPU memory is insufficient.

### 4. Controlled ablation

```bash
cd CDDFuse
python run_ablation.py
```

The runner compares `amp_only`, `phase_only`, and full modules at two cutoff values with the same frozen backbone, loss, epoch count, and random seed. Results are written to `corrected_ablation_results.txt`.

### 5. Three-seed paired fine-tuning

```bash
cd CDDFuse
python run_continuous_multiseed.py
python summarize_three_seed.py
```

The runner compares the no-plug-in baseline, original BandSplitFuse, and continuous-routing alternative under the same 20-epoch fine-tuning protocol. It resumes matching final checkpoints instead of overwriting them. Aggregated per-seed and mean ± sample-standard-deviation tables are provided under `results/`; raw logs and checkpoints are excluded. The continuous alternative did not show a consistent additional advantage and is not the selected final model.

### 6. Latency and parameter count

```bash
cd CDDFuse
python measure_latency.py models/YOUR_BANDSPLITFUSE.pth --baseline models/CDDFuse_IVF.pth
```

Optional arguments: `--iterations 30 --device cuda`.

### 7. DenseFuse cross-backbone evaluation

Place the pretrained DenseFuse autoencoder at `densefuse_pytorch-master/train_result/model_weight.pkl`, then run:

```bash
cd densefuse_pytorch-master
python train_densefuse_plugin.py 20 1e-4
python test_densefuse.py models/YOUR_DENSEFUSE_PLUGIN.pth MSRS,TNO,RoadScene
```

### 8. Downstream zero-shot detection

Prepare `MSRS-main/detection/{ir,vi,labels}` and the official `MSRS-main/test/vi` partition as described in `MSRS-main/README.md`, then generate fusion inputs and evaluate all methods with the same fixed detector:

```bash
python CDDFuse/run_downstream_fusion.py \
  --checkpoint CDDFuse/models/YOUR_BANDSPLITFUSE.pth \
  --expected-count 80

python downstream_detection/evaluate_detection.py \
  --weights downstream_detection/yolov8n.pt \
  --device 0

python downstream_detection/evaluate_heldout_detection.py \
  --expected-count 23
```

The first two commands reproduce the saved predictions and COCO ground truth for the original 80-image labeled subset. The final command filters those saved predictions to the 23 images explicitly present in the released MSRS test partition and reruns COCOeval without new inference. For the manuscript protocol, 48 training-partition images and nine images absent from the released train/test partitions are excluded; the retained subset contains 39 person and 12 car instances. The detector is not fine-tuned, and this small held-out study should not be generalized to the full MSRS detection benchmark without additional experiments.

## Reproducibility notes

- The infrared feature is always the first argument: `BandSplitFuse.forward(f_ir, f_vi)`.
- Full BandSplitFuse training uses batch size 2, unlike the original CDDFuse baseline training script.
- Full training uses `kornia.losses.SSIMLoss`. The baseline training script tries the original `kornia.losses.SSIM` first and falls back to `SSIMLoss` when unavailable; the selected implementation is printed and must be retained in protocol descriptions.
- Controlled ablation results must be regenerated with `run_ablation.py`; older experimental checkpoints and logs are intentionally excluded.
- GPU latency depends on hardware, PyTorch, CUDA, input size, warmup, and synchronization settings.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for checkpoint formats, evaluation details, and current limitations.

## Authors

Zhengwei Wang, Wenzhi Xia, Tao Liu, and Xiuhua Zeng  
School of Physics and Electronic Engineering, Engineering Research Center of Intelligent Systems and Advanced Materials, Qujing Normal University, Qujing 655011, China

Corresponding author: Zhengwei Wang (`wangzw@mail.qjnu.edu.cn`)

## Acknowledgements

This project builds on CDDFuse and DenseFuse. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and licensing items that must be checked before public release.

## Citation

The accompanying manuscript is under review. A formal citation and DOI should be added only after publication. Software metadata is provided in `CITATION.cff`.
