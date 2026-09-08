# Pneumonia Detection from Chest X-Rays (CNN + Transfer Learning)

A binary image classifier that flags pneumonia in pediatric chest X-rays, built for a
data-science interview presentation on a CNN / computer-vision problem.

## Problem

Radiologists screening chest X-rays for pneumonia is slow and subject to inter-reader
variability. A CNN triage model can't replace a radiologist, but it can prioritize
worklists (flag likely-positive scans for faster review) and act as a second reader.
The clinical cost of a **false negative** (missed pneumonia) is much higher than a
**false positive** (extra review), which shapes almost every modeling decision below.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
— Kermany et al., 5,863 pediatric chest X-rays, 2 classes (`NORMAL`, `PNEUMONIA`),
pre-split into `train/val/test`. The training split is imbalanced (~74% pneumonia,
~26% normal), which is handled explicitly in `src/train.py` rather than ignored.

Download it and arrange it as:

```
data/chest_xray/
├── train/{NORMAL,PNEUMONIA}/*.jpeg
├── val/{NORMAL,PNEUMONIA}/*.jpeg
└── test/{NORMAL,PNEUMONIA}/*.jpeg
```

```bash
# via Kaggle CLI (requires ~/.kaggle/kaggle.json credentials)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data --unzip
```

## Approach

- **Transfer learning**: ResNet-18 pretrained on ImageNet, fine-tuned end-to-end. With
  only ~5k training images, training a CNN from scratch overfits fast; low-level
  ImageNet features (edges, textures) transfer well to X-rays.
- **Class imbalance**: inverse-frequency class weights in the loss, so a false negative
  on the minority (`NORMAL`) class is penalized more, and recall on `PNEUMONIA` is
  tracked as the primary metric, not accuracy.
- **Augmentation is domain-aware**: mild rotation/translation/brightness jitter to
  mimic patient positioning and exposure variance — deliberately **no vertical flips**
  and no aggressive crops, since those produce anatomically implausible X-rays.
- **Interpretability**: Grad-CAM overlays showing which regions of the X-ray drove each
  prediction — important for any model a clinician is asked to trust.

## Project layout

```
src/
├── dataset.py    # ImageFolder wrapper + transforms
├── model.py      # ResNet-18 backbone, swapped classification head
├── train.py      # training loop, class-weighted loss, checkpointing, early stopping
├── evaluate.py   # accuracy/precision/recall/F1/ROC-AUC, confusion matrix, ROC plot
├── gradcam.py    # Grad-CAM heatmap generation for sample predictions
└── utils.py      # seeding, class-weight computation, plotting helpers
scripts/
└── make_dummy_data.py  # generates synthetic images so the pipeline can be smoke-tested
                         # without downloading the real (1GB+) dataset
```

## Running it

```bash
pip install -r requirements.txt

# 1. sanity-check the whole pipeline in ~30s with synthetic data (no real dataset needed)
python scripts/make_dummy_data.py --out data/dummy_xray
python src/train.py --data-dir data/dummy_xray --epochs 1 --batch-size 8

# 2. real training run (after downloading the Kaggle dataset into data/chest_xray)
python src/train.py --data-dir data/chest_xray --epochs 15 --batch-size 32

# 3. evaluate the best checkpoint on the held-out test split
python src/evaluate.py --data-dir data/chest_xray --checkpoint outputs/best_model.pt

# 4. Grad-CAM on a handful of test images
python src/gradcam.py --data-dir data/chest_xray --checkpoint outputs/best_model.pt --n 6
```

## Results

*(Fill in after running on the real dataset — see `outputs/metrics.json` and
`outputs/confusion_matrix.png` once `evaluate.py` has been run. Typical published
baselines on this dataset land around 92-96% test accuracy / 0.96+ recall on
`PNEUMONIA` with a fine-tuned ResNet-18.)*


## Limitations & future work

- Single-institution pediatric dataset — no external validation, so generalization to
  adult patients or other scanners is unverified.
- Binary label only; doesn't distinguish bacterial vs. viral pneumonia, which changes
  treatment.
- No segmentation/localization ground truth to quantitatively check whether Grad-CAM
  attention matches radiologist-marked regions.
- Next steps: multi-site external test set, bacterial/viral sub-typing, calibrated
  probability outputs (temperature scaling) so a clinical threshold can be chosen
  deliberately rather than defaulting to 0.5.
