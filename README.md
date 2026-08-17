[README.md](https://github.com/user-attachments/files/31151005/README.md)
# Face Verification / Re-Identification Pipeline
**Biz Tech Analytics — Technical Assessment**

<img width="1000" height="560" alt="crop_preview_margin0 25_conf0 9" src="https://github.com/user-attachments/assets/c3bad984-857e-4845-bdfd-f7c795aa8734" />
<img width="650" height="391" alt="Screenshot 2026-08-18 003734" src="https://github.com/user-attachments/assets/540fecfe-c567-4963-b672-68287d0679ea" />

---

## Table of Contents
1. [Dataset Description](#1-dataset-description)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Script Reference](#4-script-reference)
5. [End-to-End Usage](#5-end-to-end-usage)
6. [Model Architecture](#6-model-architecture)
7. [Training Methodology](#7-training-methodology)
8. [Positive / Negative Pair Generation Strategy](#8-positive--negative-pair-generation-strategy)
9. [Gallery / Probe Methodology](#9-gallery--probe-methodology)
10. [Final Cosine Threshold](#10-final-cosine-threshold)
11. [Results](#11-results)
12. [Known Limitations](#12-known-limitations)

---

## 1. Dataset Description

**Source:** [Labeled Faces in the Wild — deepfunneled variant (LFW)](http://vis-www.cs.umass.edu/lfw/)

**License / Usage Conditions:**
LFW is a public academic benchmark dataset collected and distributed for non-commercial research purposes. Images were sourced from news photographs and are freely available for academic use. No law-enforcement mugshots or criminal-record data are used.

**Dataset Statistics (after filtering):**

| Split | Identities | Images |
|-------|-----------|--------|
| Train | ~427 | ~2,800+ |
| Val | ~91 | ~600+ |
| Test | ~92 | ~600+ |
| **Total** | **~610** | **~4,000+** |

**Filtering criteria:** Only identities with **≥ 4 images** were retained (LFW has 5,749 identities, most with only 1 image). This produced 610 qualifying identities.

**Train / Val / Test Split:**
Splits are performed **at the identity level** — no identity appears in more than one split. The exact assignment is saved to `results/identity_split.json` during training and is reused by all downstream scripts. Test identities are entirely held out during training and validation to demonstrate generalization to unseen faces.

---

## 2. Project Structure

```
submission/
├── README.md                    ← this file
├── requirements.txt             ← Python dependencies
│
├── dataset_preparation.py       ← Task 1a: filters LFW into person_XXX/ layout
├── preprocess_dataset.py        ← Task 1b: MTCNN detect + align + crop + resize
├── model.py                     ← Task 2:  ResNet face embedding model definition
├── train.py                     ← Task 2:  training loop (CE + Triplet Loss)
├── generate_pairs.py            ← Task 3:  positive & negative pair generation
├── evaluate_pairs.py            ← Task 4:  cosine similarity scoring for all pairs
├── roc_analysis.py              ← Task 6+7: ROC curve, AUC, EER, threshold selection
├── gallery_probe.py             ← Task 5:  gallery build + probe identification
│
├── checkpoints/
│   └── best_model.pth           ← best checkpoint (saved by train.py)
│
└── results/
    ├── identity_split.json      ← train/val/test identity assignment
    ├── training_history.json    ← per-epoch loss and accuracy
    ├── test_pairs.csv           ← (image_a, image_b, label) pairs
    ├── pair_similarity_scores.csv ← pairs with cosine similarity scores
    ├── roc_metrics.json         ← AUC, EER, TAR@FAR, F1, threshold
    ├── roc_curve.png            ← ROC + distribution + F1 + FAR/FRR figure
    └── gallery_probe_results.csv ← (written by gallery_probe.py if extended)
```

---

## 3. Installation

**Python:** 3.9 or 3.10 recommended (tested on 3.10).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install mtcnn opencv-python pillow numpy tqdm scikit-learn pandas matplotlib
```

Or install everything from the requirements file:

```bash
pip install -r requirements.txt
```

**Requirements.txt contents:**
```
torch>=2.0.0
torchvision>=0.15.0
mtcnn>=0.1.1
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
tqdm>=4.65.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
```

> **GPU note:** Training is significantly faster on CUDA. The scripts auto-detect `cuda` vs `cpu` — no manual changes needed.

---

## 4. Script Reference

### `dataset_preparation.py`
**Purpose:** Filters the raw LFW-deepfunneled download (5,749 subfolders, most with 1 image) down to only the identities that meet the assessment's minimum image count, then renames and reorganises them into a clean `person_001/ … person_610/` layout.

**What it does step by step:**
1. Scans every subfolder inside `SOURCE_DIR` (the raw LFW download).
2. Counts valid image files (`.jpg`, `.jpeg`, `.png`) per identity.
3. Keeps only identities with `>= MIN_IMAGES` (default: 4).
4. Optionally caps the total identities with `MAX_IDENTITIES`.
5. Copies (or moves) images into `OUTPUT_DIR/person_NNN/img_NN.ext` using zero-padded indices.
6. Writes `manifest.csv` mapping the new `person_NNN` IDs back to the original LFW identity names.

**Config block (edit before running):**
```python
SOURCE_DIR      = r"D:\biztech\lfw-deepfunneled"
OUTPUT_DIR      = r"D:\biztech\DATA"
MIN_IMAGES      = 4
MAX_IDENTITIES  = None   # None = keep all qualifying
COPY_MODE       = "copy" # "copy" keeps LFW original intact
```

**Run:**
```bash
python dataset_preparation.py
```

**Output:** `D:\biztech\DATA\person_001\ … person_610\` + `manifest.csv`

---

### `preprocess_dataset.py`
**Purpose:** Runs every raw image through a full offline face preprocessing pipeline — MTCNN detection, eye-based geometric alignment, margin crop, resize to 224×224 — and saves the processed crops. These crops are what the model trains and evaluates on.

**What it does step by step:**
1. Loads each image as a NumPy RGB array.
2. Runs **MTCNN** face detection; discards any image where no face scores above `CONF_THRESHOLD` (0.90).
3. Picks the **largest detected face** (most likely the main subject).
4. **Aligns** the full image by rotating so the inter-eye line is horizontal (`cv2.getRotationMatrix2D` on the eye midpoint).
5. Re-detects on the aligned image for a more accurate bounding box.
6. Crops with a **margin** (`MARGIN = 0.25` → 25% of the face box added on each side) so the model sees some forehead, chin, and cheek context.
7. Resizes to `224 × 224` (`cv2.INTER_CUBIC`).
8. Saves to `OUTPUT_DIR` mirroring the source folder structure.
9. Normalization (ImageNet mean/std) is intentionally left to the DataLoader (`APPLY_NORMALIZATION = False`) — this is the standard PyTorch practice and avoids lossy uint8 round-tripping.

**Config block:**
```python
SOURCE_DIR          = r"D:\biztech\DATA"
OUTPUT_DIR          = r"D:\biztech\preprocessed_images"
INPUT_SIZE          = 224
MARGIN              = 0.25
CONF_THRESHOLD      = 0.90
APPLY_NORMALIZATION = False
```

**Run:**
```bash
python preprocess_dataset.py
```

**Output:** `D:\biztech\preprocessed_images\person_001\ … person_610\` (aligned, cropped, 224×224 faces) + `preprocessing_summary.csv`

---

### `model.py`
**Purpose:** Defines the `FaceEmbeddingModel` class — the neural network used throughout the pipeline. Can be run standalone for a quick sanity check.

**Architecture:**
```
Input: (B, 3, 224, 224)
    ↓
ResNet50 backbone (ImageNet-pretrained, fc layer replaced with Identity)
    ↓  outputs (B, 2048)
Linear embedding layer: 2048 → 512
    ↓  outputs (B, 512)
L2 Normalization (unit sphere)
    ↓  outputs (B, 512),  ‖embedding‖₂ = 1.0
[Optional] Linear classifier: 512 → num_classes  ← only during training
```

**Key design choices:**
- The original ResNet `fc` head is replaced with `nn.Identity()` to expose the 2048-D global average pooled features.
- An explicit `embedding_layer` projects those to 512-D.
- `nn.functional.normalize(..., p=2, dim=1)` is applied inside `forward()` so every output embedding is guaranteed to lie on the unit hypersphere. This makes cosine similarity equivalent to a simple dot product, which is numerically stable and fast.
- The optional `classifier` head (used during CE training) sits on top of the normalized embedding — it never sees the raw backbone features directly.

**Run standalone (sanity check):**
```bash
python model.py
```
Prints parameter counts, runs a dummy batch through both the embedding-only and embedding+logits forward paths, and (if you set `input_image`) tests with a real crop.

---

### `train.py`
**Purpose:** Fine-tunes the ResNet50 embedding model on the training identities using a combined Cross-Entropy + Online Triplet Loss strategy. Saves the best checkpoint and the identity split for use by all downstream scripts.

**What it does step by step:**
1. Splits all `person_NNN` folders into train / val / test **by identity** (70% / 15% / 15%) using a fixed seed — test identities are never touched during training.
2. Saves the split to `results/identity_split.json`.
3. Builds a `FaceDataset` (one sample = one cropped face image + integer identity label).
4. Uses a `WeightedRandomSampler` so each identity is seen equally often per epoch regardless of how many images it has.
5. **Freezes** the ResNet50 backbone for the first `WARMUP_EPOCHS` (5) epochs — only the embedding and classifier heads train. This prevents destroying ImageNet-pretrained features before the heads have learned anything useful.
6. After warmup, **unfreezes** the backbone and fine-tunes end-to-end with a lower backbone learning rate (`1e-4` vs `1e-3` for the heads).
7. Each batch computes:
   - **Cross-Entropy loss** on classification logits (identity prediction).
   - **Online batch-hard Triplet Loss** on the normalized embeddings.
   - Total loss = `CE + 0.5 × Triplet`.
8. Saves `checkpoints/best_model.pth` whenever validation loss improves.

**Training command:**
```bash
python train.py
```

**Resume from checkpoint:**
```bash
python train.py --resume checkpoints/best_model.pth
```

**Key hyperparameters (edit CONFIG block):**
```python
BATCH_SIZE       = 32
NUM_EPOCHS       = 30
LR_HEAD          = 1e-3
LR_BACKBONE      = 1e-4
WARMUP_EPOCHS    = 5
TRIPLET_MARGIN   = 0.3
LAMBDA_TRIPLET   = 0.5
```

**Output:** `checkpoints/best_model.pth`, `results/identity_split.json`, `results/training_history.json`

---

### `generate_pairs.py`
**Purpose:** Builds the balanced evaluation pair list (positive + negative) **exclusively from test-split identities** — the identities the model has never seen.

**What it does step by step:**
1. Reads `results/identity_split.json` to get the exact test identity names.
2. For each test identity, collects all valid image paths. Identities with fewer than 2 images are skipped (can't form a positive pair).
3. **Positive pairs:** For each identity, enumerates all `C(n, 2)` unique image combinations. Shuffles them, then takes up to `MAX_POS_PAIRS_PER_IDENTITY` (40) to prevent any single identity with many images from dominating. Stops once `TARGET_POSITIVE` (5,000) are collected.
4. **Negative pairs:** Randomly samples two different identities and one image from each, generating the same number of pairs as positives (balanced set). A shared `seen` frozenset rejects duplicates and mirrored pairs.
5. Shuffles the combined set and writes to `results/test_pairs.csv`.

**Duplicate / leakage prevention:**
- Every accepted pair is stored as `frozenset({path_a, path_b})`. Because `frozenset({A, B}) == frozenset({B, A})`, mirrored pairs are automatically deduplicated.
- The split JSON guarantees zero identity overlap with training data.

**Run:**
```bash
python generate_pairs.py
```

**Output:** `results/test_pairs.csv` — columns: `image_a, image_b, label, identity_a, identity_b, pair_type`

---

### `evaluate_pairs.py`
**Purpose:** Computes cosine similarity for every pair in `test_pairs.csv` and saves the scores. This is the bridge between pair generation (Task 3) and ROC analysis (Task 6).

**What it does step by step:**
1. Loads `test_pairs.csv`.
2. Extracts the **set of unique image paths** referenced across all pairs (many images appear in multiple pairs).
3. Runs all unique images through the model **once each** in batches, caching `path → 512-D embedding` in memory. This avoids redundant forward passes and is significantly faster than a naive per-pair loop.
4. For each pair, looks up both embeddings from the cache and computes `cosine_similarity = dot(emb_a, emb_b)` (valid because embeddings are L2-normalized).
5. Saves results to `results/pair_similarity_scores.csv`.
6. Prints a summary: mean/std/min/max similarity for genuine vs impostor pairs, and the mean separation gap.

**Run:**
```bash
python evaluate_pairs.py
```

**Output:** `results/pair_similarity_scores.csv` — columns: `image_a, image_b, label, identity_a, identity_b, pair_type, cosine_similarity`

---

### `roc_analysis.py`
**Purpose:** Consumes the scored pairs and produces all Task 6 + Task 7 metrics and a multi-panel publication-quality figure.

**What it does step by step:**
1. Reads `results/pair_similarity_scores.csv`.
2. Computes the full ROC curve (`sklearn.metrics.roc_curve`) from genuine/impostor similarity scores.
3. Calculates:
   - **AUC** (Area Under the Curve)
   - **EER** (Equal Error Rate — where FAR ≈ FRR)
   - **TAR @ FAR = 1%** and **TAR @ FAR = 0.1%**
   - **Best F1 threshold** (sweeps all thresholds, picks the one maximising F1)
   - **Precision, Recall, F1** at the best threshold
4. Saves metrics to `results/roc_metrics.json`.
5. Generates a dark-themed 5-panel figure (`results/roc_curve.png`) containing:
   - ROC curve with EER and TAR@FAR=1% annotated
   - Genuine vs impostor score distribution histogram
   - F1 score vs threshold curve
   - FAR & FRR vs threshold curve
   - Metrics summary table

**Run:**
```bash
python roc_analysis.py
```

**Output:** `results/roc_metrics.json`, `results/roc_curve.png`

---

### `gallery_probe.py`
**Purpose:** Implements Task 5 — builds a gallery of enrolled identities, then identifies a probe (query) image against the gallery using cosine similarity ranking. Displays a visual result window.

**What it does step by step:**

**Gallery build:**
1. Walks `GALLERY_DIR` — each subfolder is one enrolled identity (name = folder name).
2. For each identity, runs every image through MTCNN detect → align → crop → model embed.
3. If an identity has multiple images, all their embeddings are **mean-averaged and L2-renormalized** into a single representative gallery embedding. This makes the gallery more robust to lighting/pose variation than a single-image enrollment.

**Probe identification:**
1. Loads the probe image, runs MTCNN → align → crop → embed.
2. Computes cosine similarity between the probe embedding and every gallery identity embedding.
3. Ranks results highest-to-lowest.
4. Applies threshold: if top-1 similarity `≥ THRESHOLD` → identity accepted; else → `UNKNOWN`.

**Visualisation:**
- Left card: probe image with MTCNN bounding box and keypoints.
- Right card: best-match gallery representative image.
- Bottom panel: ranked similarity bars for top-K matches + verdict badge.

**Run:**
```bash
python gallery_probe.py
```

Edit the CONFIG block first:
```python
GALLERY_DIR   = r"D:\biztech\Gallery"   # folder with identity subfolders
PROBE_IMAGE   = r"path\to\probe.jpg"
MODEL_PATH    = r"D:\biztech\checkpoints\best_model.pth"
THRESHOLD     = 0.50
TOP_K         = 3
```

---

## 5. End-to-End Usage

Run the scripts in this exact order:

```bash
# Step 1 — filter and organise the raw LFW dataset
python dataset_preparation.py

# Step 2 — MTCNN detect, align, crop, resize all images
python preprocess_dataset.py

# Step 3 — train the ResNet50 embedding model
python train.py

# Step 4 — generate evaluation pairs from test identities only
python generate_pairs.py

# Step 5 — compute cosine similarity for every pair
python evaluate_pairs.py

# Step 6 — ROC curve, AUC, EER, threshold selection
python roc_analysis.py

# Step 7 — gallery/probe identification demo
python gallery_probe.py
```

---

## 6. Model Architecture

```
Input image (224 × 224 × 3)
        │
        ▼
┌─────────────────────────────┐
│  ResNet50 Backbone          │
│  (ImageNet pretrained)      │
│  conv1 → BN → ReLU         │
│  maxpool                    │
│  layer1 (3 × Bottleneck)   │
│  layer2 (4 × Bottleneck)   │
│  layer3 (6 × Bottleneck)   │
│  layer4 (3 × Bottleneck)   │
│  AdaptiveAvgPool2d → 2048D  │
└──────────────┬──────────────┘
               │  (2048-D features)
               ▼
┌─────────────────────────────┐
│  Embedding Layer            │
│  Linear(2048 → 512)         │
└──────────────┬──────────────┘
               │  (512-D embedding)
               ▼
┌─────────────────────────────┐
│  L2 Normalization           │
│  ‖embedding‖₂ = 1.0         │
└──────────────┬──────────────┘
               │  (512-D unit-norm embedding)
               ▼
  [Training only] Linear(512 → num_classes)
```

**Total parameters:** ~23.9M (ResNet50) + ~1.05M (embedding layer) ≈ **~25M**

---

## 7. Training Methodology

**Strategy: Cross-Entropy Identity Classification + Online Batch-Hard Triplet Loss**

**Why this combination?**

Cross-Entropy alone trains the model to distinguish training identities but does not explicitly shape the embedding space for the verification task (same/different person). The model can achieve high classification accuracy while producing embeddings that are poorly separated in cosine space for unseen identities.

Triplet Loss alone is unstable at the start of training — it requires embeddings that are already somewhat meaningful before hard triplet mining produces useful gradients.

The combination solves both problems: Cross-Entropy provides stable gradients and discriminative feature learning from the start; Triplet Loss simultaneously pulls same-identity embeddings together and pushes different-identity embeddings apart in the 512-D space, directly optimizing for the cosine similarity metric used in Tasks 4–7.

**Triplet mining:** Batch-hard online mining (Hermans et al., 2017) — for each anchor in the batch, the hardest positive (furthest same-identity sample) and hardest negative (closest different-identity sample) are selected. This is more efficient and stable than offline mining.

**Training schedule:**
- Epochs 1–5 (warmup): backbone frozen, only embedding + classifier heads trained at `lr=1e-3`.
- Epochs 6–30: backbone unfrozen and fine-tuned at `lr=1e-4`; heads continue at `lr=1e-3`.
- Optimizer: AdamW with weight decay `1e-4`.
- LR schedule: Cosine annealing (`eta_min=1e-6`).
- Label smoothing (`ε=0.1`) on Cross-Entropy for regularization.
- Gradient clipping (`max_norm=5.0`) for training stability.
- Weighted random sampler ensures each identity is sampled equally per epoch.

---

## 8. Positive / Negative Pair Generation Strategy

**Scope:** Test-split identities only (never seen during training or validation).

**Positive pairs (label = 1):**
- All `C(n, 2)` unique image combinations are enumerated per identity.
- Capped at 40 pairs per identity (`MAX_POS_PAIRS_PER_IDENTITY`) so identities with many images don't dominate.
- Target: 5,000 positive pairs total (or as many as the test set permits).

**Negative pairs (label = 0):**
- Two different identities are randomly sampled, then one image from each.
- Count is matched to the achieved positive count for a balanced evaluation set.

**Duplicate / leakage prevention:**
Every accepted pair is keyed by `frozenset({path_a, path_b})` in a shared `seen` set. Because `frozenset` is order-independent, `(A, B)` and `(B, A)` hash identically — mirrored duplicates are automatically rejected. The same `seen` set covers both positive and negative generation, so there is no cross-contamination.

---

## 9. Gallery / Probe Methodology

**Gallery construction:**
- One subfolder per enrolled identity inside `GALLERY_DIR`.
- Multiple images per identity are supported: each image is independently detected, aligned, cropped, and embedded; the embeddings are mean-averaged and L2-renormalized into a single gallery embedding.
- Mean-pooling across multiple enrollment images improves robustness to pose and lighting variation compared to single-image enrollment.

**Probe evaluation:**
- Probe image goes through the same MTCNN → align → crop → embed pipeline.
- Cosine similarity is computed against every gallery identity's embedding.
- Results are ranked highest-to-lowest.
- A configurable threshold decides match vs unknown: `sim ≥ THRESHOLD → identified`.

**Metrics reported:**
- Rank-1 accuracy (does the top-1 match equal the true identity?)
- Top-K ranked results displayed in the visualisation window.

---

## 10. Final Cosine Threshold

The threshold is selected from `roc_analysis.py` output. Two principled candidates are produced:

| Criterion | Threshold | Use case |
|-----------|-----------|----------|
| **EER** (FAR = FRR) | see `roc_metrics.json` → `eer_threshold` | Balanced security / convenience |
| **Best F1** | see `roc_metrics.json` → `best_f1_threshold` | Maximises F1 on the evaluation set |

The value used in `gallery_probe.py` (`THRESHOLD = 0.50`) is a reasonable starting point; replace it with `eer_threshold` from your `roc_metrics.json` for a principled, data-driven choice.

---

## 11. Results

After running the full pipeline, key metrics are saved to `results/roc_metrics.json`. Example output fields:

```json
{
  "auc": ...,
  "eer": ...,
  "eer_threshold": ...,
  "tar_at_far_1pct": ...,
  "tar_at_far_01pct": ...,
  "best_f1_threshold": ...,
  "best_f1": ...,
  "precision": ...,
  "recall": ...,
  "n_genuine": ...,
  "n_impostor": ...
}
```

> Fill in actual values after your training run completes.

**Rank-1 Accuracy:** Reported in the `gallery_probe.py` console output during evaluation.

---

## 12. Known Limitations

- **Test set size:** With ~92 held-out test identities, the maximum possible positive pairs is bounded by `C(images_per_identity, 2)` summed across identities. This may fall below the 5,000-pair target; `generate_pairs.py` reports the achieved count and matches negatives to it.
- **LFW image quality:** LFW images were collected from news photographs and vary significantly in resolution, lighting, and pose. Some images fail MTCNN detection at `conf ≥ 0.90` and are skipped — this is logged in `preprocessing_summary.csv`.
- **Single GPU assumption:** Training and evaluation scripts assume one GPU (or CPU). Multi-GPU is not implemented.
- **Gallery_probe.py threshold:** The `THRESHOLD = 0.50` in `gallery_probe.py` is a starting point. For production use, replace it with the EER threshold from `roc_metrics.json`.
- **No ArcFace / CosFace:** Training uses standard Cross-Entropy + Triplet Loss rather than ArcFace/CosFace margin losses. Angular margin losses typically improve open-set verification performance and would be the natural next upgrade.
- **Static gallery:** The gallery is rebuilt from scratch each run. A production system would persist gallery embeddings to disk.
