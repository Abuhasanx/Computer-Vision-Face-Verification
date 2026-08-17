import os
import json
import random
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model import FaceEmbeddingModel

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_DIR         = r"D:\biztech\preprocessed_images"
CHECKPOINT_DIR   = r"D:\biztech\checkpoints"
RESULTS_DIR      = r"D:\biztech\results"

BACKBONE         = "resnet50"
EMBEDDING_DIM    = 512
INPUT_SIZE       = 224

# Split ratios (by identity — test identities never seen during training)
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
# TEST_RATIO     = 0.15  (remainder, used later in gallery_probe.py / roc_analysis.py)

# Training hyperparameters
BATCH_SIZE       = 32
NUM_EPOCHS       = 30
LR_HEAD          = 1e-3       # learning rate for embedding + classifier heads
LR_BACKBONE      = 1e-4       # learning rate for backbone (fine-tune after warmup)
WARMUP_EPOCHS    = 5          # backbone frozen for first N epochs (head warmup)
WEIGHT_DECAY     = 1e-4
TRIPLET_MARGIN   = 0.3        # margin for triplet loss
LAMBDA_TRIPLET   = 0.5        # weight of triplet loss (CE weight = 1.0)

# Augmentation
USE_AUGMENTATION = True

# Reproducibility
SEED             = 42

# DataLoader
NUM_WORKERS      = 4
PIN_MEMORY       = True       # set False if you hit memory errors

# Save checkpoint every N epochs (best val loss is always saved separately)
SAVE_EVERY_N_EPOCHS = 5
# ─────────────────────────────────────────────────────────────


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    Loads pre-cropped face images from:
        DATA_DIR/
            person_001/img_01.jpg ...
            person_002/img_01.jpg ...

    Each identity gets a consecutive integer label (0 ... N-1).
    """

    def __init__(self, identity_folders: list, transform=None):
        """
        Args:
            identity_folders: list of Path objects, one per identity
            transform: torchvision transform pipeline
        """
        self.transform = transform
        self.samples   = []   # list of (image_path, class_idx)
        self.class_to_idx = {}
        self.idx_to_class = {}

        valid_exts = (".jpg", ".jpeg", ".png")

        for class_idx, person_dir in enumerate(sorted(identity_folders)):
            self.class_to_idx[person_dir.name] = class_idx
            self.idx_to_class[class_idx]        = person_dir.name

            images = [
                f for f in person_dir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_exts
            ]
            for img_path in sorted(images):
                self.samples.append((img_path, class_idx))

        self.num_classes = len(identity_folders)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transforms(input_size, use_augmentation):
    """
    Training transform: augmentation + normalize.
    Validation transform: just resize + normalize (no augmentation).
    """
    imagenet_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if use_augmentation:
        train_tf = transforms.Compose([
            transforms.Resize((input_size + 20, input_size + 20)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            imagenet_norm,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            imagenet_norm,
        ])

    val_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        imagenet_norm,
    ])

    return train_tf, val_tf


def split_identities(data_dir: Path, train_ratio: float, val_ratio: float, seed: int):
    """
    Splits identity folders into train / val / test at the IDENTITY level.
    Test identities are completely disjoint from training — critical for
    demonstrating generalization in gallery_probe.py.

    Returns:
        train_dirs, val_dirs, test_dirs — lists of Path objects
    """
    person_dirs = sorted([
        p for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("person_")
    ])

    if not person_dirs:
        raise FileNotFoundError(f"No person_XXX folders found in {data_dir}")

    rng = random.Random(seed)
    rng.shuffle(person_dirs)

    n       = len(person_dirs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_dirs = person_dirs[:n_train]
    val_dirs   = person_dirs[n_train:n_train + n_val]
    test_dirs  = person_dirs[n_train + n_val:]

    return train_dirs, val_dirs, test_dirs


# ── Online Triplet Loss ───────────────────────────────────────────────────────

class OnlineTripletLoss(nn.Module):
    """
    Batch-hard online triplet mining.

    For each anchor in the batch:
        - Hardest positive: same class, MAXIMUM distance to anchor
        - Hardest negative: different class, MINIMUM distance to anchor

    This is more stable and efficient than offline triplet generation.

    Reference: "In Defense of the Triplet Loss" (Hermans et al., 2017)
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            embeddings: (B, D) L2-normalized embeddings
            labels    : (B,) integer class labels

        Returns:
            scalar triplet loss, number of valid (non-zero) triplets
        """
        # Pairwise cosine distance matrix (1 - cosine_sim for L2-normalized vecs)
        # Since embeddings are L2-normalized: cosine_sim = dot product
        dot_product = torch.mm(embeddings, embeddings.t())   # (B, B)
        # Clamp for numerical safety before sqrt isn't needed here; use angular dist
        dist_matrix = 1.0 - dot_product                      # (B, B), range [0, 2]

        B = labels.size(0)
        labels_row = labels.unsqueeze(1).expand(B, B)        # (B, B)
        labels_col = labels.unsqueeze(0).expand(B, B)        # (B, B)

        pos_mask = (labels_row == labels_col)                 # same identity
        neg_mask = ~pos_mask                                  # different identity

        # Remove self-pairs from positive mask
        eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        pos_mask = pos_mask & ~eye

        # Batch-hard: hardest positive per anchor (max dist, same class)
        pos_dist = dist_matrix * pos_mask.float()
        hardest_pos_dist = pos_dist.max(dim=1)[0]            # (B,)

        # Batch-hard: hardest negative per anchor (min dist, diff class)
        # Fill same-class positions with large value so they don't win the min
        neg_dist = dist_matrix + (~neg_mask).float() * 1e9
        hardest_neg_dist = neg_dist.min(dim=1)[0]            # (B,)

        # Triplet loss
        losses = torch.relu(hardest_pos_dist - hardest_neg_dist + self.margin)

        # Only count anchors that have at least one valid positive
        valid_mask = pos_mask.any(dim=1)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device), 0

        loss = losses[valid_mask].mean()
        n_valid = valid_mask.sum().item()
        return loss, n_valid


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, ce_criterion, triplet_criterion,
                    optimizer, device, lambda_triplet, epoch):
    model.train()

    total_loss     = 0.0
    total_ce       = 0.0
    total_triplet  = 0.0
    correct        = 0
    total_samples  = 0

    pbar = tqdm(loader, desc=f"  Train E{epoch}", leave=False, unit="batch")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        embeddings, logits = model(images, return_logits=True)

        # Cross-Entropy loss (identity classification)
        ce_loss = ce_criterion(logits, labels)

        # Triplet loss (embedding space metric learning)
        trip_loss, n_valid = triplet_criterion(embeddings, labels)

        loss = ce_loss + lambda_triplet * trip_loss
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        batch_size     = images.size(0)
        total_loss    += loss.item()    * batch_size
        total_ce      += ce_loss.item() * batch_size
        total_triplet += trip_loss.item() * batch_size if isinstance(trip_loss, torch.Tensor) else 0
        total_samples += batch_size

        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ce":   f"{ce_loss.item():.4f}",
            "trip": f"{trip_loss.item() if isinstance(trip_loss, torch.Tensor) else 0:.4f}",
            "acc":  f"{correct/total_samples:.3f}",
        })

    return {
        "loss":     total_loss    / total_samples,
        "ce_loss":  total_ce      / total_samples,
        "trip_loss":total_triplet / total_samples,
        "acc":      correct       / total_samples,
    }


@torch.no_grad()
def validate(model, loader, ce_criterion, triplet_criterion, device, lambda_triplet, epoch):
    model.eval()

    total_loss    = 0.0
    total_ce      = 0.0
    total_triplet = 0.0
    correct       = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"  Val   E{epoch}", leave=False, unit="batch")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings, logits = model(images, return_logits=True)

        ce_loss   = ce_criterion(logits, labels)
        trip_loss, _ = triplet_criterion(embeddings, labels)
        loss      = ce_loss + lambda_triplet * trip_loss

        batch_size     = images.size(0)
        total_loss    += loss.item()     * batch_size
        total_ce      += ce_loss.item()  * batch_size
        total_triplet += trip_loss.item() * batch_size if isinstance(trip_loss, torch.Tensor) else 0
        total_samples += batch_size

        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

    return {
        "loss":      total_loss    / total_samples,
        "ce_loss":   total_ce      / total_samples,
        "trip_loss": total_triplet / total_samples,
        "acc":       correct       / total_samples,
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path, split_info):
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "scheduler_state":scheduler.state_dict() if scheduler else None,
        "metrics":        metrics,
        "backbone":       model.backbone_name,
        "embedding_dim":  model.embedding_dim,
        "num_classes":    model.num_classes,
        "split_info":     split_info,    # save which identities are train/val/test
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt.get("metrics", {})


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR,    exist_ok=True)

    # ── Split identities ──────────────────────────────────────────────
    print(f"\nScanning dataset: {DATA_DIR}")
    train_dirs, val_dirs, test_dirs = split_identities(
        Path(DATA_DIR), TRAIN_RATIO, VAL_RATIO, SEED
    )
    print(f"  Train identities : {len(train_dirs)}")
    print(f"  Val   identities : {len(val_dirs)}")
    print(f"  Test  identities : {len(test_dirs)}  (held out for gallery/probe + ROC)")

    # Save split so other scripts (gallery_probe, roc) use the same partitioning
    split_info = {
        "train": [d.name for d in train_dirs],
        "val":   [d.name for d in val_dirs],
        "test":  [d.name for d in test_dirs],
    }
    split_path = os.path.join(RESULTS_DIR, "identity_split.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Split saved to   : {split_path}")

    # ── Datasets & DataLoaders ────────────────────────────────────────
    train_tf, val_tf = build_transforms(INPUT_SIZE, USE_AUGMENTATION)

    train_dataset = FaceDataset(train_dirs, transform=train_tf)
    val_dataset   = FaceDataset(val_dirs,   transform=val_tf)

    num_classes = train_dataset.num_classes
    print(f"\n  Train samples : {len(train_dataset)} from {num_classes} identities")
    print(f"  Val   samples : {len(val_dataset)}")

    # Weighted sampler so each identity is sampled equally per epoch
    # (prevents bias toward identities with more images)
    labels_list = [s[1] for s in train_dataset.samples]
    class_counts = defaultdict(int)
    for lbl in labels_list:
        class_counts[lbl] += 1
    weights = [1.0 / class_counts[lbl] for lbl in labels_list]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,     # needed for triplet mining — avoid single-sample batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # ── Model ─────────────────────────────────────────────────────────
    print(f"\nBuilding {BACKBONE} model...")
    model = FaceEmbeddingModel(
        backbone_name=BACKBONE,
        embedding_dim=EMBEDDING_DIM,
        pretrained=True,
        num_classes=num_classes,
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")

    # ── Freeze backbone for warmup epochs ─────────────────────────────
    def freeze_backbone(m):
        for param in m.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(m):
        for param in m.backbone.parameters():
            param.requires_grad = True

    freeze_backbone(model)
    print(f"  Backbone FROZEN for first {WARMUP_EPOCHS} warmup epochs.")

    # ── Optimizer ─────────────────────────────────────────────────────
    # Separate param groups so backbone can have a lower LR when unfrozen
    head_params     = list(model.embedding_layer.parameters()) + \
                      list(model.classifier.parameters())
    backbone_params = list(model.backbone.parameters())

    optimizer = optim.AdamW([
        {"params": head_params,     "lr": LR_HEAD},
        {"params": backbone_params, "lr": LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine annealing LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # ── Loss functions ────────────────────────────────────────────────
    ce_criterion      = nn.CrossEntropyLoss(label_smoothing=0.1)
    triplet_criterion = OnlineTripletLoss(margin=TRIPLET_MARGIN)

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch  = 1
    best_val_loss = float("inf")
    history      = []

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        start_epoch, prev_metrics = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1
        best_val_loss = prev_metrics.get("val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch - 1}, best val loss: {best_val_loss:.4f}")

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"  CE + {LAMBDA_TRIPLET} × Triplet  |  margin={TRIPLET_MARGIN}  |  batch={BATCH_SIZE}")
    print("=" * 65)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Unfreeze backbone after warmup
        if epoch == WARMUP_EPOCHS + 1:
            unfreeze_backbone(model)
            print(f"\n  [Epoch {epoch}] Backbone UNFROZEN — fine-tuning with lr={LR_BACKBONE}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, ce_criterion, triplet_criterion,
            optimizer, device, LAMBDA_TRIPLET, epoch
        )

        # Validate
        val_metrics = validate(
            model, val_loader, ce_criterion, triplet_criterion,
            device, LAMBDA_TRIPLET, epoch
        )

        scheduler.step()

        elapsed = time.time() - epoch_start
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"E{epoch:03d}/{NUM_EPOCHS}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_acc={train_metrics['acc']:.3f}  |  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['acc']:.3f}  |  "
            f"lr={lr_now:.2e}  {elapsed:.0f}s"
        )

        # Save history
        row = {
            "epoch":      epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
            "lr":         lr_now,
        }
        history.append(row)

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, scheduler, epoch,
                            {"val_loss": best_val_loss, **val_metrics},
                            best_path, split_info)
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f}) -> {best_path}")

        # Periodic checkpoint
        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:03d}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch,
                            {"val_loss": val_metrics["loss"]},
                            ckpt_path, split_info)

    # ── Save training history ─────────────────────────────────────────
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print(f"  Best val loss  : {best_val_loss:.4f}")
    print(f"  Best model     : {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")
    print(f"  History saved  : {history_path}")
    print(f"  Split saved    : {split_path}")
    print("\nNext step: run generate_pairs.py using checkpoints/best_model.pth")


if __name__ == "__main__":
    main()