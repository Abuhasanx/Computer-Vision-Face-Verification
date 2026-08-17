import os
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from model import FaceEmbeddingModel

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
PAIRS_CSV       = r"D:\biztech\results\test_pairs.csv"
MODEL_PATH      = r"D:\biztech\checkpoints\best_model.pth"
OUTPUT_CSV      = r"D:\biztech\results\pair_similarity_scores.csv"

INPUT_SIZE      = 224
BATCH_SIZE      = 64     # for computing unique-image embeddings
NUM_WORKERS     = 4
# ─────────────────────────────────────────────────────────────


_preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_trained_model(checkpoint_path, device):
    """Same loading logic as compare_faces.py — see that script for details."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = FaceEmbeddingModel(
        backbone_name=ckpt["backbone"],
        embedding_dim=ckpt["embedding_dim"],
        pretrained=False,
        num_classes=None,
    )

    state_dict = {
        k: v for k, v in ckpt["model_state"].items()
        if not k.startswith("classifier.")
    }
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  backbone      : {ckpt['backbone']}")
    print(f"  embedding_dim : {ckpt['embedding_dim']}")
    print(f"  trained epoch : {ckpt.get('epoch', '?')}")

    return model


def read_pairs(csv_path):
    """Reads test_pairs.csv into a list of dicts."""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pairs = list(reader)
    return pairs


class ImagePathDataset(torch.utils.data.Dataset):
    """Simple dataset over a list of unique image paths, for batched embedding."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, idx   # return idx so we can map back to path after batching


@torch.no_grad()
def compute_embeddings_for_unique_images(unique_paths, model, device, batch_size, num_workers):
    """
    Computes and caches one embedding per unique image path.

    Returns:
        dict: { path_str: numpy array of shape (embedding_dim,) }
    """
    dataset = ImagePathDataset(unique_paths, _preprocess)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    embedding_cache = {}

    for images, idxs in tqdm(loader, desc="Embedding unique images", unit="batch"):
        images = images.to(device, non_blocking=True)
        embeddings = model(images)   # (B, D), already L2-normalized
        embeddings = embeddings.cpu().numpy()

        for i, idx in enumerate(idxs.tolist()):
            path = unique_paths[idx]
            embedding_cache[path] = embeddings[i]

    return embedding_cache


def cosine_sim_from_vectors(vec_a, vec_b):
    """
    Cosine similarity between two already-L2-normalized numpy vectors.
    Since both are unit-norm, cosine similarity == dot product.
    """
    return float(np.dot(vec_a, vec_b))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print(f"Loading pairs from: {PAIRS_CSV}")
    pairs = read_pairs(PAIRS_CSV)
    print(f"  Total pairs: {len(pairs)}")

    # Collect the set of unique image paths across every pair
    unique_paths = sorted(set(
        [p["image_a"] for p in pairs] + [p["image_b"] for p in pairs]
    ))
    print(f"  Unique images referenced: {len(unique_paths)} "
          f"(vs {len(pairs) * 2} total if we didn't dedupe)")

    print("\nLoading trained model...")
    model = load_trained_model(MODEL_PATH, device)

    print("\nComputing embeddings for all unique images (cached, one pass each)...")
    start = time.time()
    embedding_cache = compute_embeddings_for_unique_images(
        unique_paths, model, device, BATCH_SIZE, NUM_WORKERS
    )
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({elapsed / len(unique_paths) * 1000:.1f} ms/image)")

    print("\nComputing cosine similarity for every pair...")
    results = []
    missing = 0
    for row in tqdm(pairs, desc="Scoring pairs", unit="pair"):
        img_a, img_b = row["image_a"], row["image_b"]

        if img_a not in embedding_cache or img_b not in embedding_cache:
            missing += 1
            continue  # shouldn't happen, but guard against a bad path

        sim = cosine_sim_from_vectors(embedding_cache[img_a], embedding_cache[img_b])

        results.append({
            "image_a": img_a,
            "image_b": img_b,
            "label": row["label"],
            "identity_a": row["identity_a"],
            "identity_b": row["identity_b"],
            "pair_type": row["pair_type"],
            "cosine_similarity": f"{sim:.6f}",
        })

    if missing:
        print(f"  (warning) {missing} pairs skipped — image path not found in embedding cache.")

    # ── Save results ──────────────────────────────────────────────────
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_a", "image_b", "label", "identity_a", "identity_b",
                  "pair_type", "cosine_similarity"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSimilarity scores written to: {output_path}")

    # ── Quick summary stats (sanity check before ROC analysis) ─────────
    genuine_scores  = [float(r["cosine_similarity"]) for r in results if r["label"] == "1"]
    impostor_scores = [float(r["cosine_similarity"]) for r in results if r["label"] == "0"]

    print("\n" + "=" * 55)
    print("SIMILARITY SCORE SUMMARY")
    print("=" * 55)
    print(f"  Genuine pairs (label=1)  : n={len(genuine_scores)}")
    print(f"    mean = {np.mean(genuine_scores):.4f}   std = {np.std(genuine_scores):.4f}")
    print(f"    min  = {np.min(genuine_scores):.4f}   max = {np.max(genuine_scores):.4f}")
    print(f"  Impostor pairs (label=0) : n={len(impostor_scores)}")
    print(f"    mean = {np.mean(impostor_scores):.4f}   std = {np.std(impostor_scores):.4f}")
    print(f"    min  = {np.min(impostor_scores):.4f}   max = {np.max(impostor_scores):.4f}")
    print(f"\n  Mean separation (genuine - impostor): "
          f"{np.mean(genuine_scores) - np.mean(impostor_scores):.4f}")
    print("  (larger gap = model discriminates identity better)")

    print("\nNext step: run roc_analysis.py on this CSV to compute ROC curve, "
          "AUC, EER, and select the final match/non-match threshold.")


if __name__ == "__main__":
    main()