import os
import csv
import json
import random
from pathlib import Path
from itertools import combinations
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_DIR          = r"D:\biztech\preprocessed_images"
SPLIT_JSON_PATH   = r"D:\biztech\results\identity_split.json"   # written by train.py
OUTPUT_CSV        = r"D:\biztech\results\test_pairs.csv"

VALID_EXTS = (".jpg", ".jpeg", ".png")

TARGET_POSITIVE            = 5000   # assessment's suggested target
MAX_POS_PAIRS_PER_IDENTITY = 40     # cap so identities with lots of images don't dominate
SEED                        = 42
# ─────────────────────────────────────────────────────────────


def load_test_identities(split_json_path, data_dir):
    """
    Loads the exact test-identity list saved by train.py, so the pairs
    generated here are guaranteed disjoint from anything the model was
    trained or validated on.

    Falls back with a loud warning if the split file is missing (you'd
    then be evaluating on possibly-seen identities — avoid this).
    """
    if not os.path.isfile(split_json_path):
        raise FileNotFoundError(
            f"Could not find {split_json_path}.\n"
            f"This file is written by train.py during training. Run train.py "
            f"first, or point SPLIT_JSON_PATH at your saved identity_split.json."
        )

    with open(split_json_path, "r") as f:
        split_info = json.load(f)

    test_names = split_info["test"]
    data_dir = Path(data_dir)

    test_dirs = []
    for name in test_names:
        p = data_dir / name
        if p.is_dir():
            test_dirs.append(p)
        else:
            print(f"  (warning) test identity '{name}' not found in {data_dir} — skipping.")

    return test_dirs


def collect_images(person_dir):
    """Returns sorted list of valid image paths for one identity folder."""
    return sorted([
        f for f in person_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    ])


def build_identity_image_map(test_dirs):
    """
    Returns dict: { person_name: [list of image Paths] }
    Only identities with >= 2 images are kept (need at least 2 images to
    form a positive pair).
    """
    id_to_images = {}
    dropped = []
    for person_dir in test_dirs:
        images = collect_images(person_dir)
        if len(images) >= 2:
            id_to_images[person_dir.name] = images
        else:
            dropped.append((person_dir.name, len(images)))

    if dropped:
        print(f"  (info) {len(dropped)} test identities skipped (fewer than 2 images):")
        for name, n in dropped[:10]:
            print(f"    {name}: {n} image(s)")
        if len(dropped) > 10:
            print(f"    ... and {len(dropped) - 10} more")

    return id_to_images


def generate_positive_pairs(id_to_images, target_count, max_per_identity, rng, seen):
    """
    Builds positive (same-identity) pairs.

    For each identity, forms all C(n, 2) unique image-pairs, shuffles
    them, and takes up to `max_per_identity` — this caps how much any
    single identity with many images can dominate the evaluation set.

    Stops early once `target_count` positive pairs have been collected
    (unless the dataset simply can't reach that many, in which case it
    returns everything available).
    """
    positive_pairs = []
    identities = list(id_to_images.keys())
    rng.shuffle(identities)

    for person_name in identities:
        images = id_to_images[person_name]
        all_combos = list(combinations(images, 2))   # every unique image pair, no (A,A), no mirrors
        rng.shuffle(all_combos)

        taken_for_this_identity = 0
        for img_a, img_b in all_combos:
            if taken_for_this_identity >= max_per_identity:
                break

            key = frozenset({str(img_a), str(img_b)})
            if key in seen:
                continue  # duplicate/mirrored pair — skip

            seen.add(key)
            positive_pairs.append({
                "image_a": str(img_a),
                "image_b": str(img_b),
                "label": 1,
                "identity_a": person_name,
                "identity_b": person_name,
                "pair_type": "positive",
            })
            taken_for_this_identity += 1

            if len(positive_pairs) >= target_count:
                return positive_pairs

    return positive_pairs


def generate_negative_pairs(id_to_images, target_count, rng, seen, max_attempts_multiplier=20):
    """
    Builds negative (different-identity) pairs by repeatedly sampling
    two DIFFERENT identities at random, then one random image from each.

    Uses the same `seen` frozenset-based dedup as positives, so a
    negative pair can never accidentally duplicate an already-used
    positive pair's image combination (impossible here since identities
    differ, but the shared `seen` set also prevents duplicate negatives
    with each other).

    Caps total attempts to avoid an infinite loop if the identity pool
    is too small to reach `target_count` unique negative pairs.
    """
    negative_pairs = []
    identities = list(id_to_images.keys())

    if len(identities) < 2:
        print("  (warning) fewer than 2 test identities available — cannot build negative pairs.")
        return negative_pairs

    max_attempts = target_count * max_attempts_multiplier
    attempts = 0

    while len(negative_pairs) < target_count and attempts < max_attempts:
        attempts += 1

        person_a, person_b = rng.sample(identities, 2)   # two DIFFERENT identities, no replacement
        img_a = rng.choice(id_to_images[person_a])
        img_b = rng.choice(id_to_images[person_b])

        key = frozenset({str(img_a), str(img_b)})
        if key in seen:
            continue  # duplicate/mirrored pair — skip and try again

        seen.add(key)
        negative_pairs.append({
            "image_a": str(img_a),
            "image_b": str(img_b),
            "label": 0,
            "identity_a": person_a,
            "identity_b": person_b,
            "pair_type": "negative",
        })

    if len(negative_pairs) < target_count:
        print(f"  (info) could only generate {len(negative_pairs)} unique negative pairs "
              f"after {attempts} attempts (target was {target_count}).")

    return negative_pairs


def write_pairs_csv(pairs, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image_a", "image_b", "label", "identity_a", "identity_b", "pair_type"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    print(f"\nPairs written to: {output_path}")


def main():
    rng = random.Random(SEED)

    print(f"Loading test-identity split from: {SPLIT_JSON_PATH}")
    test_dirs = load_test_identities(SPLIT_JSON_PATH, DATA_DIR)
    print(f"  Test identities found on disk: {len(test_dirs)}")

    print("\nBuilding identity -> images map (test set only)...")
    id_to_images = build_identity_image_map(test_dirs)
    usable_identities = len(id_to_images)
    total_images = sum(len(v) for v in id_to_images.values())
    print(f"  Usable test identities (>=2 images): {usable_identities}")
    print(f"  Total usable test images: {total_images}")

    if usable_identities < 2:
        raise RuntimeError("Need at least 2 usable test identities to generate negative pairs.")

    seen = set()   # shared across positive + negative generation -> global dedup

    print(f"\nGenerating positive pairs (target: {TARGET_POSITIVE}, "
          f"max {MAX_POS_PAIRS_PER_IDENTITY} per identity)...")
    positive_pairs = generate_positive_pairs(
        id_to_images, TARGET_POSITIVE, MAX_POS_PAIRS_PER_IDENTITY, rng, seen
    )
    print(f"  Positive pairs generated: {len(positive_pairs)}")

    # Balance: negatives target = however many positives we actually got
    negative_target = len(positive_pairs)
    print(f"\nGenerating negative pairs (target: {negative_target}, matched to positives)...")
    negative_pairs = generate_negative_pairs(id_to_images, negative_target, rng, seen)
    print(f"  Negative pairs generated: {len(negative_pairs)}")

    all_pairs = positive_pairs + negative_pairs
    rng.shuffle(all_pairs)

    print("\n" + "=" * 55)
    print("PAIR GENERATION SUMMARY")
    print("=" * 55)
    print(f"  Test identities used   : {usable_identities}")
    print(f"  Positive pairs         : {len(positive_pairs)}")
    print(f"  Negative pairs         : {len(negative_pairs)}")
    print(f"  Total pairs             : {len(all_pairs)}")
    print(f"  Balanced?               : {'yes' if len(positive_pairs) == len(negative_pairs) else 'no'}")
    if len(positive_pairs) < TARGET_POSITIVE:
        print(f"  Note: target was {TARGET_POSITIVE} positives; dataset size limited this to "
              f"{len(positive_pairs)}. This is expected with a ~92-identity held-out test set "
              f"and should be documented in the README as a known limitation.")

    write_pairs_csv(all_pairs, OUTPUT_CSV)

    print("\nNext step: run evaluate_pairs.py to compute cosine similarity for every pair "
          "and produce genuine/impostor score distributions for ROC analysis.")


if __name__ == "__main__":
    main()