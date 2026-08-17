import os
import csv
import time
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from mtcnn import MTCNN

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SOURCE_DIR          = r"D:\biztech\DATA"
OUTPUT_DIR          = r"D:\biztech\preprocessed_images"
INPUT_SIZE          = 224          # final saved image size (pixels), matches ResNet input
MARGIN              = 0.25         # fraction of face box added as padding on each side
CONF_THRESHOLD      = 0.90         # MTCNN minimum detection confidence
VALID_EXTS          = (".jpg", ".jpeg", ".png")
SAVE_FORMAT         = "JPEG"       # PIL save format for output crops
SAVE_QUALITY        = 95           # JPEG quality (ignored for PNG)

# Normalization stats (ImageNet) — applied before saving so images are
# ready to load directly into the model without re-normalizing.
# NOTE: if you prefer to normalize inside your DataLoader (more standard),
# set APPLY_NORMALIZATION = False and do it in transforms instead.
APPLY_NORMALIZATION = False        # see note above — recommended: False
IMAGENET_MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD        = np.array([0.229, 0.224, 0.225],  dtype=np.float32)
# ─────────────────────────────────────────────────────────────


def align_face(image_rgb: np.ndarray, left_eye: tuple, right_eye: tuple) -> np.ndarray:
    """
    Rotates the full image so the eye-to-eye line is horizontal.

    Args:
        image_rgb : (H, W, 3) numpy array, RGB
        left_eye  : (x, y) from MTCNN keypoints["left_eye"]
        right_eye : (x, y) from MTCNN keypoints["right_eye"]

    Returns:
        Rotated image, same shape as input.
    """
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = (
        (left_eye[0] + right_eye[0]) / 2.0,
        (left_eye[1] + right_eye[1]) / 2.0,
    )

    h, w = image_rgb.shape[:2]
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    rotated = cv2.warpAffine(image_rgb, rot_mat, (w, h), flags=cv2.INTER_CUBIC)
    return rotated


def detect_align_crop(
    image_rgb: np.ndarray,
    detector: MTCNN,
    conf_threshold: float,
    margin: float,
) -> np.ndarray | None:
    """
    Full preprocessing chain for one image:
        detect -> pick largest face -> align -> re-detect -> crop with margin

    Returns:
        Cropped face as (H', W', 3) numpy RGB array, or None if no face found.
    """
    # ── Step 1: Detect on original image ──────────────────────────────
    results = detector.detect_faces(image_rgb)
    results = [f for f in results if f["confidence"] >= conf_threshold]

    if not results:
        return None

    # Pick largest face (most likely the subject)
    face = max(results, key=lambda f: f["box"][2] * f["box"][3])

    kps       = face["keypoints"]
    left_eye  = kps["left_eye"]
    right_eye = kps["right_eye"]

    # ── Step 2: Align full image by eye angle ─────────────────────────
    aligned = align_face(image_rgb, left_eye, right_eye)

    # ── Step 3: Re-detect on aligned image for accurate bbox ──────────
    results_aligned = detector.detect_faces(aligned)
    results_aligned = [f for f in results_aligned if f["confidence"] >= conf_threshold]

    if results_aligned:
        face_box = max(results_aligned, key=lambda f: f["box"][2] * f["box"][3])["box"]
    else:
        face_box = face["box"]   # fallback to pre-alignment bbox

    x, y, w, h = face_box
    x, y = max(0, x), max(0, y)

    # ── Step 4: Add margin and crop ───────────────────────────────────
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(aligned.shape[1], x + w + mx)
    y2 = min(aligned.shape[0], y + h + my)

    crop = aligned[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def resize_and_optionally_normalize(
    crop_rgb: np.ndarray,
    size: int,
    apply_norm: bool,
) -> np.ndarray:
    """
    Resizes the crop to (size x size) and optionally applies ImageNet
    channel normalization.

    If apply_norm=False (recommended), returns a standard uint8 image
    that PIL can save normally — normalization happens inside your DataLoader.

    If apply_norm=True, normalizes to float32 and clips back to uint8 for
    saving (lossy — not ideal for training, but useful for visualization).
    """
    # ── Resize ────────────────────────────────────────────────────────
    resized = cv2.resize(crop_rgb, (size, size), interpolation=cv2.INTER_CUBIC)

    if not apply_norm:
        return resized  # uint8 (H, W, 3), values 0-255

    # ── Normalize (ImageNet stats) ────────────────────────────────────
    # Convert to float [0, 1], subtract mean, divide by std
    img_float = resized.astype(np.float32) / 255.0
    img_float = (img_float - IMAGENET_MEAN) / IMAGENET_STD

    # Clip and rescale back to uint8 for saving
    # Note: this is lossy — prefer apply_norm=False + DataLoader normalization
    img_float = np.clip(img_float * 0.5 + 0.5, 0, 1)  # rough re-scale for saving
    return (img_float * 255).astype(np.uint8)


def process_identity(
    person_folder: Path,
    output_person_dir: Path,
    detector: MTCNN,
    conf_threshold: float,
    margin: float,
    size: int,
    apply_norm: bool,
    valid_exts: tuple,
    save_format: str,
    save_quality: int,
) -> dict:
    """
    Processes all images in one person folder.

    Returns a stats dict:
        {
            "person_id": str,
            "total": int,
            "saved": int,
            "skipped": int,
            "skip_reasons": [str, ...]
        }
    """
    output_person_dir.mkdir(parents=True, exist_ok=True)

    images = [f for f in person_folder.iterdir()
              if f.is_file() and f.suffix.lower() in valid_exts]
    images.sort()

    saved   = 0
    skipped = 0
    skip_reasons = []

    for img_path in images:
        try:
            image_rgb = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            skipped += 1
            skip_reasons.append(f"{img_path.name}: load error — {e}")
            continue

        crop = detect_align_crop(image_rgb, detector, conf_threshold, margin)

        if crop is None:
            skipped += 1
            skip_reasons.append(f"{img_path.name}: no face detected (conf >= {conf_threshold})")
            continue

        processed = resize_and_optionally_normalize(crop, size, apply_norm)

        out_path = output_person_dir / img_path.name
        try:
            pil_img = Image.fromarray(processed)
            if save_format == "JPEG":
                pil_img.save(out_path, format=save_format, quality=save_quality)
            else:
                pil_img.save(out_path, format=save_format)
            saved += 1
        except Exception as e:
            skipped += 1
            skip_reasons.append(f"{img_path.name}: save error — {e}")

    return {
        "person_id":    person_folder.name,
        "total":        len(images),
        "saved":        saved,
        "skipped":      skipped,
        "skip_reasons": skip_reasons,
    }


def write_summary_csv(stats_list: list, output_dir: Path):
    csv_path = output_dir / "preprocessing_summary.csv"
    fieldnames = ["person_id", "total_images", "saved", "skipped", "skip_reasons"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in stats_list:
            writer.writerow({
                "person_id":    s["person_id"],
                "total_images": s["total"],
                "saved":        s["saved"],
                "skipped":      s["skipped"],
                "skip_reasons": " | ".join(s["skip_reasons"]) if s["skip_reasons"] else "",
            })
    print(f"\nSummary CSV written to: {csv_path}")


def main():
    source_dir = Path(SOURCE_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all person_XXX subfolders
    person_folders = sorted([
        p for p in source_dir.iterdir()
        if p.is_dir() and p.name.startswith("person_")
    ])

    if not person_folders:
        print(f"ERROR: No 'person_XXX' subfolders found in: {source_dir}")
        return

    print(f"Found {len(person_folders)} identity folders in: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Settings: size={INPUT_SIZE}px | margin={MARGIN} | conf>={CONF_THRESHOLD} | norm={APPLY_NORMALIZATION}")
    print("-" * 60)

    print("Initializing MTCNN detector...")
    detector = MTCNN()   # uses CPU by default; fast enough for offline batch
    print("MTCNN ready.\n")

    stats_list      = []
    total_saved     = 0
    total_skipped   = 0
    failed_identities = []
    start_time      = time.time()

    for person_folder in tqdm(person_folders, desc="Processing identities", unit="id"):
        output_person_dir = output_dir / person_folder.name

        stats = process_identity(
            person_folder    = person_folder,
            output_person_dir = output_person_dir,
            detector         = detector,
            conf_threshold   = CONF_THRESHOLD,
            margin           = MARGIN,
            size             = INPUT_SIZE,
            apply_norm       = APPLY_NORMALIZATION,
            valid_exts       = VALID_EXTS,
            save_format      = SAVE_FORMAT,
            save_quality     = SAVE_QUALITY,
        )

        stats_list.append(stats)
        total_saved   += stats["saved"]
        total_skipped += stats["skipped"]

        # Flag identities that lost images (so you can inspect them)
        if stats["saved"] < 4:
            failed_identities.append(
                f"  {stats['person_id']}: only {stats['saved']} faces saved "
                f"(had {stats['total']} images)"
            )

    elapsed = time.time() - start_time

    # ── Final report ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Identities processed : {len(person_folders)}")
    print(f"Total images saved   : {total_saved}")
    print(f"Total images skipped : {total_skipped}  (no face detected / load errors)")
    print(f"Time elapsed         : {elapsed:.1f}s  ({elapsed/len(person_folders):.2f}s/identity avg)")
    print(f"Output location      : {output_dir}")

    if failed_identities:
        print(f"\n⚠  {len(failed_identities)} identities have < 4 saved faces "
              f"(below assessment minimum — consider removing them):")
        for msg in failed_identities:
            print(msg)
    else:
        print("\n✓ All identities retained >= 4 face crops.")

    if total_saved >= 500:
        print("✓ Meets 500+ image target.")
    else:
        print(f"⚠  Only {total_saved} images — below 500 target. Note in README limitations.")

    write_summary_csv(stats_list, output_dir)

    print("\nNext step: run split_dataset.py to create train/val/test splits.")


if __name__ == "__main__":
    main()