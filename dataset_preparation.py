import os
import shutil
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────
SOURCE_DIR      = r"D:\biztech\lfw-deepfunneled"   # original LFW folder (5749 subfolders)
OUTPUT_DIR      = r"D:\biztech\DATA"        # where the filtered dataset will be written
MIN_IMAGES      = 4                                 # minimum images per identity to keep
MAX_IDENTITIES  = None                              # e.g. 200 to cap it, or None for "keep all that qualify"
VALID_EXTS      = (".jpg", ".jpeg", ".png")
COPY_MODE       = "copy"                            # "copy" or "move" (copy is safer — keeps original LFW intact)
# ─────────────────────────────────────────


def find_qualifying_identities(source_dir, min_images):
    """
    Scans source_dir for subfolders (one per identity) that contain
    at least `min_images` valid image files.

    Returns a list of tuples: (identity_name, [list_of_image_paths])
    """
    source_dir = Path(source_dir)
    qualifying = []

    all_subfolders = [p for p in source_dir.iterdir() if p.is_dir()]
    print(f"Scanning {len(all_subfolders)} identity folders in source...")

    for person_folder in all_subfolders:
        images = [
            f for f in person_folder.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTS
        ]
        if len(images) >= min_images:
            qualifying.append((person_folder.name, sorted(images)))

    # Sort by number of images descending, so you get your richest identities first
    qualifying.sort(key=lambda x: len(x[1]), reverse=True)
    return qualifying


def build_dataset(qualifying, output_dir, max_identities, copy_mode):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_identities is not None:
        qualifying = qualifying[:max_identities]

    total_images = 0
    manifest_rows = []

    for idx, (identity_name, images) in enumerate(qualifying, start=1):
        person_id = f"person_{idx:03d}"
        person_out_dir = output_dir / person_id
        person_out_dir.mkdir(parents=True, exist_ok=True)

        for img_idx, img_path in enumerate(images, start=1):
            ext = img_path.suffix.lower()
            out_name = f"img_{img_idx:02d}{ext}"
            out_path = person_out_dir / out_name

            if copy_mode == "copy":
                shutil.copy2(img_path, out_path)
            elif copy_mode == "move":
                shutil.move(str(img_path), str(out_path))
            else:
                raise ValueError("COPY_MODE must be 'copy' or 'move'")

            manifest_rows.append({
                "person_id": person_id,
                "original_identity_name": identity_name,
                "image_index": img_idx,
                "output_path": str(out_path),
            })
            total_images += 1

    return qualifying, total_images, manifest_rows


def write_manifest(manifest_rows, output_dir):
    """Writes a CSV mapping person_id -> original LFW identity name.
    Useful for your README documentation and for sanity-checking splits later."""
    import csv
    manifest_path = Path(output_dir) / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["person_id", "original_identity_name", "image_index", "output_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Manifest written to: {manifest_path}")


def main():
    qualifying = find_qualifying_identities(SOURCE_DIR, MIN_IMAGES)
    print(f"Identities with >= {MIN_IMAGES} images: {len(qualifying)}")

    if len(qualifying) < 50:
        print(f"WARNING: only {len(qualifying)} identities qualify — assessment wants 50+.")
        print("Consider lowering MIN_IMAGES, though the assessment requires 4+ images/identity, so that's a hard floor.")

    selected, total_images, manifest_rows = build_dataset(
        qualifying, OUTPUT_DIR, MAX_IDENTITIES, COPY_MODE
    )

    print("\n──────── SUMMARY ────────")
    print(f"Identities selected : {len(selected)}")
    print(f"Total images copied : {total_images}")
    print(f"Avg images/identity : {total_images / len(selected):.2f}")
    print(f"Output location     : {OUTPUT_DIR}")

    if len(selected) < 50:
        print("⚠ Below the 50-identity minimum required by the assessment.")
    else:
        print("✓ Meets the 50+ identity requirement.")

    if total_images < 500:
        print("⚠ Below the 500-image 'preferably' target — still valid but note it in README limitations.")
    else:
        print("✓ Meets the 500+ image target.")

    write_manifest(manifest_rows, OUTPUT_DIR)


if __name__ == "__main__":
    main()