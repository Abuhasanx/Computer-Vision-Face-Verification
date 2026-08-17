"""
compare_faces.py

Loads your trained checkpoint (checkpoints/best_model.pth) and compares
two face images: "Are these the same person?"

Pipeline:
    Image 1 -> MTCNN detect -> align -> crop -> trained ResNet -> Embedding 1
    Image 2 -> MTCNN detect -> align -> crop -> trained ResNet -> Embedding 2
                                    |
                            Cosine Similarity
                                    |
                        Match / Non-Match (threshold)

Requires model.py (FaceEmbeddingModel) in the same folder.

Usage:
    Edit IMAGE_1 / IMAGE_2 / MODEL_PATH below, then:
    python compare_faces.py
"""

import os
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import FaceEmbeddingModel

# ─────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────
IMAGE_1 = r"D:\biztech\preprocessed_images\person_003\img_02.jpg"
IMAGE_2 = r"D:\biztech\preprocessed_images\person_019\img_29.jpg"

MODEL_PATH = r"D:\biztech\checkpoints\best_model.pth"

THRESHOLD              = 0.50   # cosine-similarity match threshold — tune later via Task 7 (ROC/EER)
DETECT_CONF_THRESHOLD  = 0.90   # MTCNN face-detection confidence filter
INPUT_SIZE             = 224    # must match training input size
MARGIN                 = 0.20   # crop margin, same convention as preprocess_dataset.py
# ─────────────────────────────────────────


_preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_trained_model(checkpoint_path, device):
    """
    Loads FaceEmbeddingModel from a training checkpoint saved by train.py.
    """
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
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  backbone      : {ckpt['backbone']}")
    print(f"  embedding_dim : {ckpt['embedding_dim']}")
    print(f"  trained epoch : {ckpt.get('epoch', '?')}")
    if missing:
        print(f"  (info) missing keys ignored: {missing}")
    if unexpected:
        print(f"  (info) unexpected keys ignored: {unexpected}")

    return model


def align_face(image_rgb, left_eye, right_eye):
    """Rotates the image so the eye-to-eye line is horizontal."""
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = (
        (left_eye[0] + right_eye[0]) / 2.0,
        (left_eye[1] + right_eye[1]) / 2.0,
    )

    h, w = image_rgb.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    rotated = cv2.warpAffine(image_rgb, rot_matrix, (w, h), flags=cv2.INTER_CUBIC)
    return rotated


def detect_align_crop(image_path, detector, conf_threshold, margin=0.2):
    """
    Full preprocessing chain for one raw image.
    Returns (PIL.Image of cropped face, detection_info dict).
    detection_info contains bbox and keypoints on the ORIGINAL image for visualization.
    """
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image path not found: '{image_path}' "
                                 f"— set IMAGE_1 / IMAGE_2 at the top of the script.")

    image_rgb = np.array(Image.open(image_path).convert("RGB"))

    results = detector.detect_faces(image_rgb)
    results = [f for f in results if f["confidence"] >= conf_threshold]
    if not results:
        raise ValueError(f"No face detected above confidence {conf_threshold} in: {image_path}")

    face = max(results, key=lambda f: f["box"][2] * f["box"][3])
    kps = face["keypoints"]
    left_eye, right_eye = kps["left_eye"], kps["right_eye"]

    # Store original detection info for visualization
    detection_info = {
        "box": face["box"],
        "keypoints": kps,
        "confidence": face["confidence"],
    }

    aligned = align_face(image_rgb, left_eye, right_eye)

    results_aligned = detector.detect_faces(aligned)
    results_aligned = [f for f in results_aligned if f["confidence"] >= conf_threshold]
    if results_aligned:
        x, y, w, h = max(results_aligned, key=lambda f: f["box"][2] * f["box"][3])["box"]
    else:
        x, y, w, h = face["box"]

    x, y = max(0, x), max(0, y)
    mx, my = int(w * margin), int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(aligned.shape[1], x + w + mx)
    y2 = min(aligned.shape[0], y + h + my)

    crop = aligned[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty crop produced for: {image_path}")

    return Image.fromarray(crop), detection_info


def get_embedding(model, pil_image, device):
    """Preprocesses a cropped face image and returns its L2-normalized embedding."""
    tensor = _preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
    return embedding


def cosine_similarity(embedding_a, embedding_b):
    return F.cosine_similarity(embedding_a, embedding_b, dim=1).item()


# ─────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────

# Design tokens
MATCH_COLOR    = (80, 200, 120)    # soft green   — BGR
NO_MATCH_COLOR = (70, 100, 240)    # soft red-ish — BGR
ACCENT_COLOR   = (200, 200, 200)   # light grey for secondary text
BG_COLOR       = (18, 18, 18)      # near-black background
PANEL_COLOR    = (28, 28, 28)      # slightly lighter panel
DIVIDER_COLOR  = (45, 45, 45)      # subtle divider
TEXT_COLOR     = (230, 230, 230)   # primary text
DIM_COLOR      = (110, 110, 110)   # muted text

CARD_W  = 520   # width of each image card
CARD_H  = 520   # height of each image card
PAD     = 28    # outer padding
GAP     = 40    # gap between cards
BAR_H   = 160   # results bar height at bottom
LABEL_H = 44    # label strip under each image

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO  = cv2.FONT_HERSHEY_DUPLEX


def _put_text(img, text, pos, font=FONT, scale=0.55, color=TEXT_COLOR, thickness=1, anchor="tl"):
    """Draw text with optional right/center anchor."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if anchor == "tr":
        x -= tw
    elif anchor == "tc":
        x -= tw // 2
    cv2.putText(img, text, (x, y + th), font, scale, color, thickness, cv2.LINE_AA)
    return tw, th


def draw_face_card(raw_image_path, detection_info, label, panel_color, accent):
    """
    Renders one face card: image with a thin bbox + keypoints, label strip at bottom.
    Returns BGR numpy array of shape (CARD_H, CARD_W, 3).
    """
    card = np.full((CARD_H, CARD_W, 3), panel_color, dtype=np.uint8)

    # ── load & fit image inside the card (leave room for label strip) ──────
    img_area_h = CARD_H - LABEL_H
    raw = cv2.imread(raw_image_path)
    if raw is None:
        raw = np.zeros((img_area_h, CARD_W, 3), dtype=np.uint8)

    ih, iw = raw.shape[:2]
    scale_f = min(CARD_W / iw, img_area_h / ih)
    new_w = int(iw * scale_f)
    new_h = int(ih * scale_f)
    resized = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # centre in card
    ox = (CARD_W - new_w) // 2
    oy = (img_area_h - new_h) // 2
    card[oy:oy + new_h, ox:ox + new_w] = resized

    # ── scale bbox & keypoints to resized coordinate system ────────────────
    x, y, w, h = detection_info["box"]
    kps = detection_info["keypoints"]

    def sx(v): return int(v * scale_f + ox)
    def sy(v): return int(v * scale_f + oy)

    bx1, by1 = sx(max(0, x)), sy(max(0, y))
    bx2, by2 = sx(x + w),     sy(y + h)

    # thin bbox with corner ticks
    BOX_T = 1
    TICK  = 10
    cv2.rectangle(card, (bx1, by1), (bx2, by2), accent, BOX_T, cv2.LINE_AA)

    # corner ticks (L-shaped)
    corners = [(bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2)]
    dx_tick  = [1, -1, 1, -1]
    dy_tick  = [1,  1, -1, -1]
    for (cx, cy), ddx, ddy in zip(corners, dx_tick, dy_tick):
        cv2.line(card, (cx, cy), (cx + ddx * TICK, cy), accent, 2, cv2.LINE_AA)
        cv2.line(card, (cx, cy), (cx, cy + ddy * TICK), accent, 2, cv2.LINE_AA)

    # keypoints — tiny filled circles
    kp_pts = [kps["left_eye"], kps["right_eye"], kps["nose"],
              kps["mouth_left"], kps["mouth_right"]]
    for pt in kp_pts:
        cv2.circle(card, (sx(pt[0]), sy(pt[1])), 3, accent, -1, cv2.LINE_AA)

    # confidence badge (top-right inside image)
    conf_txt = f"{detection_info['confidence']:.2f}"
    (ctw, cth), _ = cv2.getTextSize(conf_txt, FONT, 0.42, 1)
    badge_pad = 5
    badge_x1 = bx2 - ctw - badge_pad * 2
    badge_y1 = by1
    badge_x2 = bx2
    badge_y2 = by1 + cth + badge_pad * 2
    cv2.rectangle(card, (badge_x1, badge_y1), (badge_x2, badge_y2), accent, -1)
    cv2.putText(card, conf_txt,
                (badge_x1 + badge_pad, badge_y2 - badge_pad),
                FONT, 0.42, BG_COLOR, 1, cv2.LINE_AA)

    # ── label strip ────────────────────────────────────────────────────────
    strip_y = img_area_h
    cv2.rectangle(card, (0, strip_y), (CARD_W, CARD_H), panel_color, -1)
    cv2.line(card, (0, strip_y), (CARD_W, strip_y), DIVIDER_COLOR, 1)
    _put_text(card, label, (CARD_W // 2, strip_y + 10),
              font=FONT_MONO, scale=0.58, color=TEXT_COLOR, thickness=1, anchor="tc")

    # filename (truncated, dimmed)
    fname = os.path.basename(raw_image_path)
    if len(fname) > 32:
        fname = "…" + fname[-30:]
    _put_text(card, fname, (CARD_W // 2, strip_y + 26),
              scale=0.40, color=DIM_COLOR, anchor="tc")

    return card


def draw_score_bar(similarity, is_match, threshold):
    """
    Renders the bottom results bar.
    Returns BGR numpy array of shape (BAR_H, total_w, 3).
    """
    total_w = PAD * 2 + CARD_W * 2 + GAP
    bar = np.full((BAR_H, total_w, 3), BG_COLOR, dtype=np.uint8)

    accent = MATCH_COLOR if is_match else NO_MATCH_COLOR
    verdict = "MATCH" if is_match else "NO MATCH"

    # top divider
    cv2.line(bar, (0, 0), (total_w, 0), DIVIDER_COLOR, 1)

    # ── left: similarity score ──────────────────────────────────────────────
    col1_cx = PAD + CARD_W // 2

    _put_text(bar, "SIMILARITY", (col1_cx, 18),
              scale=0.38, color=DIM_COLOR, thickness=1, anchor="tc")

    score_txt = f"{similarity:.4f}"
    _put_text(bar, score_txt, (col1_cx, 32),
              font=FONT_MONO, scale=1.20, color=TEXT_COLOR, thickness=2, anchor="tc")

    _put_text(bar, f"threshold  {threshold:.2f}", (col1_cx, 82),
              scale=0.40, color=DIM_COLOR, anchor="tc")

    # ── progress bar (similarity meter) ────────────────────────────────────
    bar_x1 = PAD + 20
    bar_x2 = PAD + CARD_W - 20
    bar_y  = 108
    bar_th = 4
    # track
    cv2.line(bar, (bar_x1, bar_y), (bar_x2, bar_y), DIVIDER_COLOR, bar_th, cv2.LINE_AA)
    # fill
    fill_x = int(bar_x1 + (bar_x2 - bar_x1) * max(0.0, min(1.0, similarity)))
    cv2.line(bar, (bar_x1, bar_y), (fill_x, bar_y), accent, bar_th, cv2.LINE_AA)
    # threshold tick
    thr_x = int(bar_x1 + (bar_x2 - bar_x1) * threshold)
    cv2.line(bar, (thr_x, bar_y - 8), (thr_x, bar_y + 8), DIM_COLOR, 1, cv2.LINE_AA)
    _put_text(bar, "thr", (thr_x, bar_y + 10), scale=0.32, color=DIM_COLOR, anchor="tc")

    # ── vertical divider ────────────────────────────────────────────────────
    mid_x = total_w // 2
    cv2.line(bar, (mid_x, 14), (mid_x, BAR_H - 14), DIVIDER_COLOR, 1)

    # ── right: verdict ──────────────────────────────────────────────────────
    col2_cx = mid_x + (PAD + CARD_W // 2)

    _put_text(bar, "DECISION", (col2_cx, 18),
              scale=0.38, color=DIM_COLOR, thickness=1, anchor="tc")

    _put_text(bar, verdict, (col2_cx, 32),
              font=FONT_MONO, scale=1.10, color=accent, thickness=2, anchor="tc")

    icon = "✓" if is_match else "✗"
    # cv2 can't render unicode — draw a circle/cross shape instead
    icon_cx, icon_cy = col2_cx, 112
    cv2.circle(bar, (icon_cx, icon_cy), 18, accent, 1, cv2.LINE_AA)
    if is_match:
        pts = np.array([
            [icon_cx - 8, icon_cy],
            [icon_cx - 2, icon_cy + 7],
            [icon_cx + 9, icon_cy - 8],
        ], dtype=np.int32)
        cv2.polylines(bar, [pts], False, accent, 2, cv2.LINE_AA)
    else:
        cv2.line(bar, (icon_cx - 8, icon_cy - 8), (icon_cx + 8, icon_cy + 8), accent, 2, cv2.LINE_AA)
        cv2.line(bar, (icon_cx + 8, icon_cy - 8), (icon_cx - 8, icon_cy + 8), accent, 2, cv2.LINE_AA)

    return bar


def show_comparison_window(image1_path, image2_path,
                            det1, det2,
                            similarity, is_match, threshold):
    """
    Builds and shows the full comparison popup window.
    Press any key or close window to exit.
    """
    accent = MATCH_COLOR if is_match else NO_MATCH_COLOR

    # ── build two face cards ────────────────────────────────────────────────
    card1 = draw_face_card(image1_path, det1, "Image 1", PANEL_COLOR, accent)
    card2 = draw_face_card(image2_path, det2, "Image 2", PANEL_COLOR, accent)

    # ── assemble side-by-side ────────────────────────────────────────────────
    total_w = PAD * 2 + CARD_W * 2 + GAP
    top_h   = PAD * 2 + CARD_H
    canvas  = np.full((top_h, total_w, 3), BG_COLOR, dtype=np.uint8)

    x1 = PAD
    x2 = PAD + CARD_W + GAP
    canvas[PAD:PAD + CARD_H, x1:x1 + CARD_W] = card1
    canvas[PAD:PAD + CARD_H, x2:x2 + CARD_W] = card2

    # thin connector line between cards (horizontal centre)
    mid_y = PAD + CARD_H // 2
    lx1, lx2 = x1 + CARD_W, x2
    cv2.line(canvas, (lx1, mid_y), (lx2, mid_y), DIVIDER_COLOR, 1, cv2.LINE_AA)
    cv2.circle(canvas, (lx1 + GAP // 2, mid_y), 3, DIM_COLOR, -1, cv2.LINE_AA)

    # ── score bar ────────────────────────────────────────────────────────────
    score_bar = draw_score_bar(similarity, is_match, threshold)
    full = np.vstack([canvas, score_bar])

    # ── window ───────────────────────────────────────────────────────────────
    win_name = "Face Comparison"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, total_w, full.shape[0])
    cv2.imshow(win_name, full)

    print("\nVisual window open — press any key or close window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Loading MTCNN detector...")
    detector = MTCNN()

    print("Loading trained model...")
    model = load_trained_model(MODEL_PATH, device)

    print(f"\nProcessing Image 1: {IMAGE_1}")
    face_1, det_info_1 = detect_align_crop(IMAGE_1, detector, DETECT_CONF_THRESHOLD, MARGIN)
    embedding_1 = get_embedding(model, face_1, device)

    print(f"Processing Image 2: {IMAGE_2}")
    face_2, det_info_2 = detect_align_crop(IMAGE_2, detector, DETECT_CONF_THRESHOLD, MARGIN)
    embedding_2 = get_embedding(model, face_2, device)

    similarity = cosine_similarity(embedding_1, embedding_2)
    is_match   = similarity >= THRESHOLD

    print("\n" + "=" * 50)
    print(f"Cosine similarity : {similarity:.4f}")
    print(f"Threshold          : {THRESHOLD}")
    print(f"Decision           : {'MATCH (same person)' if is_match else 'NON-MATCH (different person)'}")
    print("=" * 50)

    show_comparison_window(
        IMAGE_1, IMAGE_2,
        det_info_1, det_info_2,
        similarity, is_match, THRESHOLD,
    )


if __name__ == "__main__":
    main()