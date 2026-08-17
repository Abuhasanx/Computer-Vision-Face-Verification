import os
import json
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
GALLERY_DIR   = r"D:\biztech\Gallery"
PROBE_IMAGE   = r"D:\biztech\kaggle1\Tom Cruise\075_eed20fb4.jpg"
MODEL_PATH    = r"D:\biztech\checkpoints\best_model.pth"

THRESHOLD             = 0.50   # above → identity accepted, below → unknown
DETECT_CONF_THRESHOLD = 0.90
INPUT_SIZE            = 224
MARGIN                = 0.20
TOP_K                 = 3      # how many top matches to print in console
# ─────────────────────────────────────────


_preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────

def load_trained_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = FaceEmbeddingModel(
        backbone_name=ckpt["backbone"],
        embedding_dim=ckpt["embedding_dim"],
        pretrained=False,
        num_classes=None,
    )
    state_dict = {k: v for k, v in ckpt["model_state"].items()
                  if not k.startswith("classifier.")}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"[model] loaded  : {checkpoint_path}")
    print(f"        backbone: {ckpt['backbone']}  |  embed_dim: {ckpt['embedding_dim']}  |  epoch: {ckpt.get('epoch','?')}")
    return model


# ─────────────────────────────────────────────────────────
# FACE PIPELINE  (detect → align → crop → embed)
# ─────────────────────────────────────────────────────────

def align_face(image_rgb, left_eye, right_eye):
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) / 2.0,
                   (left_eye[1] + right_eye[1]) / 2.0)
    h, w = image_rgb.shape[:2]
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    return cv2.warpAffine(image_rgb, M, (w, h), flags=cv2.INTER_CUBIC)


def detect_align_crop(image_path, detector, conf_thr, margin):
    """
    Returns (PIL crop, detection_info) or raises on failure.
    detection_info = { box, keypoints, confidence } on the ORIGINAL image.
    """
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    results = [f for f in detector.detect_faces(image_rgb)
               if f["confidence"] >= conf_thr]
    if not results:
        raise ValueError(f"No face detected in: {image_path}")

    face = max(results, key=lambda f: f["box"][2] * f["box"][3])
    kps = face["keypoints"]
    det_info = {"box": face["box"], "keypoints": kps, "confidence": face["confidence"]}

    aligned = align_face(image_rgb, kps["left_eye"], kps["right_eye"])

    results2 = [f for f in detector.detect_faces(aligned)
                if f["confidence"] >= conf_thr]
    x, y, w, h = (max(results2, key=lambda f: f["box"][2] * f["box"][3])["box"]
                  if results2 else face["box"])

    x, y = max(0, x), max(0, y)
    mx, my = int(w * margin), int(h * margin)
    x1 = max(0, x - mx);  y1 = max(0, y - my)
    x2 = min(aligned.shape[1], x + w + mx)
    y2 = min(aligned.shape[0], y + h + my)

    crop = aligned[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty crop for: {image_path}")
    return Image.fromarray(crop), det_info


def embed(model, pil_img, device):
    t = _preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model(t)
    return e  # already L2-normalised inside FaceEmbeddingModel


# ─────────────────────────────────────────────────────────
# GALLERY BUILD
# ─────────────────────────────────────────────────────────

def build_gallery(gallery_dir, model, detector, device):
    """
    Walks gallery_dir, extracts embeddings per identity.
    Multiple images → averaged (mean) embedding, re-normalised.

    Returns:
        gallery : dict  { identity_name -> {"embedding": tensor, "image_paths": [...], "rep_image": str} }
    """
    if not os.path.isdir(gallery_dir):
        raise FileNotFoundError(f"Gallery folder not found: {gallery_dir}")

    gallery = {}
    identities = sorted([d for d in os.listdir(gallery_dir)
                         if os.path.isdir(os.path.join(gallery_dir, d))])
    if not identities:
        raise ValueError(f"No identity sub-folders found in: {gallery_dir}")

    print(f"\n[gallery] Building from {len(identities)} identities in: {gallery_dir}")

    for identity in identities:
        id_dir = os.path.join(gallery_dir, identity)
        img_files = [os.path.join(id_dir, f) for f in sorted(os.listdir(id_dir))
                     if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
        if not img_files:
            print(f"  [skip] {identity} — no images found")
            continue

        embeddings = []
        good_paths = []
        for img_path in img_files:
            try:
                crop, _ = detect_align_crop(img_path, detector, DETECT_CONF_THRESHOLD, MARGIN)
                e = embed(model, crop, device)
                embeddings.append(e)
                good_paths.append(img_path)
            except Exception as ex:
                print(f"  [warn] {identity}/{os.path.basename(img_path)}: {ex}")

        if not embeddings:
            print(f"  [skip] {identity} — no valid faces extracted")
            continue

        # average embeddings → re-normalise
        stacked = torch.cat(embeddings, dim=0)          # (N, D)
        mean_emb = stacked.mean(dim=0, keepdim=True)    # (1, D)
        mean_emb = F.normalize(mean_emb, p=2, dim=1)   # L2 re-norm

        gallery[identity] = {
            "embedding":   mean_emb,
            "image_paths": good_paths,
            "rep_image":   good_paths[0],               # representative image for visualisation
        }
        print(f"  [ok] {identity:30s}  ({len(good_paths)} image{'s' if len(good_paths)>1 else ''})")

    print(f"[gallery] Ready — {len(gallery)} identities enrolled.\n")
    return gallery


# ─────────────────────────────────────────────────────────
# MATCHING
# ─────────────────────────────────────────────────────────

def search_gallery(probe_embedding, gallery, top_k=5):
    """
    Computes cosine similarity of probe vs every gallery identity.
    Returns list of (identity, similarity) sorted highest first.
    """
    scores = []
    for identity, data in gallery.items():
        sim = F.cosine_similarity(probe_embedding, data["embedding"], dim=1).item()
        scores.append((identity, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ─────────────────────────────────────────────────────────
# VISUALISATION  (same design language as compare_faces.py)
# ─────────────────────────────────────────────────────────

MATCH_COLOR    = (80, 200, 120)
NO_MATCH_COLOR = (70, 100, 240)
BG_COLOR       = (18, 18, 18)
PANEL_COLOR    = (28, 28, 28)
DIVIDER_COLOR  = (45, 45, 45)
TEXT_COLOR     = (230, 230, 230)
DIM_COLOR      = (110, 110, 110)
RANK_COLORS    = [                          # rank bar accent colours
    (80, 200, 120),  # rank-1  green
    (100, 180, 240), # rank-2
    (140, 140, 200), # rank-3
    (100, 100, 160), # rank-4
    (70,  70, 120),  # rank-5
]

CARD_W  = 500
CARD_H  = 500
PAD     = 28
GAP     = 40
LABEL_H = 50
BAR_H   = 200   # results panel height

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_DUPLEX


def _txt(img, text, pos, font=FONT, scale=0.55, color=TEXT_COLOR, thick=1, anchor="tl"):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    if anchor == "tr": x -= tw
    elif anchor == "tc": x -= tw // 2
    cv2.putText(img, text, (x, y + th), font, scale, color, thick, cv2.LINE_AA)
    return tw, th


def draw_card(image_path, det_info, title, subtitle, accent):
    """Single image card with bbox, keypoints, title & subtitle strip."""
    card = np.full((CARD_H, CARD_W, 3), PANEL_COLOR, dtype=np.uint8)
    img_h = CARD_H - LABEL_H

    raw = cv2.imread(image_path)
    if raw is None:
        raw = np.zeros((img_h, CARD_W, 3), dtype=np.uint8)

    ih, iw = raw.shape[:2]
    sf = min(CARD_W / iw, img_h / ih)
    nw, nh = int(iw * sf), int(ih * sf)
    resized = cv2.resize(raw, (nw, nh), interpolation=cv2.INTER_AREA)
    ox, oy = (CARD_W - nw) // 2, (img_h - nh) // 2
    card[oy:oy+nh, ox:ox+nw] = resized

    # bbox
    x, y, w, h = det_info["box"]
    def sx(v): return int(v * sf + ox)
    def sy(v): return int(v * sf + oy)

    bx1, by1, bx2, by2 = sx(max(0,x)), sy(max(0,y)), sx(x+w), sy(y+h)
    cv2.rectangle(card, (bx1,by1), (bx2,by2), accent, 1, cv2.LINE_AA)

    TICK = 10
    for (cx,cy), ddx, ddy in zip(
        [(bx1,by1),(bx2,by1),(bx1,by2),(bx2,by2)],
        [1,-1,1,-1], [1,1,-1,-1]
    ):
        cv2.line(card,(cx,cy),(cx+ddx*TICK,cy), accent, 2, cv2.LINE_AA)
        cv2.line(card,(cx,cy),(cx,cy+ddy*TICK), accent, 2, cv2.LINE_AA)

    # keypoints
    kps = det_info["keypoints"]
    for pt in [kps["left_eye"],kps["right_eye"],kps["nose"],kps["mouth_left"],kps["mouth_right"]]:
        cv2.circle(card,(sx(pt[0]),sy(pt[1])),3,accent,-1,cv2.LINE_AA)

    # confidence badge
    conf_txt = f"{det_info['confidence']:.2f}"
    (ctw,cth),_ = cv2.getTextSize(conf_txt, FONT, 0.40, 1)
    bp = 4
    cv2.rectangle(card,(bx2-ctw-bp*2, by1),(bx2, by1+cth+bp*2), accent,-1)
    cv2.putText(card, conf_txt,(bx2-ctw-bp, by1+cth+bp), FONT,0.40, BG_COLOR,1,cv2.LINE_AA)

    # label strip
    sy0 = img_h
    cv2.line(card,(0,sy0),(CARD_W,sy0), DIVIDER_COLOR,1)
    _txt(card, title,    (CARD_W//2, sy0+6),  font=FONT_MONO, scale=0.60, color=TEXT_COLOR, thick=1, anchor="tc")
    fname = os.path.basename(image_path)
    if len(fname) > 34: fname = "…" + fname[-32:]
    _txt(card, fname,    (CARD_W//2, sy0+26), scale=0.38, color=DIM_COLOR, anchor="tc")
    _txt(card, subtitle, (CARD_W//2, sy0+38), scale=0.36, color=accent,    anchor="tc")

    return card


def draw_results_panel(top_matches, threshold, gallery, probe_det):
    """Bottom panel: rank bars + identity list."""
    total_w = PAD * 2 + CARD_W * 2 + GAP
    panel = np.full((BAR_H, total_w, 3), BG_COLOR, dtype=np.uint8)
    cv2.line(panel,(0,0),(total_w,0), DIVIDER_COLOR,1)

    # ── left half: similarity bars ──────────────────────────────────────────
    left_cx = PAD
    _txt(panel,"TOP MATCHES",(left_cx, 10), scale=0.38, color=DIM_COLOR)

    bar_x1 = left_cx
    bar_x2 = PAD + CARD_W
    row_h  = 30
    for i,(identity,sim) in enumerate(top_matches):
        y0 = 28 + i * row_h
        is_accepted = (i == 0 and sim >= threshold)
        col = RANK_COLORS[i] if i < len(RANK_COLORS) else DIM_COLOR

        # rank label
        _txt(panel, f"#{i+1}", (bar_x1, y0), scale=0.38, color=col)

        # name
        name_short = identity if len(identity)<=18 else identity[:17]+"…"
        _txt(panel, name_short, (bar_x1+28, y0), scale=0.42, color=TEXT_COLOR)

        # bar track
        track_x1 = bar_x1 + 160
        track_x2 = bar_x2 - 10
        bar_mid_y = y0 + 8
        cv2.line(panel,(track_x1, bar_mid_y),(track_x2, bar_mid_y), DIVIDER_COLOR,3,cv2.LINE_AA)
        fill_x = int(track_x1 + (track_x2-track_x1) * max(0.0,min(1.0,sim)))
        cv2.line(panel,(track_x1, bar_mid_y),(fill_x, bar_mid_y), col,3,cv2.LINE_AA)

        # threshold tick on bar
        thr_x = int(track_x1 + (track_x2-track_x1) * threshold)
        cv2.line(panel,(thr_x, bar_mid_y-5),(thr_x, bar_mid_y+5), DIM_COLOR,1,cv2.LINE_AA)

        # score text
        _txt(panel, f"{sim:.4f}", (track_x2+6, y0), scale=0.38, color=col)

    # threshold legend
    _txt(panel, f"threshold {threshold:.2f}", (bar_x1, 28 + len(top_matches)*row_h + 6),
         scale=0.35, color=DIM_COLOR)

    # ── vertical divider ────────────────────────────────────────────────────
    mid_x = total_w // 2
    cv2.line(panel,(mid_x,12),(mid_x,BAR_H-12), DIVIDER_COLOR,1)

    # ── right half: verdict ─────────────────────────────────────────────────
    right_cx = mid_x + (PAD + CARD_W // 2)
    top_identity, top_sim = top_matches[0]
    accepted = top_sim >= threshold
    accent   = MATCH_COLOR if accepted else NO_MATCH_COLOR
    verdict  = "IDENTIFIED" if accepted else "UNKNOWN"

    _txt(panel,"IDENTITY",(right_cx,10), scale=0.38, color=DIM_COLOR, anchor="tc")
    _txt(panel, top_identity if accepted else "— unknown —",
         (right_cx, 26), font=FONT_MONO, scale=0.80, color=accent, thick=2, anchor="tc")

    _txt(panel,"SIMILARITY",(right_cx,68), scale=0.35, color=DIM_COLOR, anchor="tc")
    _txt(panel, f"{top_sim:.4f}",
         (right_cx,82), font=FONT_MONO, scale=1.10, color=TEXT_COLOR, thick=2, anchor="tc")

    # verdict badge
    badge_y = 130
    badge_w, badge_h = 160, 38
    bx = right_cx - badge_w//2
    cv2.rectangle(panel,(bx,badge_y),(bx+badge_w,badge_y+badge_h), accent,1,cv2.LINE_AA)
    _txt(panel, verdict,(right_cx, badge_y+9), font=FONT_MONO, scale=0.70,
         color=accent, thick=2, anchor="tc")

    # check/cross inside badge
    ico_x = right_cx
    ico_y = badge_y + badge_h + 20
    cv2.circle(panel,(ico_x,ico_y),14, accent,1,cv2.LINE_AA)
    if accepted:
        pts = np.array([[ico_x-6,ico_y],[ico_x-1,ico_y+5],[ico_x+7,ico_y-6]],np.int32)
        cv2.polylines(panel,[pts],False,accent,2,cv2.LINE_AA)
    else:
        cv2.line(panel,(ico_x-6,ico_y-6),(ico_x+6,ico_y+6),accent,2,cv2.LINE_AA)
        cv2.line(panel,(ico_x+6,ico_y-6),(ico_x-6,ico_y+6),accent,2,cv2.LINE_AA)

    return panel


def show_result_window(probe_path, probe_det,
                       top_matches, gallery, threshold):
    """
    Shows popup:   [PROBE image]  |  [BEST MATCH gallery image]
                   [ranked similarity bar panel]
    """
    top_identity, top_sim = top_matches[0]
    accepted = top_sim >= threshold
    accent   = MATCH_COLOR if accepted else NO_MATCH_COLOR

    # ── probe card ──────────────────────────────────────────────────────────
    probe_card = draw_card(
        probe_path, probe_det,
        title    = "PROBE",
        subtitle = "unknown input",
        accent   = accent,
    )

    # ── best match card ─────────────────────────────────────────────────────
    match_img  = gallery[top_identity]["rep_image"]
    # build a dummy det_info for the gallery rep image
    match_det_info = _get_det_info_for_gallery_rep(match_img)

    match_label = top_identity if accepted else "NO MATCH"
    match_card  = draw_card(
        match_img,
        match_det_info,
        title    = f"BEST MATCH — {top_identity}" if accepted else "BEST MATCH",
        subtitle = f"sim {top_sim:.4f}  {'✓ accepted' if accepted else '✗ below threshold'}",
        accent   = accent,
    )

    # ── canvas ──────────────────────────────────────────────────────────────
    total_w = PAD * 2 + CARD_W * 2 + GAP
    top_h   = PAD * 2 + CARD_H
    canvas  = np.full((top_h, total_w, 3), BG_COLOR, dtype=np.uint8)
    canvas[PAD:PAD+CARD_H, PAD:PAD+CARD_W]            = probe_card
    canvas[PAD:PAD+CARD_H, PAD+CARD_W+GAP:PAD+CARD_W*2+GAP] = match_card

    # connector line
    mid_y = PAD + CARD_H // 2
    lx1   = PAD + CARD_W
    lx2   = PAD + CARD_W + GAP
    cv2.line(canvas,(lx1,mid_y),(lx2,mid_y), DIVIDER_COLOR,1,cv2.LINE_AA)
    cv2.circle(canvas,(lx1+GAP//2, mid_y),3, DIM_COLOR,-1,cv2.LINE_AA)

    # ── results panel ────────────────────────────────────────────────────────
    panel = draw_results_panel(top_matches, threshold, gallery, probe_det)
    full  = np.vstack([canvas, panel])

    win = "Gallery Probe — Face Identification"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, total_w, full.shape[0])
    cv2.imshow(win, full)
    print("\nResult window open — press any key or close to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# gallery rep image may be preprocessed (already cropped), detect face for viz
_detector_cache = None

def _get_det_info_for_gallery_rep(image_path):
    """
    Best-effort: detect face in gallery rep image for bbox viz.
    Falls back to a full-image bbox if detection fails.
    """
    global _detector_cache
    try:
        if _detector_cache is None:
            _detector_cache = MTCNN()
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        results = [f for f in _detector_cache.detect_faces(image_rgb)
                   if f["confidence"] >= DETECT_CONF_THRESHOLD]
        if results:
            face = max(results, key=lambda f: f["box"][2]*f["box"][3])
            return {"box": face["box"], "keypoints": face["keypoints"],
                    "confidence": face["confidence"]}
    except Exception:
        pass

    # fallback: cover the whole image
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    h, w = image_rgb.shape[:2]
    return {
        "box": [0, 0, w, h],
        "keypoints": {"left_eye":(w//3,h//3),"right_eye":(2*w//3,h//3),
                      "nose":(w//2,h//2),"mouth_left":(w//3,2*h//3),
                      "mouth_right":(2*w//3,2*h//3)},
        "confidence": 0.0,
    }


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # ── load model ───────────────────────────────────────────────────────────
    model    = load_trained_model(MODEL_PATH, device)
    detector = MTCNN()

    # ── build gallery ────────────────────────────────────────────────────────
    gallery = build_gallery(GALLERY_DIR, model, detector, device)

    # ── process probe ────────────────────────────────────────────────────────
    if not os.path.isfile(PROBE_IMAGE):
        raise FileNotFoundError(f"Probe image not found: {PROBE_IMAGE}")

    print(f"[probe] Processing: {PROBE_IMAGE}")
    probe_crop, probe_det = detect_align_crop(PROBE_IMAGE, detector, DETECT_CONF_THRESHOLD, MARGIN)
    probe_emb             = embed(model, probe_crop, device)

    # ── search ───────────────────────────────────────────────────────────────
    top_matches = search_gallery(probe_emb, gallery, top_k=TOP_K)

    # ── console output ───────────────────────────────────────────────────────
    top_identity, top_sim = top_matches[0]
    accepted = top_sim >= THRESHOLD

    print("\n" + "=" * 55)
    print(f"  Probe image   : {os.path.basename(PROBE_IMAGE)}")
    print(f"  Gallery size  : {len(gallery)} identities")
    print(f"  Threshold     : {THRESHOLD}")
    print("-" * 55)
    for rank,(ident,sim) in enumerate(top_matches, 1):
        flag = " ← BEST" if rank==1 else ""
        print(f"  Rank-{rank}  {ident:25s}  sim={sim:.4f}{flag}")
    print("-" * 55)
    if accepted:
        print(f"  DECISION : IDENTIFIED  →  {top_identity}  (sim={top_sim:.4f})")
    else:
        print(f"  DECISION : UNKNOWN  (best sim={top_sim:.4f} below threshold {THRESHOLD})")
    print("=" * 55)

    # ── visual window ────────────────────────────────────────────────────────
    show_result_window(PROBE_IMAGE, probe_det, top_matches, gallery, THRESHOLD)


if __name__ == "__main__":
    main()