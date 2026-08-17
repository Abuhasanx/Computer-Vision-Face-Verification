
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc,
    f1_score, precision_score, recall_score,
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SCORES_CSV   = r"D:\biztech\results\pair_similarity_scores.csv"
OUTPUT_DIR   = r"D:\biztech\results"
# ─────────────────────────────────────────

# ── colour palette (dark theme) ─────────────────────────
BG     = '#111111'
PANEL  = '#1a1a1a'
GRID   = '#2a2a2a'
GREEN  = '#50c878'
RED    = '#e05c6a'
BLUE   = '#5b8dee'
ORANGE = '#f4a340'
PURPLE = '#b57bee'
TEXT   = '#e0e0e0'
DIM    = '#666666'


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def tar_at_far(fpr, tpr, thresholds, target_far):
    """Return (TAR, threshold) at the closest FAR <= target_far."""
    idx = np.searchsorted(fpr, target_far, side='right') - 1
    idx = max(0, min(idx, len(fpr) - 1))
    return float(tpr[idx]), float(thresholds[idx])


def best_f1_threshold(y_true, y_score, thresholds):
    """Sweep thresholds and return (best_thr, best_f1, precision, recall)."""
    best_thr, best_f1 = thresholds[0], 0.0
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        f1   = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    pred  = (y_score >= best_thr).astype(int)
    return (float(best_thr),
            float(best_f1),
            float(precision_score(y_true, pred, zero_division=0)),
            float(recall_score(y_true, pred, zero_division=0)))


# ─────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────

def compute_metrics(y_true, y_score):
    """Compute all Task 6 + Task 7 metrics. Returns a dict."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    # EER — point where FAR ≈ FRR
    fnr     = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_thr = float(thresholds[eer_idx])

    tar_1,    thr_1    = tar_at_far(fpr, tpr, thresholds, 0.01)
    tar_01,   thr_01   = tar_at_far(fpr, tpr, thresholds, 0.001)

    best_thr, best_f1, prec, rec = best_f1_threshold(y_true, y_score, thresholds)

    return {
        "auc":           roc_auc,
        "eer":           eer,
        "eer_threshold": eer_thr,
        "tar_at_far_1pct":    tar_1,
        "thr_at_far_1pct":    thr_1,
        "tar_at_far_01pct":   tar_01,
        "thr_at_far_01pct":   thr_01,
        "best_f1_threshold":  best_thr,
        "best_f1":            best_f1,
        "precision":          prec,
        "recall":             rec,
        "n_genuine":   int((y_true == 1).sum()),
        "n_impostor":  int((y_true == 0).sum()),
        # raw arrays (not saved to JSON)
        "_fpr": fpr, "_tpr": tpr, "_thr": thresholds,
    }


# ─────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────

def apply_dark_theme():
    plt.rcParams.update({
        'figure.facecolor':  BG,
        'axes.facecolor':    PANEL,
        'axes.edgecolor':    GRID,
        'axes.labelcolor':   TEXT,
        'axes.titlecolor':   TEXT,
        'xtick.color':       DIM,
        'ytick.color':       DIM,
        'text.color':        TEXT,
        'grid.color':        GRID,
        'grid.linewidth':    0.6,
        'font.family':       'monospace',
        'font.size':         9,
    })


def plot_roc(ax, fpr, tpr, thresholds, metrics):
    roc_auc = metrics["auc"]
    eer_thr = metrics["eer_threshold"]
    eer     = metrics["eer"]

    ax.plot(fpr, tpr, color=GREEN, lw=2,
            label=f'ROC  (AUC = {roc_auc:.4f})', zorder=3)
    ax.fill_between(fpr, tpr, alpha=0.07, color=GREEN)
    ax.plot([0, 1], [0, 1], color=DIM, lw=1, linestyle='--',
            label='Random (AUC = 0.50)')

    # EER point
    eer_idx = int(np.argmin(np.abs(fpr - (1 - tpr))))
    ax.scatter([fpr[eer_idx]], [tpr[eer_idx]], color=ORANGE, s=60, zorder=5)
    ax.annotate(f'EER={eer:.3f}\nthr={eer_thr:.2f}',
                xy=(fpr[eer_idx], tpr[eer_idx]),
                xytext=(fpr[eer_idx] + 0.07, tpr[eer_idx] - 0.10),
                color=ORANGE, fontsize=7.5,
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1))

    # TAR@FAR=1%
    idx_far1 = np.searchsorted(fpr, 0.01, side='right') - 1
    ax.scatter([fpr[idx_far1]], [tpr[idx_far1]], color=PURPLE, s=60, zorder=5)
    ax.annotate(f'TAR={metrics["tar_at_far_1pct"]:.3f}\n@FAR=1%',
                xy=(fpr[idx_far1], tpr[idx_far1]),
                xytext=(fpr[idx_far1] + 0.07, tpr[idx_far1] - 0.10),
                color=PURPLE, fontsize=7.5,
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1))

    ax.set_xlabel('False Positive Rate (FPR) — Impostors accepted')
    ax.set_ylabel('True Positive Rate (TPR) — Genuines accepted')
    ax.set_title('ROC Curve', fontweight='bold', pad=8)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.text(0.52, 0.14, f'AUC = {roc_auc:.4f}',
            color=GREEN, fontsize=11, fontweight='bold',
            transform=ax.transAxes, ha='center')


def plot_distribution(ax, y_true, y_score, metrics):
    genuine  = y_score[y_true == 1]
    impostor = y_score[y_true == 0]
    bins = np.linspace(-0.6, 1.0, 60)

    ax.hist(impostor, bins=bins, color=RED,   alpha=0.65, density=True,
            label=f'Impostors (n={len(impostor)})')
    ax.hist(genuine,  bins=bins, color=GREEN, alpha=0.65, density=True,
            label=f'Genuines  (n={len(genuine)})')
    ax.axvline(metrics["best_f1_threshold"], color=ORANGE, lw=1.5, linestyle='--',
               label=f'Best threshold={metrics["best_f1_threshold"]:.2f}')
    ax.axvline(metrics["eer_threshold"], color=PURPLE, lw=1.0, linestyle=':',
               label=f'EER thr={metrics["eer_threshold"]:.2f}')

    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution', fontweight='bold', pad=8)
    ax.legend(fontsize=7.5, framealpha=0.3)
    ax.grid(True, alpha=0.4)


def plot_f1_curve(ax, y_true, y_score, thresholds, metrics):
    f1s = np.array([f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
                    for t in thresholds])
    mask = (thresholds >= -0.5) & (thresholds <= 1.0)

    ax.plot(thresholds[mask], f1s[mask], color=BLUE, lw=1.8)
    ax.axvline(metrics["best_f1_threshold"], color=ORANGE, lw=1.5, linestyle='--',
               label=f'Best F1={metrics["best_f1"]:.4f}\nthr={metrics["best_f1_threshold"]:.4f}')
    ax.scatter([metrics["best_f1_threshold"]], [metrics["best_f1"]],
               color=ORANGE, s=55, zorder=5)

    ax.set_xlabel('Cosine Similarity Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Threshold', fontweight='bold', pad=8)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.4)


def plot_far_frr(ax, fpr, tpr, thresholds, metrics):
    frr = 1.0 - tpr
    mask = (thresholds >= -0.5) & (thresholds <= 1.0)

    ax.plot(thresholds[mask], fpr[mask], color=RED,   lw=1.8,
            label='FAR (False Accept Rate)')
    ax.plot(thresholds[mask], frr[mask], color=GREEN, lw=1.8,
            label='FRR (False Reject Rate)')
    ax.axvline(metrics["eer_threshold"], color=ORANGE, lw=1.2, linestyle='--',
               label=f'EER={metrics["eer"]:.4f} @ thr={metrics["eer_threshold"]:.2f}')
    ax.scatter([metrics["eer_threshold"]], [metrics["eer"]],
               color=ORANGE, s=55, zorder=5)

    ax.set_xlabel('Cosine Similarity Threshold')
    ax.set_ylabel('Error Rate')
    ax.set_title('FAR & FRR vs Threshold', fontweight='bold', pad=8)
    ax.legend(fontsize=7.5, framealpha=0.3)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-0.02, 1.05)


def plot_summary(ax, metrics):
    ax.axis('off')
    ax.set_title('Metrics Summary', fontweight='bold', pad=8, color=TEXT)

    rows = [
        ('AUC',              f'{metrics["auc"]:.4f}',                  GREEN),
        ('EER',              f'{metrics["eer"]:.4f}',                  ORANGE),
        ('EER Threshold',    f'{metrics["eer_threshold"]:.4f}',        ORANGE),
        ('TAR @ FAR=1%',     f'{metrics["tar_at_far_1pct"]:.4f}',      PURPLE),
        ('TAR @ FAR=0.1%',   f'{metrics["tar_at_far_01pct"]:.4f}',     PURPLE),
        ('Best F1 Score',    f'{metrics["best_f1"]:.4f}',              BLUE),
        ('Best Threshold',   f'{metrics["best_f1_threshold"]:.4f}',    BLUE),
        ('Precision',        f'{metrics["precision"]:.4f}',            BLUE),
        ('Recall',           f'{metrics["recall"]:.4f}',               BLUE),
        ('Genuine pairs',    f'{metrics["n_genuine"]}',                TEXT),
        ('Impostor pairs',   f'{metrics["n_impostor"]}',               TEXT),
    ]

    y = 0.97
    for label, value, color in rows:
        ax.text(0.05, y, label, transform=ax.transAxes,
                fontsize=8.5, color=DIM, va='top')
        ax.text(0.72, y, value, transform=ax.transAxes,
                fontsize=8.5, color=color, va='top',
                fontweight='bold', ha='right')
        y -= 0.085


def generate_figure(y_true, y_score, metrics, out_path):
    apply_dark_theme()
    fpr = metrics["_fpr"]
    tpr = metrics["_tpr"]
    thr = metrics["_thr"]

    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle('Face Verification — ROC Analysis',
                 fontsize=15, color=TEXT, fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.32,
                           left=0.06, right=0.97, top=0.92, bottom=0.07)

    plot_roc(fig.add_subplot(gs[:, 0]),  fpr, tpr, thr, metrics)
    plot_distribution(fig.add_subplot(gs[0, 1]), y_true, y_score, metrics)
    plot_f1_curve(fig.add_subplot(gs[0, 2]),  y_true, y_score, thr, metrics)
    plot_far_frr(fig.add_subplot(gs[1, 1]),   fpr, tpr, thr, metrics)
    plot_summary(fig.add_subplot(gs[1, 2]),   metrics)

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"[saved] {out_path}")


# ─────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────

def print_report(metrics):
    sep = "=" * 55
    print(f"\n{sep}")
    print("  FACE VERIFICATION — ROC ANALYSIS REPORT")
    print(sep)
    print(f"  Dataset")
    print(f"    Genuine pairs   : {metrics['n_genuine']}")
    print(f"    Impostor pairs  : {metrics['n_impostor']}")
    print(f"\n  ROC / AUC")
    print(f"    AUC             : {metrics['auc']:.4f}")
    print(f"\n  Equal Error Rate (EER)")
    print(f"    EER             : {metrics['eer']:.4f}  ({metrics['eer']*100:.2f}%)")
    print(f"    EER threshold   : {metrics['eer_threshold']:.4f}")
    print(f"\n  Operating Points")
    print(f"    TAR @ FAR=1%    : {metrics['tar_at_far_1pct']:.4f}  (thr={metrics['thr_at_far_1pct']:.4f})")
    print(f"    TAR @ FAR=0.1%  : {metrics['tar_at_far_01pct']:.4f}  (thr={metrics['thr_at_far_01pct']:.4f})")
    print(f"\n  Classification (F1-optimal threshold)")
    print(f"    Threshold       : {metrics['best_f1_threshold']:.4f}")
    print(f"    F1 Score        : {metrics['best_f1']:.4f}")
    print(f"    Precision       : {metrics['precision']:.4f}")
    print(f"    Recall          : {metrics['recall']:.4f}")
    print(sep)
    print("\n  HOW THRESHOLD AFFECTS ERRORS")
    print("  ─────────────────────────────────────────────")
    print("  ↑ Threshold (stricter):")
    print("    → FAR ↓  (fewer impostors accepted = fewer false accepts)")
    print("    → FRR ↑  (more genuines rejected  = more false rejects)")
    print("    → Better security, worse convenience")
    print()
    print("  ↓ Threshold (looser):")
    print("    → FAR ↑  (more impostors accepted = more false accepts)")
    print("    → FRR ↓  (fewer genuines rejected = fewer false rejects)")
    print("    → Better convenience, worse security")
    print()
    print("  EER = point where FAR == FRR  (balanced operating point)")
    print(f"  Best F1 threshold balances precision & recall = {metrics['best_f1_threshold']:.4f}")
    print(sep)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(SCORES_CSV):
        raise FileNotFoundError(f"Scores file not found: {SCORES_CSV}\n"
                                f"Run evaluate_pairs.py first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading: {SCORES_CSV}")
    df      = pd.read_csv(SCORES_CSV)
    y_true  = df['label'].values
    y_score = df['cosine_similarity'].values

    print(f"Pairs   : {len(df)}  ({(y_true==1).sum()} genuine, {(y_true==0).sum()} impostor)")

    # ── compute ──────────────────────────────────────────
    metrics = compute_metrics(y_true, y_score)

    # ── report ───────────────────────────────────────────
    print_report(metrics)

    # ── save JSON (drop raw arrays) ───────────────────────
    json_path = os.path.join(OUTPUT_DIR, "roc_metrics.json")
    save_metrics = {k: v for k, v in metrics.items() if not k.startswith('_')}
    with open(json_path, 'w') as f:
        json.dump(save_metrics, f, indent=2)
    print(f"[saved] {json_path}")

    # ── save figure ───────────────────────────────────────
    png_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    generate_figure(y_true, y_score, metrics, png_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
