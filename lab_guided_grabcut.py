#!/usr/bin/env python3
"""Trial 7: LAB-guided GrabCut fallback. Run from ~/nosh_cv_assignment."""
from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path.home() / "nosh_cv_assignment"
SPLIT = ROOT / "splits/dev.txt"
REF_DIR = ROOT / "outputs/automatic_masks"
OUT = ROOT / "outputs/trial7_lab_guided_grabcut"
MASKS, OVERLAYS = OUT / "masks", OUT / "overlays"

CHANNEL = 0
BACKGROUND_SIGMA = 80
MEDIAN_KERNEL = 5
MORPH_SIZE = 5
MIN_AREA, MAX_AREA = 0.02, 0.40
MAX_ASPECT, MIN_CIRCULARITY = 2.8, 0.25


def reference_index():
    idx = {}
    for p in REF_DIR.glob("*_mask.png"):
        idx[p.name[:-9].rsplit("__", 1)[-1]] = p
    return idx


def illumination_correct(channel):
    x = channel.astype(np.float32)
    bg = cv2.GaussianBlur(x, (0, 0), BACKGROUND_SIGMA)
    x = x - bg
    x -= x.min()
    if x.max() > 0:
        x *= 255.0 / x.max()
    return np.clip(x, 0, 255).astype(np.uint8)


def binary_candidates(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    corrected = illumination_correct(lab[:, :, CHANNEL])
    corrected = cv2.medianBlur(corrected, MEDIAN_KERNEL)
    _, bright = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_SIZE, MORPH_SIZE))
    out = []
    for name, b in (("bright", bright), ("dark", cv2.bitwise_not(bright))):
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
        out.append((name, b))
    return corrected, out


def stats(contour, shape):
    h, w = shape[:2]
    area = cv2.contourArea(contour)
    x, y, bw, bh = cv2.boundingRect(contour)
    per = cv2.arcLength(contour, True)
    circ = 4 * np.pi * area / (per * per) if per > 0 else 0
    ratio = area / (h * w)
    aspect = max(bw, bh) / max(1, min(bw, bh))
    cx, cy = x + bw / 2, y + bh / 2
    center_dist = np.hypot((cx - w/2)/(w/2), (cy - h/2)/(h/2))
    return area, ratio, aspect, circ, center_dist


def choose_contour(binary, shape, relaxed=False):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area, ratio, aspect, circ, dist = stats(c, shape)
        limits = (0.008, 0.60, 4.5, 0.08) if relaxed else (MIN_AREA, MAX_AREA, MAX_ASPECT, MIN_CIRCULARITY)
        if not limits[0] <= ratio <= limits[1] or aspect > limits[2] or circ < limits[3]:
            continue
        score = 0.5 * min(circ, 1) + 0.25 * (1 - min(abs(aspect-1.45)/2.5, 1)) + 0.25 * (1-min(dist/1.5, 1))
        if best is None or score > best[0]:
            best = (score, c, (area, ratio, aspect, circ, dist))
    return (None, None) if best is None else (best[1], best[2])


def contour_mask(contour, shape):
    m = np.zeros(shape[:2], np.uint8)
    if contour is not None:
        cv2.drawContours(m, [contour], -1, 255, cv2.FILLED)
    return m


def lab_primary(image, candidates):
    best, best_score = None, -1
    for _, binary in candidates:
        contour, s = choose_contour(binary, image.shape)
        if contour is not None:
            score = s[0] * (0.5 + min(s[3], 1))
            if score > best_score:
                best, best_score = contour, score
    return contour_mask(best, image.shape)


def adaptive_seed(image, candidates):
    h, w = image.shape[:2]
    border = max(3, round(min(h, w) * 0.025))
    seed = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    seed[:border] = cv2.GC_BGD
    seed[-border:] = cv2.GC_BGD
    seed[:, :border] = cv2.GC_BGD
    seed[:, -border:] = cv2.GC_BGD
    best = None
    for polarity, binary in candidates:
        c, s = choose_contour(binary, image.shape, relaxed=True)
        if c is None:
            continue
        area, ratio, aspect, circ, dist = s
        score = 0.55*min(circ, 1) + 0.20*(1-min(abs(aspect-1.45)/3, 1)) + 0.15*(1-min(dist/1.5, 1)) + 0.10*min(ratio/0.20, 1)
        if best is None or score > best[0]:
            best = (score, polarity, c)
    if best:
        candidate = contour_mask(best[2], image.shape)
        seed[candidate > 0] = cv2.GC_PR_FGD
        safe = candidate.copy()
        safe[:border] = 0; safe[-border:] = 0
        safe[:, :border] = 0; safe[:, -border:] = 0
        k = max(3, int(round(min(h, w)*0.025)) | 1)
        sure = cv2.erode(safe, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if cv2.countNonZero(sure):
            seed[sure > 0] = cv2.GC_FGD
        method = "lab_candidate_" + best[1]
    else:
        # Explicit last resort when thresholding finds no usable candidate.
        ellipse = np.zeros((h, w), np.uint8)
        cv2.ellipse(ellipse, (w//2, h//2), (max(2, int(w*.30)), max(2, int(h*.30))), 0, 0, 360, 255, -1)
        seed[ellipse > 0] = cv2.GC_PR_FGD
        sure = np.zeros((h, w), np.uint8)
        cv2.ellipse(sure, (w//2, h//2), (max(2, int(w*.08)), max(2, int(h*.08))), 0, 0, 360, 255, -1)
        seed[sure > 0] = cv2.GC_FGD
        method = "ellipse_last_resort"
    seed[:border] = cv2.GC_BGD; seed[-border:] = cv2.GC_BGD
    seed[:, :border] = cv2.GC_BGD; seed[:, -border:] = cv2.GC_BGD
    return seed, method


def grabcut_fallback(image, candidates):
    seed, method = adaptive_seed(image, candidates)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, seed, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
        m = np.where((seed == cv2.GC_FGD) | (seed == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        n, labels, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
        possible = [(st[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
                    if MIN_AREA <= st[i, cv2.CC_STAT_AREA] / (m.shape[0]*m.shape[1]) <= MAX_AREA]
        if not possible:
            return np.zeros_like(m), method
        label = max(possible)[1]
        return np.where(labels == label, 255, 0).astype(np.uint8), method
    except cv2.error as e:
        print("GrabCut warning:", e)
        return np.zeros(image.shape[:2], np.uint8), method + "_failed"


def metrics(pred, ref):
    p, r = pred > 0, ref > 0
    inter = np.logical_and(p, r).sum()
    union = np.logical_or(p, r).sum()
    iou = float(inter / union) if union else 1.0
    denom = p.sum() + r.sum()
    dice = float(2*inter/denom) if denom else 1.0
    return iou, dice, float(p.mean()), float(r.mean())


def save_overlay(image, mask, path):
    overlay = image.copy()
    fg = mask > 0
    overlay[fg] = (0.55*overlay[fg] + 0.45*np.array([0, 220, 0])).astype(np.uint8)
    cv2.imwrite(str(path), overlay)


def main():
    if not SPLIT.exists():
        raise FileNotFoundError(f"Missing {SPLIT}; run from the existing project layout.")
    if not REF_DIR.exists():
        raise FileNotFoundError(f"Missing SAM pseudo-label directory: {REF_DIR}")
    MASKS.mkdir(parents=True, exist_ok=True); OVERLAYS.mkdir(parents=True, exist_ok=True)
    refs = reference_index()
    paths = [Path(x.strip()) if Path(x.strip()).is_absolute() else ROOT/ x.strip()
             for x in SPLIT.read_text().splitlines() if x.strip()]
    rows, fallback_count, missing = [], 0, 0
    for i, path in enumerate(paths, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"[{i}/{len(paths)}] SKIP unreadable: {path}")
            continue
        _, candidates = binary_candidates(image)
        pred = lab_primary(image, candidates)
        fallback = not np.any(pred)
        method = "not_used"
        if fallback:
            fallback_count += 1
            pred, method = grabcut_fallback(image, candidates)
        stem = path.stem
        mp, op = MASKS/f"{stem}_mask.png", OVERLAYS/f"{stem}_overlay.jpg"
        cv2.imwrite(str(mp), pred); save_overlay(image, pred, op)
        ref_path = refs.get(stem)
        if ref_path is None:
            missing += 1
            iou = dice = fg = ref_fg = float("nan")
        else:
            ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
            if ref is None:
                missing += 1; iou = dice = fg = ref_fg = float("nan")
            else:
                if ref.shape != pred.shape:
                    ref = cv2.resize(ref, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                iou, dice, fg, ref_fg = metrics(pred, ref)
        rows.append({
            "image": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "iou_vs_sam_pseudolabel": iou, "dice_vs_sam_pseudolabel": dice,
            "prediction_foreground_ratio": fg, "reference_foreground_ratio": ref_fg,
            "used_grabcut_fallback": fallback, "grabcut_seed_method": method,
            "mask_path": str(mp.relative_to(ROOT)), "overlay_path": str(op.relative_to(ROOT))
        })
        print(f"[{i}/{len(paths)}] {path.name}: IoU={iou:.3f}" if np.isfinite(iou)
              else f"[{i}/{len(paths)}] {path.name}: reference missing")
    csv_path = OUT/"metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    valid = [r for r in rows if np.isfinite(r["iou_vs_sam_pseudolabel"])]
    if valid:
        ious = np.array([r["iou_vs_sam_pseudolabel"] for r in valid])
        dices = np.array([r["dice_vs_sam_pseudolabel"] for r in valid])
        summary = [
            "TRIAL 7: LAB-guided GrabCut fallback",
            f"Images in split: {len(paths)}", f"Images evaluated: {len(valid)}",
            f"Missing references: {missing}", f"GrabCut fallback calls: {fallback_count}",
            f"Mean IoU vs SAM pseudo-labels: {ious.mean():.4f}",
            f"Median IoU vs SAM pseudo-labels: {np.median(ious):.4f}",
            f"Mean Dice vs SAM pseudo-labels: {dices.mean():.4f}",
            f"Median Dice vs SAM pseudo-labels: {np.median(dices):.4f}",
            f"Zero-IoU images: {int((ious == 0).sum())}/{len(valid)}",
            f"Empty predictions: {int((np.array([r['prediction_foreground_ratio'] for r in valid]) == 0).sum())}/{len(valid)}",
            "", "SAM masks are pseudo-labels, not independently verified ground truth.",
            "Compare these metrics with Trial 5 on the same dev split."
        ]
    else:
        summary = ["No evaluable images. Check the split paths and SAM mask filenames."]
    (OUT/"summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary))
    print(f"\nMetrics: {csv_path}\nSummary: {OUT/'summary.txt'}")


if __name__ == "__main__":
    main()
