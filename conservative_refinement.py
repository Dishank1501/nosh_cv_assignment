#!/usr/bin/env python3
"""
Trial 8: Conservative morphological cleanup of Trial 5 masks.

This experiment does NOT rerun segmentation or use SAM masks to make predictions.
It post-processes Trial 5's existing masks with a small morphological opening,
keeps the largest plausible connected component, and rejects the refinement if
it removes too much foreground. SAM pseudo-labels are used only for evaluation.

Run from ~/nosh_cv_assignment:
    source .venv/bin/activate
    python conservative_refinement.py

Outputs:
    outputs/trial8_conservative_refinement/
        masks/
        overlays/
        metrics.csv
        summary.txt

Important: This is a controlled experiment, not a guaranteed improvement.
"""

from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path.home() / "nosh_cv_assignment"
SPLIT_FILE = ROOT / "splits" / "dev.txt"
TRIAL5_CSV = ROOT / "outputs" / "trial5_grabcut_fallback" / "metrics.csv"
REFERENCE_DIR = ROOT / "outputs" / "automatic_masks"
OUT_DIR = ROOT / "outputs" / "trial8_conservative_refinement"
MASK_DIR = OUT_DIR / "masks"
OVERLAY_DIR = OUT_DIR / "overlays"

# Conservative settings: only remove very narrow structures.
OPEN_KERNEL_SIZE = 5
MIN_COMPONENT_RATIO = 0.02
MAX_COMPONENT_RATIO = 0.40
MAX_ALLOWED_AREA_LOSS = 0.15  # reject cleanup if it removes >15% of original foreground


def load_trial5_rows():
    if not TRIAL5_CSV.exists():
        raise FileNotFoundError(f"Trial 5 metrics not found: {TRIAL5_CSV}")
    with TRIAL5_CSV.open(newline="", encoding="utf-8") as f:
        return {row["image"]: row for row in csv.DictReader(f)}


def build_reference_index():
    index = {}
    for p in REFERENCE_DIR.glob("*_mask.png"):
        stem = p.name[:-len("_mask.png")]
        image_stem = stem.rsplit("__", 1)[-1]
        index[image_stem] = p
    return index


def keep_largest_plausible_component(mask):
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    possible = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        ratio = area / float(h * w)
        if MIN_COMPONENT_RATIO <= ratio <= MAX_COMPONENT_RATIO:
            possible.append((area, label))
    if not possible:
        return np.zeros_like(mask)
    chosen = max(possible, key=lambda x: x[0])[1]
    return np.where(labels == chosen, 255, 0).astype(np.uint8)


def conservative_refine(mask):
    """Try a small opening; revert if foreground loss is too large."""
    original = np.where(mask > 0, 255, 0).astype(np.uint8)
    original_area = int(cv2.countNonZero(original))
    if original_area == 0:
        return original, False, "empty_input"

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE)
    )
    refined = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)
    refined = keep_largest_plausible_component(refined)
    refined_area = int(cv2.countNonZero(refined))

    if refined_area == 0:
        return original, False, "reverted_empty"
    area_loss = max(0.0, (original_area - refined_area) / original_area)
    if area_loss > MAX_ALLOWED_AREA_LOSS:
        return original, False, "reverted_area_loss"

    changed = not np.array_equal(original, refined)
    return refined, changed, "refined" if changed else "unchanged"


def score(prediction, reference):
    pred = prediction > 0
    ref = reference > 0
    inter = int(np.logical_and(pred, ref).sum())
    union = int(np.logical_or(pred, ref).sum())
    denom = int(pred.sum() + ref.sum())
    iou = inter / union if union else 1.0
    dice = 2 * inter / denom if denom else 1.0
    return float(iou), float(dice), float(pred.mean()), float(ref.mean())


def save_overlay(image, mask, path):
    overlay = image.copy()
    foreground = mask > 0
    green = np.array([0, 220, 0], dtype=np.float32)
    overlay[foreground] = (
        0.55 * overlay[foreground].astype(np.float32) + 0.45 * green
    ).astype(np.uint8)
    cv2.imwrite(str(path), overlay)


def main():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Development split not found: {SPLIT_FILE}")
    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(f"SAM mask directory not found: {REFERENCE_DIR}")

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    trial5_rows = load_trial5_rows()
    refs = build_reference_index()

    paths = [
        Path(line.strip()) if Path(line.strip()).is_absolute() else ROOT / line.strip()
        for line in SPLIT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows = []
    missing_trial5 = 0
    missing_refs = 0

    for number, image_path in enumerate(paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{number}/{len(paths)}] SKIP unreadable image: {image_path}")
            continue

        rel = str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else str(image_path)
        t5_row = trial5_rows.get(rel)
        if t5_row is None:
            missing_trial5 += 1
            print(f"[{number}/{len(paths)}] Missing Trial 5 row: {rel}")
            continue

        mask_path = ROOT / t5_row["prediction_mask"]
        baseline = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if baseline is None:
            missing_trial5 += 1
            print(f"[{number}/{len(paths)}] Missing Trial 5 mask: {mask_path}")
            continue

        if baseline.shape != image.shape[:2]:
            baseline = cv2.resize(
                baseline, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        refined, applied, reason = conservative_refine(baseline)
        stem = image_path.stem
        out_mask = MASK_DIR / f"{stem}_mask.png"
        out_overlay = OVERLAY_DIR / f"{stem}_overlay.jpg"
        cv2.imwrite(str(out_mask), refined)
        save_overlay(image, refined, out_overlay)

        ref_path = refs.get(stem)
        if ref_path is None:
            missing_refs += 1
            ref = None
        else:
            ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
            if ref is None:
                missing_refs += 1
                ref = None
            elif ref.shape != refined.shape:
                ref = cv2.resize(
                    ref, (refined.shape[1], refined.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

        base_iou = base_dice = refined_iou = refined_dice = ref_ratio = float("nan")
        base_fg = float((baseline > 0).mean())
        refined_fg = float((refined > 0).mean())

        if ref is not None:
            base_iou, base_dice, _, ref_ratio = score(baseline, ref)
            refined_iou, refined_dice, _, _ = score(refined, ref)

        rows.append({
            "image": rel,
            "trial5_iou_vs_sam": base_iou,
            "trial8_iou_vs_sam": refined_iou,
            "iou_change": refined_iou - base_iou if np.isfinite(base_iou) else float("nan"),
            "trial5_dice_vs_sam": base_dice,
            "trial8_dice_vs_sam": refined_dice,
            "dice_change": refined_dice - base_dice if np.isfinite(base_dice) else float("nan"),
            "trial5_foreground_ratio": base_fg,
            "trial8_foreground_ratio": refined_fg,
            "reference_foreground_ratio": ref_ratio,
            "refinement_changed_mask": applied,
            "refinement_status": reason,
            "trial5_mask": str(mask_path.relative_to(ROOT)),
            "trial8_mask": str(out_mask.relative_to(ROOT)),
            "overlay": str(out_overlay.relative_to(ROOT)),
        })

        iou_msg = f"IoU {base_iou:.3f}->{refined_iou:.3f}" if np.isfinite(base_iou) else "reference missing"
        print(f"[{number}/{len(paths)}] {image_path.name}: {iou_msg}; {reason}")

    metrics_path = OUT_DIR / "metrics.csv"
    if rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    valid = [r for r in rows if np.isfinite(r["trial5_iou_vs_sam"]) and np.isfinite(r["trial8_iou_vs_sam"])]
    if valid:
        base_ious = np.array([r["trial5_iou_vs_sam"] for r in valid])
        new_ious = np.array([r["trial8_iou_vs_sam"] for r in valid])
        base_dices = np.array([r["trial5_dice_vs_sam"] for r in valid])
        new_dices = np.array([r["trial8_dice_vs_sam"] for r in valid])
        delta = new_ious - base_ious
        summary = [
            "TRIAL 8: Conservative morphological refinement of Trial 5",
            f"Images in split: {len(paths)}",
            f"Images evaluated: {len(valid)}",
            f"Missing Trial 5 masks/rows: {missing_trial5}",
            f"Missing SAM references: {missing_refs}",
            f"Refinement changed masks: {sum(bool(r['refinement_changed_mask']) for r in valid)}",
            f"Mean IoU Trial 5: {base_ious.mean():.4f}",
            f"Mean IoU Trial 8: {new_ious.mean():.4f}",
            f"Median IoU Trial 5: {np.median(base_ious):.4f}",
            f"Median IoU Trial 8: {np.median(new_ious):.4f}",
            f"Mean Dice Trial 5: {base_dices.mean():.4f}",
            f"Mean Dice Trial 8: {new_dices.mean():.4f}",
            f"Zero-IoU Trial 5: {int((base_ious == 0).sum())}/{len(valid)}",
            f"Zero-IoU Trial 8: {int((new_ious == 0).sum())}/{len(valid)}",
            f"Trial 8 improved IoU: {int((delta > 1e-9).sum())}",
            f"Trial 8 worsened IoU: {int((delta < -1e-9).sum())}",
            f"Trial 8 tied IoU: {int((np.abs(delta) <= 1e-9).sum())}",
            "",
            "SAM masks are pseudo-labels, not independently verified ground truth.",
            "This trial uses SAM masks only for evaluation, never for prediction/refinement.",
        ]
    else:
        summary = ["No evaluable images. Check Trial 5 mask paths and SAM reference index."]

    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary))
    print(f"\nMetrics: {metrics_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
