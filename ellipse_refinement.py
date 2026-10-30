import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / "nosh_cv_assignment"
SPLIT_FILE = ROOT / "splits" / "dev.txt"
REFERENCE_DIR = ROOT / "outputs" / "automatic_masks"

EXPERIMENT = "trial6_ellipse_refinement"
OUTPUT_DIR = ROOT / "outputs" / EXPERIMENT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep the Trial 1 illumination settings unchanged.
CHANNEL = 0
BACKGROUND_SIGMA = 80
SMOOTHING = "median"
MEDIAN_KERNEL = 5
BILATERAL_PARAMS = (9, 50, 50)

MORPH_KERNEL_SIZE = 5
MIN_AREA_RATIO = 0.02
MAX_AREA_RATIO = 0.40
MAX_ASPECT_RATIO = 2.8
MIN_CIRCULARITY = 0.25

KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
)


def read_image_paths():
    paths = []

    for line in SPLIT_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        path = Path(line)
        if not path.is_absolute():
            path = ROOT / path

        if path.exists():
            paths.append(path)
        else:
            print(f"WARNING: Missing image: {path}")

    return paths


def build_reference_index():
    index = {}

    for path in REFERENCE_DIR.glob("*_mask.png"):
        mask_stem = path.name[:-len("_mask.png")]
        image_stem = mask_stem.rsplit("__", 1)[-1]
        index[image_stem] = path

    return index


def correct_illumination(channel):
    channel = channel.astype(np.float32)

    background = cv2.GaussianBlur(
        channel, (0, 0), BACKGROUND_SIGMA
    )

    corrected = channel - background
    corrected = cv2.normalize(
        corrected, None, 0, 255, cv2.NORM_MINMAX
    )

    return corrected.astype(np.uint8)


def smooth(image):
    if SMOOTHING == "bilateral":
        d, sigma_color, sigma_space = BILATERAL_PARAMS
        return cv2.bilateralFilter(
            image, d, sigma_color, sigma_space
        )

    return cv2.medianBlur(image, MEDIAN_KERNEL)


def classical_segmentation(image):
    """Illumination correction, Otsu thresholding and shape filtering."""
    height, width = image.shape[:2]

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    corrected = correct_illumination(lab[:, :, CHANNEL])
    corrected = smooth(corrected)

    _, binary = cv2.threshold(
        corrected,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    candidates = []
    image_area = float(height * width)

    for threshold_mask in (binary, cv2.bitwise_not(binary)):
        cleaned = cv2.morphologyEx(
            threshold_mask, cv2.MORPH_OPEN, KERNEL
        )
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, KERNEL, iterations=2
        )

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            ratio = area / image_area

            if not MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO:
                continue

            _, _, box_w, box_h = cv2.boundingRect(contour)
            short_side = max(1, min(box_w, box_h))
            aspect = max(box_w, box_h) / short_side

            if aspect > MAX_ASPECT_RATIO:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = (
                4.0 * np.pi * area / (perimeter * perimeter)
                if perimeter > 0 else 0.0
            )

            if circularity < MIN_CIRCULARITY:
                continue

            score = area * (0.5 + min(circularity, 1.0))
            candidates.append((score, contour))

    output = np.zeros((height, width), dtype=np.uint8)

    if candidates:
        _, best_contour = max(candidates, key=lambda item: item[0])
        cv2.drawContours(
            output, [best_contour], -1, 255, cv2.FILLED
        )

    return output, corrected


def grabcut_fallback(image):
    """Use GrabCut only when classical segmentation returns an empty mask."""
    height, width = image.shape[:2]

    smoothed = cv2.bilateralFilter(image, 9, 50, 50)

    gc_mask = np.full(
        (height, width), cv2.GC_PR_BGD, dtype=np.uint8
    )

    bx = max(1, int(0.04 * width))
    by = max(1, int(0.04 * height))

    gc_mask[:by, :] = cv2.GC_BGD
    gc_mask[-by:, :] = cv2.GC_BGD
    gc_mask[:, :bx] = cv2.GC_BGD
    gc_mask[:, -bx:] = cv2.GC_BGD

    # Probable foreground prior near the image centre.
    cv2.ellipse(
        gc_mask,
        (width // 2, height // 2),
        (int(0.30 * width), int(0.30 * height)),
        0, 0, 360,
        cv2.GC_PR_FGD,
        -1,
    )

    # Small definite-foreground seed.
    cv2.ellipse(
        gc_mask,
        (width // 2, height // 2),
        (max(1, int(0.08 * width)), max(1, int(0.08 * height))),
        0, 0, 360,
        cv2.GC_FGD,
        -1,
    )

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            smoothed,
            gc_mask,
            None,
            bg_model,
            fg_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return np.zeros((height, width), dtype=np.uint8)

    result = np.where(
        (gc_mask == cv2.GC_FGD)
        | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        result, connectivity=8
    )

    if count <= 1:
        return np.zeros_like(result)

    # Keep the largest component that has a plausible area.
    image_area = float(height * width)
    components = []

    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        ratio = area / image_area

        if MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO:
            components.append((area, label))

    if not components:
        return np.zeros_like(result)

    _, best_label = max(components)
    component = np.where(labels == best_label, 255, 0).astype(
        np.uint8
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )

    component = cv2.morphologyEx(
        component, cv2.MORPH_OPEN, kernel
    )
    component = cv2.morphologyEx(
        component, cv2.MORPH_CLOSE, kernel
    )

    return component


def ellipse_refinement(mask):
    """
    Intersect the mask with a fitted ellipse.
    If fitting fails or removes too much foreground, keep the original.
    """
    if not np.any(mask):
        return mask.copy(), False

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return mask.copy(), False

    contour = max(contours, key=cv2.contourArea)

    if len(contour) < 5 or cv2.contourArea(contour) <= 0:
        return mask.copy(), False

    try:
        ellipse = cv2.fitEllipse(contour)
    except cv2.error:
        return mask.copy(), False

    ellipse_mask = np.zeros_like(mask)
    cv2.ellipse(
        ellipse_mask, ellipse, 255, thickness=cv2.FILLED
    )

    refined = cv2.bitwise_and(mask, ellipse_mask)

    original_area = np.count_nonzero(mask)
    refined_area = np.count_nonzero(refined)

    # Reject extreme changes. Refinement must retain at least 70%
    # of the original foreground pixels.
    if original_area == 0 or refined_area < 0.70 * original_area:
        return mask.copy(), False

    return refined, True


def calculate_metrics(prediction, reference):
    pred = prediction > 0
    ref = reference > 0

    intersection = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    denominator = pred.sum() + ref.sum()

    iou = float(intersection / union) if union else 1.0
    dice = (
        float(2 * intersection / denominator)
        if denominator else 1.0
    )

    return iou, dice


def main():
    image_paths = read_image_paths()

    if not image_paths:
        raise SystemExit(f"No images found in {SPLIT_FILE}")

    reference_index = build_reference_index()

    print(f"Development images: {len(image_paths)}")
    print(f"Reference masks indexed: {len(reference_index)}")

    rows = []
    fallback_calls = 0
    refinement_calls = 0
    missing_references = 0

    for idx, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        baseline, corrected = classical_segmentation(image)

        used_fallback = not np.any(baseline)

        if used_fallback:
            fallback_calls += 1
            baseline = grabcut_fallback(image)

        refined, was_refined = ellipse_refinement(baseline)

        if was_refined:
            refinement_calls += 1

        reference_path = reference_index.get(image_path.stem)

        if reference_path is None:
            missing_references += 1
            print(f"Missing SAM reference: {image_path.name}")
            continue

        reference = cv2.imread(
            str(reference_path), cv2.IMREAD_GRAYSCALE
        )

        if reference is None:
            missing_references += 1
            print(f"Could not read reference: {reference_path}")
            continue

        if reference.shape != refined.shape:
            reference = cv2.resize(
                reference,
                (refined.shape[1], refined.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Save both predictions for a fair, per-image comparison.
        baseline_path = (
            OUTPUT_DIR / f"{image_path.stem}_baseline.png"
        )
        refined_path = (
            OUTPUT_DIR / f"{image_path.stem}_refined.png"
        )
        corrected_path = (
            OUTPUT_DIR / f"{image_path.stem}_corrected.png"
        )

        cv2.imwrite(str(baseline_path), baseline)
        cv2.imwrite(str(refined_path), refined)
        cv2.imwrite(str(corrected_path), corrected)

        baseline_iou, baseline_dice = calculate_metrics(
            baseline, reference
        )
        refined_iou, refined_dice = calculate_metrics(
            refined, reference
        )

        rows.append({
            "image": str(image_path.relative_to(ROOT)),
            "baseline_iou_vs_sam": round(baseline_iou, 6),
            "baseline_dice_vs_sam": round(baseline_dice, 6),
            "refined_iou_vs_sam": round(refined_iou, 6),
            "refined_dice_vs_sam": round(refined_dice, 6),
            "baseline_foreground_ratio": round(
                float(np.mean(baseline > 0)), 6
            ),
            "refined_foreground_ratio": round(
                float(np.mean(refined > 0)), 6
            ),
            "used_grabcut_fallback": used_fallback,
            "ellipse_refinement_applied": was_refined,
            "baseline_mask": str(baseline_path.relative_to(ROOT)),
            "refined_mask": str(refined_path.relative_to(ROOT)),
        })

        print(
            f"[{idx}/{len(image_paths)}] "
            f"Baseline IoU={baseline_iou:.3f} "
            f"Refined IoU={refined_iou:.3f} "
            f"{image_path.name}"
        )

    csv_path = OUTPUT_DIR / "metrics.csv"

    fields = list(rows[0].keys()) if rows else [
        "image",
        "baseline_iou_vs_sam",
        "baseline_dice_vs_sam",
        "refined_iou_vs_sam",
        "refined_dice_vs_sam",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        print("No metrics calculated. Check the split and reference filenames.")
        return

    base_ious = np.array(
        [r["baseline_iou_vs_sam"] for r in rows]
    )
    new_ious = np.array(
        [r["refined_iou_vs_sam"] for r in rows]
    )
    base_dices = np.array(
        [r["baseline_dice_vs_sam"] for r in rows]
    )
    new_dices = np.array(
        [r["refined_dice_vs_sam"] for r in rows]
    )

    print("\n=== TRIAL 6: ELLIPSE REFINEMENT ===")
    print(f"Images evaluated:             {len(rows)}")
    print(f"Missing references:            {missing_references}")
    print(f"GrabCut fallback calls:        {fallback_calls}")
    print(f"Ellipse refinement applied:   {refinement_calls}")
    print()
    print(f"Baseline mean IoU:             {base_ious.mean():.4f}")
    print(f"Refined mean IoU:              {new_ious.mean():.4f}")
    print(f"Baseline median IoU:           {np.median(base_ious):.4f}")
    print(f"Refined median IoU:            {np.median(new_ious):.4f}")
    print(f"Baseline mean Dice:            {base_dices.mean():.4f}")
    print(f"Refined mean Dice:             {new_dices.mean():.4f}")
    print(f"Baseline zero-IoU images:      {(base_ious == 0).sum()}")
    print(f"Refined zero-IoU images:       {(new_ious == 0).sum()}")
    print(f"Images IoU improved:           {(new_ious > base_ious).sum()}")
    print(f"Images IoU unchanged:           {(new_ious == base_ious).sum()}")
    print(f"Images IoU worsened:           {(new_ious < base_ious).sum()}")
    print(f"\nMetrics CSV: {csv_path}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
