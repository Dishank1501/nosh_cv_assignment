import os
import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Trial 9: Multi-Candidate Egg Segmentation
# ---------------------------------------------------------

OUTPUT_DIR = "outputs/trial9_multi_candidate"
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlays")
CSV_PATH = os.path.join(OUTPUT_DIR, "metrics.csv")

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

DEV_SPLIT = "splits/dev.txt"
REF_DIR = "outputs/automatic_masks"

# Candidate component area limits relative to image area.
MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 0.40

MORPH_KERNEL = 5


def read_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def find_reference(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    parts = image_path.replace("\\", "/").split("/")

    # Expected reference naming:
    # train__egg__image_stem_mask.png
    if len(parts) >= 3:
        candidate = os.path.join(
            REF_DIR,
            f"{parts[-3]}__{parts[-2]}__{stem}_mask.png"
        )
        if os.path.exists(candidate):
            return candidate

    # Fallback: search by unique image stem.
    for filename in os.listdir(REF_DIR):
        if filename.endswith("_mask.png") and stem in filename:
            return os.path.join(REF_DIR, filename)

    return None


def make_candidate_masks(image):
    """Generate several independent foreground candidates."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    L = lab[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    candidates = []

    # Candidate 1: Bright, relatively low-saturation pixels.
    bright_white = (
        (V >= 135) &
        (S <= 125)
    ).astype(np.uint8) * 255
    candidates.append(bright_white)

    # Candidate 2: Otsu threshold on LAB lightness.
    blurred_L = cv2.GaussianBlur(L, (5, 5), 0)
    _, otsu_high = cv2.threshold(
        blurred_L, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    candidates.append(otsu_high)

    # Candidate 3: Inverse Otsu threshold.
    _, otsu_low = cv2.threshold(
        blurred_L, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    candidates.append(otsu_low)

    # Candidate 4: Difference from estimated border color in LAB.
    border = np.concatenate([
        lab[0, :, :],
        lab[-1, :, :],
        lab[:, 0, :],
        lab[:, -1, :]
    ], axis=0)

    bg_color = np.median(border, axis=0)
    distance = np.linalg.norm(
        lab.astype(np.float32) - bg_color.astype(np.float32),
        axis=2
    )

    color_diff = (distance > 18).astype(np.uint8) * 255
    candidates.append(color_diff)

    return candidates


def filter_components(mask, image_shape):
    """Keep plausible connected components without forcing one object."""
    h, w = image_shape[:2]
    image_area = h * w

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL, MORPH_KERNEL)
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8
    )

    result = np.zeros((h, w), dtype=np.uint8)

    for label in range(1, count):
        x, y, cw, ch, area = stats[label]
        ratio = area / image_area

        if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
            continue

        aspect = cw / max(ch, 1)

        # Reject only extreme, thin components.
        if aspect > 5.0 or aspect < 0.20:
            continue

        component = (labels == label).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circularity = (
            4.0 * np.pi * contour_area /
            (perimeter * perimeter)
        )

        # Keep reasonably compact components, including non-perfect eggs.
        if circularity < 0.12:
            continue

        result[labels == label] = 255

    return result


def segment(image):
    candidates = make_candidate_masks(image)

    # Keep plausible components from each thresholding strategy.
    # Combine candidates instead of selecting only the largest contour.
    combined = np.zeros(image.shape[:2], dtype=np.uint8)

    for candidate in candidates:
        filtered = filter_components(candidate, image.shape)
        combined = cv2.bitwise_or(combined, filtered)

    # A final cleanup of small isolated noise.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN, kernel
    )

    return combined


def calculate_metrics(pred, ref):
    pred = pred > 0
    ref = ref > 0

    intersection = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()

    pred_count = pred.sum()
    ref_count = ref.sum()

    iou = intersection / union if union else 1.0
    dice_denom = pred_count + ref_count
    dice = (
        2 * intersection / dice_denom
        if dice_denom else 1.0
    )

    return float(iou), float(dice)


def save_overlay(image, mask, output_path):
    overlay = image.copy()

    # Create a green overlay and blend the complete images first.
    green = np.zeros_like(image)
    green[:, :, 1] = 255

    blended = cv2.addWeighted(image, 0.45, green, 0.55, 0)

    # Apply the blend only where the predicted mask is foreground.
    selected = mask > 0
    overlay[selected] = blended[selected]

    cv2.imwrite(output_path, overlay)


def main():
    paths = read_split(DEV_SPLIT)
    rows = []

    for i, image_path in enumerate(paths, start=1):
        image = cv2.imread(image_path)

        if image is None:
            print(f"[SKIP] Cannot read image: {image_path}")
            continue

        stem = os.path.splitext(os.path.basename(image_path))[0]

        mask = segment(image)

        mask_path = os.path.join(MASK_DIR, stem + "_mask.png")
        overlay_path = os.path.join(OVERLAY_DIR, stem + "_overlay.jpg")

        cv2.imwrite(mask_path, mask)
        save_overlay(image, mask, overlay_path)

        ref_path = find_reference(image_path)

        iou = np.nan
        dice = np.nan
        ref_ratio = np.nan

        if ref_path:
            ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

            if ref is not None:
                if ref.shape != mask.shape:
                    ref = cv2.resize(
                        ref,
                        (mask.shape[1], mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                iou, dice = calculate_metrics(mask, ref)
                ref_ratio = float(np.mean(ref > 0))
        else:
            print(f"[WARN] Reference mask missing: {image_path}")

        rows.append({
            "image": image_path,
            "iou_vs_sam": iou,
            "dice_vs_sam": dice,
            "predicted_foreground_ratio": float(np.mean(mask > 0)),
            "reference_foreground_ratio": ref_ratio,
            "prediction_mask": mask_path,
            "overlay_path": overlay_path,
            "reference_mask": ref_path or ""
        })

        if i % 10 == 0 or i == len(paths):
            print(f"Processed {i}/{len(paths)}")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)

    valid = df.dropna(subset=["iou_vs_sam", "dice_vs_sam"])

    print("\n===== TRIAL 9 RESULTS: DEVELOPMENT SPLIT =====")
    print(f"Images processed: {len(df)}")
    print(f"Images with reference masks: {len(valid)}")

    if len(valid):
        print(f"Mean IoU vs SAM: {valid['iou_vs_sam'].mean():.4f}")
        print(f"Median IoU vs SAM: {valid['iou_vs_sam'].median():.4f}")
        print(f"Mean Dice vs SAM: {valid['dice_vs_sam'].mean():.4f}")
        print(f"Zero-IoU cases: {(valid['iou_vs_sam'] == 0).sum()}/{len(valid)}")

    print(f"\nMetrics saved to: {CSV_PATH}")
    print(f"Masks saved to: {MASK_DIR}")
    print(f"Overlays saved to: {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
