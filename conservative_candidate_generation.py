import os
import cv2
import numpy as np
import pandas as pd

DEV_SPLIT = "splits/dev.txt"
REF_DIR = "outputs/automatic_masks"
OUT_DIR = "outputs/trial10_conservative_candidates"
MASK_DIR = os.path.join(OUT_DIR, "masks")
OVERLAY_DIR = os.path.join(OUT_DIR, "overlays")
CSV_PATH = os.path.join(OUT_DIR, "metrics.csv")

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

MIN_AREA_RATIO = 0.008
MAX_AREA_RATIO = 0.40


def read_split():
    with open(DEV_SPLIT) as f:
        return [line.strip() for line in f if line.strip()]


def find_reference(image_path):
    parts = image_path.replace("\\", "/").split("/")
    stem = os.path.splitext(os.path.basename(image_path))[0]

    candidate = os.path.join(
        REF_DIR, f"{parts[-3]}__{parts[-2]}__{stem}_mask.png"
    )
    if os.path.exists(candidate):
        return candidate

    for name in os.listdir(REF_DIR):
        if name.endswith("_mask.png") and stem in name:
            return os.path.join(REF_DIR, name)

    return None


def get_candidates(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    L = lab[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Smooth small texture noise while preserving object boundaries.
    L_blur = cv2.GaussianBlur(L, (5, 5), 0)

    # Candidate A: bright, relatively low-saturation pixels.
    bright = ((V > 130) & (S < 120)).astype(np.uint8) * 255

    # Candidate B: bright regions in LAB lightness.
    _, light = cv2.threshold(
        L_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Candidate C: color difference from the image border.
    border = np.concatenate([
        lab[0, :, :], lab[-1, :, :],
        lab[:, 0, :], lab[:, -1, :]
    ], axis=0)

    background = np.median(border, axis=0)
    distance = np.linalg.norm(
        lab.astype(np.float32) - background.astype(np.float32),
        axis=2
    )
    contrast = (distance > 18).astype(np.uint8) * 255

    return [bright, light, contrast]


def candidate_score(contour, image_area):
    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    area_ratio = area / image_area
    if not MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return None

    aspect = w / h

    # Reject extreme thin structures such as tile lines and pot rims.
    if not 0.30 <= aspect <= 3.2:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # These are weak shape cues, not proof that the region is an egg.
    if circularity < 0.15 or solidity < 0.45:
        return None

    # Prefer compact, egg-like components without forcing a single object.
    score = (
        0.45 * min(circularity, 1.0)
        + 0.35 * min(solidity, 1.0)
        + 0.20 * min(area_ratio / 0.12, 1.0)
    )

    return score


def segment(image):
    h, w = image.shape[:2]
    image_area = h * w

    candidates = get_candidates(image)
    all_contours = []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for candidate in candidates:
        cleaned = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            score = candidate_score(contour, image_area)
            if score is not None:
                all_contours.append((score, contour))

    # Remove near-duplicate candidates from different thresholding methods.
    all_contours.sort(key=lambda item: item[0], reverse=True)

    selected = []
    for score, contour in all_contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        duplicate = False

        for _, previous in selected:
            px, py, pw, ph = cv2.boundingRect(previous)

            ix1, iy1 = max(x, px), max(y, py)
            ix2, iy2 = min(x + cw, px + pw), min(y + ch, py + ph)

            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = cw * ch + pw * ph - intersection

            if union > 0 and intersection / union > 0.65:
                duplicate = True
                break

        if not duplicate:
            selected.append((score, contour))

    mask = np.zeros((h, w), dtype=np.uint8)

    for _, contour in selected:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return mask


def metrics(pred, ref):
    pred = pred > 0
    ref = ref > 0

    intersection = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()

    pred_area = pred.sum()
    ref_area = ref.sum()

    iou = intersection / union if union else 1.0
    denom = pred_area + ref_area
    dice = 2 * intersection / denom if denom else 1.0

    return float(iou), float(dice)


def save_overlay(image, mask, path):
    green = np.zeros_like(image)
    green[:, :, 1] = 255

    blended = cv2.addWeighted(image, 0.45, green, 0.55, 0)
    overlay = image.copy()
    selected = mask > 0
    overlay[selected] = blended[selected]

    cv2.imwrite(path, overlay)


def main():
    rows = []
    paths = read_split()

    for i, image_path in enumerate(paths, 1):
        image = cv2.imread(image_path)
        if image is None:
            print("[SKIP]", image_path)
            continue

        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask = segment(image)

        mask_path = os.path.join(MASK_DIR, stem + "_mask.png")
        overlay_path = os.path.join(OVERLAY_DIR, stem + "_overlay.jpg")

        cv2.imwrite(mask_path, mask)
        save_overlay(image, mask, overlay_path)

        ref_path = find_reference(image_path)
        iou, dice, ref_ratio = np.nan, np.nan, np.nan

        if ref_path:
            ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

            if ref is not None:
                if ref.shape != mask.shape:
                    ref = cv2.resize(
                        ref, (mask.shape[1], mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                iou, dice = metrics(mask, ref)
                ref_ratio = float(np.mean(ref > 0))

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

    print("\n===== TRIAL 10 RESULTS =====")
    print("Images processed:", len(df))
    print("Images with references:", len(valid))

    if len(valid):
        print(f"Mean IoU vs SAM: {valid['iou_vs_sam'].mean():.4f}")
        print(f"Median IoU vs SAM: {valid['iou_vs_sam'].median():.4f}")
        print(f"Mean Dice vs SAM: {valid['dice_vs_sam'].mean():.4f}")
        print(f"Zero-IoU cases: {(valid['iou_vs_sam'] == 0).sum()}/{len(valid)}")
        print(
            "Mean foreground ratio:",
            f"{df['predicted_foreground_ratio'].mean():.4f}"
        )

    print("Metrics:", CSV_PATH)
    print("Masks:", MASK_DIR)
    print("Overlays:", OVERLAY_DIR)


if __name__ == "__main__":
    main()
