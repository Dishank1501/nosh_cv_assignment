import csv
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PATHS AND EXPERIMENT CONFIGURATION
# ============================================================

ROOT = Path.home() / "nosh_cv_assignment"

SPLIT_FILE = ROOT / "splits" / "dev.txt"
REFERENCE_DIR = ROOT / "outputs" / "automatic_masks"

# Separate output directory so previous experiments are preserved.
EXPERIMENT_NAME = "trial5_grabcut_fallback"
OUTPUT_DIR = ROOT / "outputs" / EXPERIMENT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LAB ILLUMINATION SEGMENTATION CONFIGURATION
# Keep the original LAB settings unchanged for this trial.
# ============================================================

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

PENALIZE_BORDER_TOUCH = False
BORDER_PENALTY = 0.3
BORDER_MARGIN = 2

KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
)


# ============================================================
# DATASET AND REFERENCE MASK HELPERS
# ============================================================

def read_image_paths():
    """Read image paths from the development split."""

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
    """
    Index SAM reference masks by their original image stems.

    Example:
    test__egg__example_image_mask.png
    becomes:
    example_image
    """

    index = {}

    for path in REFERENCE_DIR.glob("*_mask.png"):
        mask_stem = path.name[:-len("_mask.png")]
        image_stem = mask_stem.rsplit("__", 1)[-1]

        index[image_stem] = path

    return index


# ============================================================
# LAB ILLUMINATION CORRECTION
# ============================================================

def correct_illumination(channel):
    """
    Subtract large-scale background illumination.

    The signed difference is normalized to uint8 for thresholding.
    """

    channel_float = channel.astype(np.float32)

    background = cv2.GaussianBlur(
        channel_float,
        (0, 0),
        BACKGROUND_SIGMA
    )

    corrected = channel_float - background

    corrected = cv2.normalize(
        corrected,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return corrected.astype(np.uint8)


def smooth(image):
    """Apply the configured smoothing method."""

    if SMOOTHING == "bilateral":
        diameter, sigma_color, sigma_space = BILATERAL_PARAMS

        return cv2.bilateralFilter(
            image,
            diameter,
            sigma_color,
            sigma_space
        )

    return cv2.medianBlur(image, MEDIAN_KERNEL)


def touches_border(contour, width, height):
    """Check whether a contour touches the image border."""

    x, y, box_width, box_height = cv2.boundingRect(contour)

    margin = BORDER_MARGIN

    return (
        x <= margin
        or y <= margin
        or x + box_width >= width - margin
        or y + box_height >= height - margin
    )


# ============================================================
# GRABCUT FALLBACK
# Runs only if the LAB segmentation returns an empty mask.
# ============================================================

def grabcut_fallback(image):
    """
    Attempt segmentation using GrabCut when LAB segmentation fails.

    Assumption:
    The egg is approximately centered in the image.

    Limitation:
    This spatial prior may fail for off-center eggs or scenes
    containing multiple eggs.
    """

    height, width = image.shape[:2]

    if height < 5 or width < 5:
        return np.zeros((height, width), dtype=np.uint8)

    smoothed = cv2.bilateralFilter(
        image,
        9,
        50,
        50
    )

    # Initialize all pixels as probable background.
    grabcut_mask = np.full(
        (height, width),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    # Define definite background near the image borders.
    border_x = max(1, int(0.04 * width))
    border_y = max(1, int(0.04 * height))

    grabcut_mask[:border_y, :] = cv2.GC_BGD
    grabcut_mask[-border_y:, :] = cv2.GC_BGD
    grabcut_mask[:, :border_x] = cv2.GC_BGD
    grabcut_mask[:, -border_x:] = cv2.GC_BGD

    # Define a central probable foreground region.
    cv2.ellipse(
        grabcut_mask,
        (width // 2, height // 2),
        (
            max(1, int(0.30 * width)),
            max(1, int(0.30 * height))
        ),
        0,
        0,
        360,
        cv2.GC_PR_FGD,
        -1
    )

    # Define a smaller central region as definite foreground.
    cv2.ellipse(
        grabcut_mask,
        (width // 2, height // 2),
        (
            max(1, int(0.08 * width)),
            max(1, int(0.08 * height))
        ),
        0,
        0,
        360,
        cv2.GC_FGD,
        -1
    )

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            smoothed,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_MASK
        )

    except cv2.error as error:
        print(f"WARNING: GrabCut failed: {error}")

        return np.zeros(
            (height, width),
            dtype=np.uint8
        )

    # Retain definite and probable foreground pixels.
    foreground = (
        (grabcut_mask == cv2.GC_FGD)
        | (grabcut_mask == cv2.GC_PR_FGD)
    ).astype(np.uint8)

    # Extract connected components.
    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            foreground,
            connectivity=8
        )
    )

    if number_labels <= 1:
        return np.zeros(
            (height, width),
            dtype=np.uint8
        )

    # Label zero is the background.
    # Select the largest actual foreground component.
    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    largest_component = (
        labels == largest_label
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        largest_component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros(
            (height, width),
            dtype=np.uint8
        )

    contour = max(
        contours,
        key=cv2.contourArea
    )

    contour_area = cv2.contourArea(contour)
    area_ratio = contour_area / float(height * width)

    # Apply the same area bounds used by the LAB method.
    if not MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO:
        return np.zeros(
            (height, width),
            dtype=np.uint8
        )

    # Reject contours too small for ellipse fitting.
    if len(contour) < 5:
        return np.zeros(
            (height, width),
            dtype=np.uint8
        )

    result = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.drawContours(
        result,
        [contour],
        -1,
        255,
        cv2.FILLED
    )

    return result


# ============================================================
# CLASSICAL SEGMENTATION
# ============================================================

def segment(image):
    """
    Run the original LAB method first.

    Use GrabCut only if LAB returns an entirely empty mask.

    Returns:
        output: binary prediction mask
        corrected: illumination-corrected channel
        used_fallback: whether GrabCut was invoked
    """

    height, width = image.shape[:2]

    lab_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    selected_channel = lab_image[:, :, CHANNEL]

    corrected = correct_illumination(
        selected_channel
    )

    corrected = smooth(corrected)

    # Otsu thresholding.
    _, binary = cv2.threshold(
        corrected,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    candidates_by_polarity = []

    image_area = float(height * width)

    # Try bright and dark foreground interpretations.
    for candidate_mask in (
        binary,
        cv2.bitwise_not(binary)
    ):

        cleaned = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_OPEN,
            KERNEL,
            iterations=1
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            KERNEL,
            iterations=2
        )

        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        plausible = []

        for contour in contours:

            area = cv2.contourArea(contour)
            area_ratio = area / image_area

            if not MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO:
                continue

            _, _, box_width, box_height = (
                cv2.boundingRect(contour)
            )

            short_side = max(
                1,
                min(box_width, box_height)
            )

            aspect_ratio = (
                max(box_width, box_height)
                / short_side
            )

            if aspect_ratio > MAX_ASPECT_RATIO:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter > 0:
                circularity = (
                    4.0 * np.pi * area
                    / (perimeter * perimeter)
                )
            else:
                circularity = 0.0

            if circularity < MIN_CIRCULARITY:
                continue

            score = area * (
                0.5 + min(circularity, 1.0)
            )

            if (
                PENALIZE_BORDER_TOUCH
                and touches_border(
                    contour,
                    width,
                    height
                )
            ):
                score *= BORDER_PENALTY

            plausible.append(
                (score, contour)
            )

        if plausible:
            candidates_by_polarity.append(
                max(
                    plausible,
                    key=lambda item: item[0]
                )
            )

    output = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # Original LAB contour-selection logic.
    if candidates_by_polarity:

        _, selected_contour = max(
            candidates_by_polarity,
            key=lambda item: item[0]
        )

        cv2.drawContours(
            output,
            [selected_contour],
            -1,
            255,
            cv2.FILLED
        )

    # --------------------------------------------------------
    # FALLBACK: only when the LAB output is completely empty.
    # --------------------------------------------------------

    used_fallback = False

    if not np.any(output):
        used_fallback = True
        output = grabcut_fallback(image)

    return output, corrected, used_fallback


# ============================================================
# EVALUATION METRICS
# ============================================================

def metrics(prediction, reference):
    """Calculate binary-mask IoU and Dice."""

    predicted = prediction > 0
    target = reference > 0

    intersection = np.logical_and(
        predicted,
        target
    ).sum()

    union = np.logical_or(
        predicted,
        target
    ).sum()

    denominator = predicted.sum() + target.sum()

    iou = (
        float(intersection / union)
        if union
        else 1.0
    )

    dice = (
        float(2 * intersection / denominator)
        if denominator
        else 1.0
    )

    return iou, dice


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    if not SPLIT_FILE.exists():
        raise SystemExit(
            f"Split file not found: {SPLIT_FILE}"
        )

    if not REFERENCE_DIR.exists():
        raise SystemExit(
            f"SAM reference directory not found: {REFERENCE_DIR}"
        )

    image_paths = read_image_paths()

    if not image_paths:
        raise SystemExit(
            f"No images found from {SPLIT_FILE}"
        )

    reference_index = build_reference_index()

    print(f"Images in split: {len(image_paths)}")
    print(
        f"SAM reference masks indexed: "
        f"{len(reference_index)}"
    )

    rows = []

    fallback_count = 0
    missing_reference_count = 0

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"Skipping unreadable image: {image_path}"
            )
            continue

        prediction, corrected, used_fallback = segment(
            image
        )

        if used_fallback:
            fallback_count += 1

        mask_path = (
            OUTPUT_DIR
            / f"{image_path.stem}_mask.png"
        )

        corrected_path = (
            OUTPUT_DIR
            / f"{image_path.stem}_corrected.png"
        )

        cv2.imwrite(
            str(mask_path),
            prediction
        )

        cv2.imwrite(
            str(corrected_path),
            corrected
        )

        reference_path = reference_index.get(
            image_path.stem
        )

        if reference_path is None:
            missing_reference_count += 1

            print(
                f"[{index}/{len(image_paths)}] "
                f"Missing reference: {image_path.name}"
            )

            continue

        reference = cv2.imread(
            str(reference_path),
            cv2.IMREAD_GRAYSCALE
        )

        if reference is None:
            missing_reference_count += 1

            print(
                f"Could not read reference: {reference_path}"
            )

            continue

        if reference.shape != prediction.shape:
            reference = cv2.resize(
                reference,
                (
                    prediction.shape[1],
                    prediction.shape[0]
                ),
                interpolation=cv2.INTER_NEAREST
            )

        iou, dice = metrics(
            prediction,
            reference
        )

        foreground_ratio = float(
            (prediction > 0).mean()
        )

        rows.append({
            "image": str(
                image_path.relative_to(ROOT)
            ),
            "iou_vs_sam": round(iou, 6),
            "dice_vs_sam": round(dice, 6),
            "predicted_foreground_ratio": round(
                foreground_ratio,
                6
            ),
            "prediction_mask": str(
                mask_path.relative_to(ROOT)
            ),
            "used_grabcut_fallback": used_fallback
        })

        fallback_label = (
            " [GrabCut fallback]"
            if used_fallback
            else ""
        )

        print(
            f"[{index}/{len(image_paths)}] "
            f"IoU={iou:.3f} "
            f"Dice={dice:.3f} "
            f"{image_path.name}"
            f"{fallback_label}"
        )

    # Save evaluation CSV.
    csv_path = OUTPUT_DIR / "metrics.csv"

    fields = [
        "image",
        "iou_vs_sam",
        "dice_vs_sam",
        "predicted_foreground_ratio",
        "prediction_mask",
        "used_grabcut_fallback"
    ]

    with csv_path.open(
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        print(
            "\nNo metrics were calculated. "
            "Check split paths and SAM reference-mask filenames."
        )

        print(f"CSV: {csv_path}")
        return

    ious = np.array(
        [row["iou_vs_sam"] for row in rows],
        dtype=float
    )

    dices = np.array(
        [row["dice_vs_sam"] for row in rows],
        dtype=float
    )

    empty_predictions = sum(
        row["predicted_foreground_ratio"] == 0
        for row in rows
    )

    print(
        f"\n=== {EXPERIMENT_NAME.upper()} (DEV SET) ==="
    )

    print(
        f"Config: channel={CHANNEL} "
        f"sigma={BACKGROUND_SIGMA} "
        f"smoothing={SMOOTHING} "
        f"median={MEDIAN_KERNEL} "
        f"border_penalty={PENALIZE_BORDER_TOUCH}"
    )

    print(f"Images evaluated:       {len(rows)}")
    print(f"Mean IoU vs SAM:        {ious.mean():.4f}")
    print(f"Median IoU vs SAM:      {np.median(ious):.4f}")
    print(f"Mean Dice vs SAM:       {dices.mean():.4f}")
    print(f"Median Dice vs SAM:     {np.median(dices):.4f}")
    print(f"Zero-IoU images:        {(ious == 0).sum()}")
    print(f"Empty predictions:      {empty_predictions}")
    print(f"GrabCut fallback calls: {fallback_count}")
    print(f"Missing references:     {missing_reference_count}")
    print(f"Metrics CSV:            {csv_path}")
    print(f"Output directory:       {OUTPUT_DIR}")


if __name__ == "__main__":
    main()