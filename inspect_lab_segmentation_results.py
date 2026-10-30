import csv
import cv2
import numpy as np
from pathlib import Path

ROOT = Path.home() / "nosh_cv_assignment"
CSV_PATH = ROOT / "outputs/lab_illumination_trial1/metrics.csv"
REFERENCE_DIR = ROOT / "outputs/automatic_masks"
OUTPUT_DIR = ROOT / "outputs/lab_illumination_trial1/visual_review"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

rows.sort(key=lambda r: float(r["iou_vs_sam"]))

# Inspect 6 failures and 6 successes.
selected = rows[:6] + rows[-6:]

tiles = []

for row in selected:
    image_path = ROOT / row["image"]
    pred_path = ROOT / row["prediction_mask"]

    # Reference masks contain prefixes such as test__egg__.
    matches = list(
        REFERENCE_DIR.glob(f"*__{image_path.stem}_mask.png")
    )

    if not matches:
        print("Reference not found:", image_path.name)
        continue

    image = cv2.imread(str(image_path))
    reference = cv2.imread(str(matches[0]), cv2.IMREAD_GRAYSCALE)
    prediction = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

    if image is None or reference is None or prediction is None:
        print("Could not read files:", image_path.name)
        continue

    h, w = image.shape[:2]
    size = (320, 240)

    original = cv2.resize(image, size)
    ref = cv2.resize(reference, size)
    pred = cv2.resize(prediction, size)

    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    pred_rgb = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    header = np.zeros((35, 320, 3), dtype=np.uint8)
    title = f"IoU={float(row['iou_vs_sam']):.3f}"
    cv2.putText(
        header, title, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2
    )

    tiles.append(np.vstack([header, original, ref_rgb, pred_rgb]))

if tiles:
    sheet = np.hstack(tiles)
    output = OUTPUT_DIR / "lab_failure_success_review.jpg"
    cv2.imwrite(str(output), sheet)
    print("Saved:", output)
    print("Each column: IoU header, original, SAM mask, predicted mask")
else:
    print("No images could be assembled.")

