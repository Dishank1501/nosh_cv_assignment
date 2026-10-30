import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / "nosh_cv_assignment"

CSV_PATH = (
    ROOT / "outputs/trial5_grabcut_fallback/metrics.csv"
)
REFERENCE_DIR = ROOT / "outputs/automatic_masks"
OUTPUT_PATH = (
    ROOT / "outputs/trial5_grabcut_fallback/"
    "fallback_failure_review.jpg"
)

TARGETS = {
    "not_damaged_14_jpg.rf.dc96559078b03433d2b65e5055904af1",
    "not_damaged_52_jpg.rf.d62840582d63156f7e346b41ab8895d5",
    "not_damaged_119_jpg.rf.6758625c4b939166ca105d0509fc569e",
}


def add_label(image, text):
    header = np.zeros((35, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header, text, (6, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    return np.vstack([header, image])


with CSV_PATH.open(newline="") as f:
    rows = list(csv.DictReader(f))

selected = [
    row for row in rows
    if Path(row["image"]).stem in TARGETS
]

if not selected:
    raise SystemExit("No matching images found in metrics.csv")

columns = []

for row in selected:
    image_path = ROOT / row["image"]
    prediction_path = ROOT / row["prediction_mask"]

    matches = list(
        REFERENCE_DIR.glob(
            f"*__{image_path.stem}_mask.png"
        )
    )

    if not matches:
        print("Reference mask not found:", image_path.name)
        continue

    image = cv2.imread(str(image_path))
    reference = cv2.imread(
        str(matches[0]), cv2.IMREAD_GRAYSCALE
    )
    prediction = cv2.imread(
        str(prediction_path), cv2.IMREAD_GRAYSCALE
    )

    if image is None or reference is None or prediction is None:
        print("Could not read all files:", image_path.name)
        continue

    size = (320, 240)

    image = cv2.resize(image, size)
    reference = cv2.resize(
        reference, size, interpolation=cv2.INTER_NEAREST
    )
    prediction = cv2.resize(
        prediction, size, interpolation=cv2.INTER_NEAREST
    )

    reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

    score = float(row["iou_vs_sam"])
    columns.append(np.vstack([
        add_label(image, "Original"),
        add_label(reference, "SAM reference"),
        add_label(
            prediction,
            f"Hybrid prediction | IoU={score:.3f}"
        ),
    ]))

if columns:
    sheet = np.hstack(columns)
    cv2.imwrite(str(OUTPUT_PATH), sheet)
    print("Saved:", OUTPUT_PATH)
else:
    print("No comparison images could be created.")

