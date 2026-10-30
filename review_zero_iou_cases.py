from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path.home() / "nosh_cv_assignment"
CSV = ROOT / "outputs/trial5_grabcut_fallback/metrics.csv"
REF_DIR = ROOT / "outputs/automatic_masks"
OUT = ROOT / "outputs/zero_iou_review.jpg"

with CSV.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

failed = [r for r in rows if float(r["iou_vs_sam"]) == 0]

def find_reference(stem):
    for p in REF_DIR.glob("*_mask.png"):
        if p.name[:-len("_mask.png")].rsplit("__", 1)[-1] == stem:
            return p
    return None

def panel(image, title, size=(240, 240)):
    image = cv2.resize(image, size)
    cv2.rectangle(image, (0, 0), (size[0], 25), (30, 30, 30), -1)
    cv2.putText(image, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def mask_panel(mask, title):
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask > 0] = (0, 220, 0)
    return panel(image, title)

tiles = []

for row in failed:
    image_path = ROOT / row["image"]
    image = cv2.imread(str(image_path))
    if image is None:
        continue

    stem = image_path.stem
    ref_path = find_reference(stem)
    if ref_path is None:
        continue

    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    mask5 = cv2.imread(
        str(ROOT / row["prediction_mask"]), cv2.IMREAD_GRAYSCALE
    )

    trial8_path = (
        ROOT / "outputs/trial8_conservative_refinement/masks"
        / f"{stem}_mask.png"
    )
    mask8 = cv2.imread(str(trial8_path), cv2.IMREAD_GRAYSCALE)

    if ref is None or mask5 is None or mask8 is None:
        continue

    panels = [
        panel(image, "Original"),
        mask_panel(ref, "SAM reference"),
        mask_panel(mask5, "Trial 5"),
        mask_panel(mask8, "Trial 8"),
    ]

    line = np.hstack(panels)
    cv2.putText(
        line, stem[:70], (5, 235),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA
    )
    tiles.append(line)

if not tiles:
    raise SystemExit("No review images generated. Check mask paths.")

sheet = np.vstack(tiles)
cv2.imwrite(str(OUT), sheet)
print("Failure images:", len(tiles))
print("Saved:", OUT)
