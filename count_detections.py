import csv
import cv2
import numpy as np
from pathlib import Path

ROOT = Path.home() / "nosh_cv_assignment"
CSV_PATH = ROOT / "outputs/trial5_grabcut_fallback/metrics.csv"
SAM_DIR = ROOT / "outputs/automatic_masks"
OUT_DIR = ROOT / "outputs/detection_review"

OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))


def find_sam_mask(image_path):
    matches = list(
        SAM_DIR.glob(f"*__{image_path.stem}_mask.png")
    )
    return matches[0] if matches else None


def overlay_mask(image, mask):
    result = image.copy()
    colored = np.zeros_like(image)
    colored[:, :, 1] = 255

    selected = mask > 0
    result[selected] = (
        0.5 * result[selected] +
        0.5 * colored[selected]
    ).astype(np.uint8)

    return result


tile_width = 240
tile_height = 290
batch_size = 8
tiles = []

for i, row in enumerate(rows):
    image_path = ROOT / row["image"]
    hybrid_path = ROOT / row["prediction_mask"]
    sam_path = find_sam_mask(image_path)

    if sam_path is None:
        print("Missing SAM mask:", image_path.name)
        continue

    image = cv2.imread(str(image_path))
    sam = cv2.imread(str(sam_path), cv2.IMREAD_GRAYSCALE)
    hybrid = cv2.imread(str(hybrid_path), cv2.IMREAD_GRAYSCALE)

    if image is None or sam is None or hybrid is None:
        print("Unreadable file:", image_path.name)
        continue

    image = cv2.resize(image, (tile_width, tile_width))
    sam = cv2.resize(sam, (tile_width, tile_width),
                     interpolation=cv2.INTER_NEAREST)
    hybrid = cv2.resize(hybrid, (tile_width, tile_width),
                        interpolation=cv2.INTER_NEAREST)

    views = [
        ("Original", image),
        ("SAM", overlay_mask(image, sam)),
        ("Hybrid", overlay_mask(image, hybrid)),
    ]

    panels = []
    for label, view in views:
        header = np.zeros((30, tile_width, 3), dtype=np.uint8)
        cv2.putText(header, label, (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        panels.append(np.vstack([header, view]))

    tile = np.hstack(panels)

    # Include index and IoU for identifying the image later.
    footer = np.zeros((30, tile.shape[1], 3), dtype=np.uint8)
    text = f"#{i+1} IoU={float(row['iou_vs_sam']):.2f} {image_path.name[:28]}"
    cv2.putText(footer, text, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)

    tiles.append(np.vstack([footer, tile]))

for start in range(0, len(tiles), batch_size):
    batch = tiles[start:start + batch_size]
    sheet = np.vstack(batch)

    output = OUT_DIR / f"review_{start+1:03d}_{start+len(batch):03d}.jpg"
    cv2.imwrite(str(output), sheet)
    print("Saved:", output)

print(f"\nPrepared visual reviews for {len(tiles)} images.")
