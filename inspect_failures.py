from pathlib import Path

import cv2
import numpy as np

output = Path("outputs/failure_review")
output.mkdir(parents=True, exist_ok=True)

image_paths = [
    "dataset/test/egg/not_damaged_4_jpg.rf.36c78b63ea4710d9642651ac3292fd58.jpg",
    "dataset/test/egg/not_damaged_125_jpg.rf.b19353b1f84dcbef1ec088b850927bb3.jpg",
    "dataset/train/egg/not_damaged_157_jpg.rf.b244a82199f2158510ccf485e54d98a2.jpg",
    "dataset/train/egg/not_damaged_104_jpg.rf.3bf8cb51d5ad8934390c8e2c3b79b32f.jpg",
    "dataset/train/egg/not_damaged_39_jpg.rf.81c4c633444db832268298ddae1010d6.jpg",
]

panels = []

for image_path in image_paths:
    relative = Path(image_path).relative_to("dataset")
    filename = str(relative).replace("/", "__")
    filename = Path(filename).with_suffix("").name

    image = cv2.imread(image_path)
    sam = cv2.imread(
        f"outputs/automatic_masks/{filename}_mask.png",
        cv2.IMREAD_GRAYSCALE,
    )
    classical = cv2.imread(
        f"outputs/classical_dev/{filename}_mask.png",
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None or sam is None or classical is None:
        print("Missing file for:", image_path)
        continue

    height, width = image.shape[:2]
    scale = min(320 / width, 240 / height)
    size = (int(width * scale), int(height * scale))

    original = cv2.resize(image, size)
    sam = cv2.resize(sam, size, interpolation=cv2.INTER_NEAREST)
    classical = cv2.resize(classical, size, interpolation=cv2.INTER_NEAREST)

    sam_view = cv2.cvtColor(sam, cv2.COLOR_GRAY2BGR)
    classical_view = cv2.cvtColor(classical, cv2.COLOR_GRAY2BGR)

    intersection = np.logical_and(sam > 0, classical > 0).sum()
    union = np.logical_or(sam > 0, classical > 0).sum()
    iou = intersection / union if union else 1.0

    panels.append((image_path, original, sam_view, classical_view, iou))

cell_width = 320
cell_height = 280
sheet = np.full((len(panels) * cell_height, cell_width * 3, 3), 255, dtype=np.uint8)

for i, (name, original, sam, classical, iou) in enumerate(panels):
    y = i * cell_height

    for j, (title, panel) in enumerate([
        ("Original", original),
        ("SAM mask", sam),
        ("Classical mask", classical),
    ]):
        x = j * cell_width
        h, w = panel.shape[:2]
        sheet[y + 25:y + 25 + h, x:x + w] = panel
        cv2.putText(
            sheet, title, (x + 5, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
        )

    cv2.putText(
        sheet, f"IoU: {iou:.3f}", (cell_width * 2 + 5, y + 265),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
    )

path = output / "worst_five_comparison.jpg"
cv2.imwrite(str(path), sheet)
print("Saved comparison:", path)
print("Images included:", len(panels))
