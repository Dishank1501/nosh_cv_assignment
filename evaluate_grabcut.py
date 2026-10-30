from pathlib import Path

import cv2
import numpy as np

split_file = Path("splits/eval.txt")
sam_dir = Path("outputs/automatic_masks")
pred_dir = Path("outputs/grabcut_eval")

pred_dir.mkdir(parents=True, exist_ok=True)

# Import the same segmentation function used during development.
from grabcut_segmentation import segment_egg

paths = [
    line.strip()
    for line in split_file.read_text().splitlines()
    if line.strip()
]

results = []

for index, image_path in enumerate(paths, start=1):
    image = cv2.imread(image_path)

    if image is None:
        print("Could not read:", image_path)
        continue

    relative = Path(image_path).relative_to("dataset")
    filename = str(relative).replace("/", "__")
    filename = Path(filename).with_suffix("").name

    sam_path = sam_dir / f"{filename}_mask.png"
    sam = cv2.imread(str(sam_path), cv2.IMREAD_GRAYSCALE)

    if sam is None:
        print("Missing SAM mask:", sam_path)
        continue

    prediction = segment_egg(image)

    cv2.imwrite(
        str(pred_dir / f"{filename}_mask.png"),
        prediction,
    )

    reference = sam > 0
    predicted = prediction > 0

    intersection = np.logical_and(reference, predicted).sum()
    union = np.logical_or(reference, predicted).sum()
    total = reference.sum() + predicted.sum()

    iou = intersection / union if union else 1.0
    dice = 2 * intersection / total if total else 1.0

    results.append((image_path, iou, dice))
    print(f"[{index}/{len(paths)}] IoU={iou:.3f}, Dice={dice:.3f}")

if not results:
    raise RuntimeError("No evaluation images were processed.")

ious = np.array([row[1] for row in results])
dices = np.array([row[2] for row in results])

print("\n=== HELD-OUT EVALUATION RESULTS ===")
print("Images evaluated:", len(results))
print(f"Mean IoU: {ious.mean():.4f}")
print(f"Median IoU: {np.median(ious):.4f}")
print(f"Mean Dice: {dices.mean():.4f}")
print(f"Median Dice: {np.median(dices):.4f}")
print("Zero-IoU images:", int((ious == 0).sum()))
print("Masks saved to:", pred_dir)
